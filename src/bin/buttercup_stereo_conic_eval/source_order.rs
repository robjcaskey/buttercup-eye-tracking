//! Replay immutable native evidence through the SAME bounded live tracker.
//!
//! Arrival stress preserves source timestamps, clocks, crop geometry and RAW.
//! Explicit pupil ablation removes only pupil evidence; all-boundary-samples
//! mode fits every retained point and publishes no withheld probes. A previous joint
//! target is a seed, as in live operation; no independent gaze average is used.
//! This is not SAM video-memory replay or an attested detector-latency trace.

use super::*;
use joint_tracking::{FrameEvidence,JointTracker,TrackingUnavailable};
use binocular_coordinator::source_pairing::PairingUnavailable;
use std::collections::{HashSet,VecDeque};

#[derive(Clone,Debug)]
struct CachePosition {
    index:u64,
    source:ExposureKey,
    file:usize,
    offset:u64,
}

fn index_cache(reader:&mut (impl BufRead+Seek),file:usize,limit:usize,
    clocks:&mut HashMap<u64,String>)->Result<Vec<CachePosition>,String> {
    let mut positions=Vec::new();
    let mut line=String::new();
    for _ in 0..limit {
        let offset=reader.stream_position().map_err(|e|e.to_string())?;
        line.clear();
        if reader.read_line(&mut line).map_err(|e|e.to_string())?==0 {break;}
        let row:Value=serde_json::from_str(&line).map_err(|e|format!("cache {file} offset {offset}: {e}"))?;
        let input=&row["input"];
        let frame=&input["frame"];
        let eye=integer(frame,"eye_id")?;
        if !(1..=2).contains(&eye) {return Err("invalid ROI identity".into());}
        let lineage=input["clock_lineage"].as_str().ok_or("missing clock lineage")?;
        let epoch=hash(lineage);
        match clocks.get(&epoch) {
            Some(previous) if previous!=lineage=>return Err("source clock hash collision".into()),
            Some(_)=>{},
            None=>{clocks.insert(epoch,lineage.to_owned());},
        }
        positions.push(CachePosition {index:integer(input,"index")?,file,offset,
            source:ExposureKey {roi:RoiId(eye as u32),clock:SourceClock {domain:1,epoch},
                sequence:integer(frame,"sequence")?,timestamp_ns:integer(frame,"timestamp_ns")?}});
    }
    Ok(positions)
}

fn schedule(positions:&mut [CachePosition],delay:[u64;2])->Result<(),String> {
    let mut indices=HashSet::new();
    let mut sources=HashSet::new();
    for p in positions.iter() {
        if !indices.insert(p.index) || !sources.insert((p.source.clock,p.source.roi,p.source.timestamp_ns)) {
            return Err("duplicate or conflicting native source receipt in replay".into());
        }
        p.source.timestamp_ns.checked_add(delay[p.source.roi.0 as usize-1])
            .ok_or("arrival delay overflows logical timestamp")?;
    }
    // No timing comparison crosses a source lineage. Eye-local sequence
    // numbers are not shared sensor clocks and do not control pair formation.
    positions.sort_by_key(|p|(p.source.clock.epoch,
        p.source.timestamp_ns+delay[p.source.roi.0 as usize-1],p.source.roi.0,p.index));
    Ok(())
}

fn source_json(source:ExposureKey)->Value {
    json!({"roi_id":source.roi.0,"clock_domain":source.clock.domain.to_string(),
        "clock_epoch":source.clock.epoch.to_string(),"sequence":source.sequence.to_string(),
        "timestamp_ns":source.timestamp_ns.to_string()})
}

pub(super) fn run(files:&[String],limit:usize,extraction:ExtractionPolicy,delay:[u64;2],export_hypotheses:bool,
    probabilistic:bool,writer:&mut impl Write)->Result<(),String> {
    run_with_tracker(files,limit,extraction,delay,export_hypotheses,probabilistic,writer,|_,_|{})
}

#[cfg(test)]
pub(super) fn run_with_supported_mode_diagnostic(files:&[String],extraction:ExtractionPolicy,
    config:conic_solver::joint::posterior::IntegrationConfig,writer:&mut impl Write)->Result<(),String> {
    run_with_tracker(files,usize::MAX,extraction,[0;2],true,true,writer,
        |tracker,_|tracker.set_posterior_diagnostic(config))
}

#[cfg(test)]
pub(super) fn run_with_posterior_diagnostic(files:&[String],config:conic_solver::joint::posterior::IntegrationConfig,
    writer:&mut impl Write)->Result<(),String> {
    run_with_tracker(files,usize::MAX,ExtractionPolicy {all_boundary_samples:true,..Default::default()},
        [0;2],false,true,writer,|tracker,_|tracker.set_posterior_diagnostic(config))
}

#[cfg(test)]
pub(super) fn run_with_selected_posterior_diagnostic(files:&[String],config:conic_solver::joint::posterior::IntegrationConfig,
    selected:&std::collections::HashSet<u64>,writer:&mut impl Write)->Result<(),String> {
    run_with_tracker(files,usize::MAX,ExtractionPolicy {all_boundary_samples:true,..Default::default()},
        [0;2],false,true,writer,|tracker,frame| {
            let mut current=config;
            if !selected.contains(&integer(&frame.input,"index").unwrap()) {current.budget=0;}
            tracker.set_posterior_diagnostic(current);
        })
}

#[cfg(test)]
pub(super) fn run_with_mask_posterior_diagnostic(files:&[String],mask_levels:bool,mask_spatial:bool,
    config:conic_solver::joint::posterior::IntegrationConfig,writer:&mut impl Write)->Result<(),String> {
    let selected = diagnostic_sources("BUTTERCUP_MASK_REFERENCE_SOURCES")?;
    if (config.annealed_reference.is_some() || config.populations.is_some()) && selected.is_empty() {
        return Err("mask reference requires an explicit nonempty native source selection".into());
    }
    let raw_outer_candidates=std::env::var("BUTTERCUP_RAW_OUTER_CANDIDATES").ok().as_deref()==Some("1");
    let withhold_rejected_outer=std::env::var("BUTTERCUP_WITHHOLD_REJECTED_OUTER").ok().as_deref()==Some("1");
    run_with_tracker(files,usize::MAX,ExtractionPolicy {mask_levels,mask_spatial,raw_outer_candidates,withhold_rejected_outer,all_boundary_samples:true,..Default::default()},
        [0;2],true,true,writer,|tracker,frame| {
            let mut current=config;
            let source=frame.packet.exposure;
            if !selected.contains(&(frame.input["clock_lineage"].as_str().unwrap().to_owned(),source.timestamp_ns))
                || source.roi.0 != 2 {current.annealed_reference=None;current.populations=None;}
            tracker.set_posterior_diagnostic(current);
        })
}

#[cfg(test)]
pub(super) fn run_with_selected_annealed_diagnostic(files:&[String],config:conic_solver::joint::posterior::IntegrationConfig,
    selected:&std::collections::HashSet<u64>,writer:&mut impl Write)->Result<(),String> {
    assert!(config.annealed_reference.is_some());
    run_with_tracker(files,usize::MAX,ExtractionPolicy {all_boundary_samples:true,..Default::default()},
        [0;2],false,true,writer,|tracker,frame| {
            let mut current=config;
            // All ordinary posterior/geometry work remains identical, including
            // unselected history. Only the independent side diagnostic is gated.
            if !selected.contains(&integer(&frame.input,"index").unwrap()) {current.annealed_reference=None;}
            tracker.set_posterior_diagnostic(current);
        })
}

fn run_with_tracker(files:&[String],limit:usize,extraction:ExtractionPolicy,delay:[u64;2],export_hypotheses:bool,
    probabilistic:bool,writer:&mut impl Write,mut configure:impl FnMut(&mut JointTracker,&Frame))->Result<(),String> {
    let camera_mount=eye_scene_model::CameraMount::for_offline_checks()?;
    #[cfg(test)]
    let counterfactual_sources = diagnostic_sources("BUTTERCUP_PUPIL_COUNTERFACTUAL_SOURCES")?;
    #[cfg(test)]
    let sensitivity_sources = diagnostic_sources("BUTTERCUP_SCENE_SENSITIVITY_SOURCES")?;
    #[cfg(test)]
    let association_sources = diagnostic_sources("BUTTERCUP_ASSOCIATION_DIAGNOSTIC_SOURCES")?;
    let mut readers=files.iter().map(|path|File::open(path).map(BufReader::new).map_err(|e|e.to_string()))
        .collect::<Result<Vec<_>,_>>()?;
    let mut clocks=HashMap::new();
    let mut positions=Vec::new();
    for (file,reader) in readers.iter_mut().enumerate() {
        positions.extend(index_cache(reader,file,limit,&mut clocks)?);
    }
    schedule(&mut positions,delay)?;
    eprintln!("source replay indexed exposures={} lineages={} arrival_delay_ns={delay:?}",positions.len(),clocks.len());
    // The disk-offset index is offline-only. Actual retained evidence mirrors
    // the live pairer's 32 entries per ROI / 1.5-second source-time envelope.
    let mut retained:[VecDeque<Arc<Frame>>;2]=std::array::from_fn(|_|VecDeque::new());
    let mut previous:[Option<(u64,[u32;2],[u32;2])>;2]=[None;2];
    let mut tracker=JointTracker::default();
    tracker.camera_mount=camera_mount;
    tracker.retain_diagnostic_hypotheses(export_hypotheses);
    tracker.set_probabilistic(probabilistic);
    let mut clock=None;
    let mut generation=0;
    let mut newest=0u64;
    let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
    let mut line=String::new();
    for (event,position) in positions.into_iter().enumerate() {
        let reader=&mut readers[position.file];
        reader.seek(SeekFrom::Start(position.offset)).map_err(|e|e.to_string())?;
        line.clear();reader.read_line(&mut line).map_err(|e|e.to_string())?;
        let frame=Arc::new(prepare_with_policy(serde_json::from_str(&line).map_err(|e|e.to_string())?,extraction)?);
        configure(&mut tracker,&frame);
        let source=frame.packet.exposure;
        if source!=position.source || integer(&frame.input,"index")?!=position.index {
            return Err("cached source changed between indexing and replay".into());
        }
        let eye=source.roi.0 as usize-1;
        if clock!=Some(source.clock) {
            clock=Some(source.clock);generation+=1;newest=0;
            retained.iter_mut().for_each(VecDeque::clear);previous=[None;2];
            tracker.begin(source.clock,generation);
        }
        let reframe=previous[eye].is_some_and(|(time,origin,size)|source.timestamp_ns>time
            && source.timestamp_ns-time<=500_000_000 && size==frame.packet.dimensions_px
            && origin!=frame.packet.sensor_origin_px);
        previous[eye]=Some((source.timestamp_ns,frame.packet.sensor_origin_px,frame.packet.dimensions_px));
        newest=newest.max(source.timestamp_ns);
        retained[eye].push_back(Arc::clone(&frame));
        for rows in &mut retained {
            rows.retain(|f|newest.saturating_sub(f.packet.exposure.timestamp_ns)<=1_500_000_000);
            while rows.len()>32 {rows.pop_front();}
        }
        let packet=||FrameEvidence {packet:frame.packet.clone(),pose:frame.pose};
        let started=Instant::now();
        let result=tracker.observe(packet(),camera);
        let elapsed=started.elapsed().as_secs_f64()*1000.0;
        let mut output=json!({"schema":"buttercup-joint-source-replay-v1","event":event,"input":frame.input,
            "camera_mount_assumption":camera_mount.label(),
            "pupil_ablation":frame.pupil_ablation,
            "all_boundary_samples":extraction.all_boundary_samples,
            "coherent_pupil_arcs":extraction.coherent_pupil_arcs,
            "shape_pupil_arcs":extraction.shape_pupil_arcs,
            "optical_pupil_arcs":extraction.optical_pupil_arcs,
            "retained_outline_direction_experiment":extraction.outline_directions,
            "generation":generation,"arrival_delay_ns":delay.map(|v|v.to_string()),
            "logical_arrival_timestamp_ns":(source.timestamp_ns+delay[eye]).to_string(),
            "source_now_ns":newest.to_string(),"native_roi_reframe":reframe,
            "contract":"Fresh native evidence through JointTracker. Delays are synthetic scheduling stress, not measured latency. Repeated publications are not additional RAW exposures."});
        if let Some(report)=&frame.pupil_directions {output["pupil_raw_directions"]=json!(report);}
        if frame.conic_pupil_paths {output["conic_pupil_paths"]=json!(true);}
        if frame.augmented_pupil_paths {output["augmented_pupil_paths"]=json!(true);}
        if frame.pupil_profile_footprint {output["pupil_profile_footprint"]=json!(true);}
        if frame.subpixel_pupil_peaks {output["subpixel_pupil_peaks"]=json!(true);}
        if frame.connected_pupil_width {output["connected_pupil_width"]=json!(true);}
        if frame.reject_weak_pupil_core {output["pupil_core_support"]=json!(frame.pupil_core_support);}
        if let Some(report)=&frame.semantic_pupil_contour {output["semantic_pupil_contour"]=report.clone();}
        if let Some(radius)=frame.pupil_search_radius_px {output["pupil_search_radius_px"]=json!(radius);}
        if frame.sliding_pupil_luma {output["sliding_pupil_luma"]=json!(true);}
        if let Some(scale)=frame.pupil_weight_scale {output["pupil_weight_scale"]=json!(scale);}
        if let Some(report)=&frame.outer_spread {output["outer_raw_spread"]=json!(report);}
        if let Some(report)=&frame.outer_position {output["outer_raw_position"]=json!(report);}
        if let Some(report)=&frame.mask_levels {output["mask_boundary_profiles"]=report.clone();}
        if let Some(report)=&frame.outer_candidates {output["raw_outer_candidates"]=report.clone();}
        if let Some(report)=&frame.rejected_outer_ablation {output["rejected_outer_ablation"]=report.clone();}
        let outside=matches!(result,Err(TrackingUnavailable::Pairing(PairingUnavailable::OutsideSourceWindow)));
        match result {
            Ok(Some(publication))=>{
                let frames=std::array::from_fn::<_,2,_>(|eye|publication.exposures[eye]
                    .and_then(|key|retained[eye].iter().find(|f|f.packet.exposure==key)).map(Arc::as_ref));
                if publication.exposures.iter().zip(frames).any(|(key,f)|key.is_some()!=f.is_some()) {
                    return Err("publication lacks exact retained RAW/source provenance".into());
                }
                if publication.exposures.iter().flatten().any(|key|key.clock!=source.clock || key.timestamp_ns!=source.timestamp_ns) {
                    return Err("live tracker paired different native source exposures".into());
                }
                output["publication_inputs"]=json!(frames.map(|f|f.map(|f|&f.input)));
                output["joint"]=solution_json(Ok(publication.solution.clone()),frames,elapsed);
                if export_hypotheses {
                    output["current_source_hypotheses"]=json!(publication.diagnostic_hypotheses.iter()
                        .map(|solution|solution_json(Ok(solution.clone()),frames,elapsed)).collect::<Vec<_>>());
                    #[cfg(test)]
                    for (row, solution) in output["current_source_hypotheses"].as_array_mut().unwrap()
                        .iter_mut().zip(&publication.diagnostic_hypotheses) {
                        row["native_factor_costs"] = json!(solution.factor_costs);
                    }
                }
                #[cfg(test)]
                if frames.iter().all(Option::is_some) && counterfactual_sources.contains(&(
                    frame.input["clock_lineage"].as_str().unwrap().to_owned(),source.timestamp_ns)) {
                    output["pupil_counterfactuals"] = current_pupil_counterfactuals(
                        frames,&publication.scene.prior,&publication.solution)?;
                }
                #[cfg(test)]
                if frames.iter().all(Option::is_some) && sensitivity_sources.contains(&(
                    frame.input["clock_lineage"].as_str().unwrap().to_owned(),source.timestamp_ns)) {
                    output["scene_sensitivity"] = current_scene_sensitivity(
                        frames,&publication.scene.prior,&publication.solution)?;
                }
                #[cfg(test)]
                if frames.iter().all(Option::is_some) && association_sources.contains(&(
                    frame.input["clock_lineage"].as_str().unwrap().to_owned(),source.timestamp_ns)) {
                    output["association_conditionals"] = current_association_conditionals(
                        frames,&publication.scene.prior,&publication.solution)?;
                }
            },
            Ok(None)=>return Err("unique source silently treated as duplicate".into()),
            Err(error)=>output["joint"]=json!({"available":false,"reason":format!("{error:?}"),"elapsed_ms":elapsed}),
        }
        let latest=std::array::from_fn::<_,2,_>(|eye|tracker.latest(eye,source.clock,newest,500_000_000));
        for eye in 0..2 {
            if latest[eye].as_ref().is_some_and(|p|p.exposures[eye].is_some_and(|key|
                previous[eye].is_some_and(|(time,_,_)|key.timestamp_ns<time))) {
                return Err(format!("source {} restored eye {} geometry older than its latest native observation",position.index,eye+1));
            }
        }
        if !outside {
            if !matches!(tracker.observe(packet(),camera),Ok(None)) {return Err("held source re-entered the solver".into());}
            for eye in 0..2 {
                let after=tracker.latest(eye,source.clock,newest,500_000_000);
                if !match (&latest[eye],&after) {(Some(a),Some(b))=>Arc::ptr_eq(a,b),(None,None)=>true,_=>false} {
                    return Err("duplicate publication changed latest evidence".into());
                }
            }
        }
        output["duplicate_suppressed"]=json!(!outside);
        output["latest"]=json!(std::array::from_fn::<_,2,_>(|eye|latest[eye].as_ref().map(|p|json!({
            "source":p.exposures[eye].map(source_json),"contributing":p.solution.contributing_eyes[eye],
            "target_camera_mm":p.solution.target_camera_mm}))));
        serde_json::to_writer(&mut *writer,&output).map_err(|e|e.to_string())?;
        writer.write_all(b"\n").map_err(|e|e.to_string())?;
        if (event+1)%500==0 {writer.flush().map_err(|e|e.to_string())?;eprintln!("source replay events={}",event+1);}
    }
    writer.flush().map_err(|e|e.to_string())?;
    eprintln!("source replay complete");
    Ok(())
}

#[cfg(test)]
fn diagnostic_sources(variable:&str)->Result<HashSet<(String,u64)>,String> {
    std::env::var(variable).ok().map(|path| -> Result<_,String> {
        let rows:Value=serde_json::from_reader(File::open(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        rows.as_array().ok_or("expected native source selection array".to_string())?.iter()
            .map(|row|Ok((row["clock_lineage"].as_str().ok_or("missing lineage")?.to_owned(),
                integer(row,"timestamp_ns")?))).collect()
    }).transpose().map(|selection|selection.unwrap_or_default())
}

#[cfg(test)]
fn current_association_conditionals(frames:[Option<&Frame>;2],scene:&JointScenePrior,
    published:&JointConicSolution)->Result<Value,String> {
    // Each counterfactual still has one shared fixation. These conditional
    // fits never enter the tracker, its source history, or the live display.
    // Preserve both acquired pose priors and both source-keyed target starts;
    // withholding observations must not silently reestimate the coarse scene.
    let prepared=frames.map(|f|f.map(|f|f.packet.prepare()));
    let evidence=prepared.each_ref().map(|p|p.as_ref().map(|p|p.evidence()));
    let full_request=JointConicRequest {eyes:evidence,scene,maximum_hypotheses:16,maximum_refinements:12,
        maximum_source_skew_ns:0,exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
    let config=posterior::IntegrationConfig {mask_state_proposals:true,mask_state_refinement:true,
        ..posterior::IntegrationConfig::live()};
    let mut variants=Vec::new();
    for (name,mask) in [("control",[true,true]),("roi1-only",[true,false]),("roi2-only",[false,true])] {
        let omitted=conic_solver::joint::association_omission_cost_diagnostic(full_request,mask)
            .map_err(|e|format!("full-source omission attribution failed: {e:?}"))?;
        let request=JointConicRequest {eyes:std::array::from_fn(|eye|if mask[eye] {evidence[eye]} else {None}),
            ..full_request};
        let result=solve_joint_conic_distribution_diagnostic(request,4,config);
        let mut variant=json!({"name":name,"observation_eyes":mask,"omitted_roi_capped_cost":omitted});
        match result {
            Err(error)=>variant["solution"]=json!({"available":false,"reason":format!("{error:?}")}),
            Ok(solutions)=>{
                let best=&solutions[0];
                if name=="control" {
                    assert_eq!(best.target_camera_mm,published.target_camera_mm);
                    assert_eq!(best.eye_gaze_directions,published.eye_gaze_directions);
                    assert_eq!(best.ellipses_roi_px,published.ellipses_roi_px);
                    assert_eq!(best.robust_cost,published.robust_cost);
                    assert_eq!(best.posterior.as_ref().map(|p|p.json()),published.posterior.as_ref().map(|p|p.json()));
                }
                let comparable=best.robust_cost+omitted.iter().sum::<f64>();
                variant["cost_with_omission_penalty"]=json!(comparable);
                variant["cost_delta_from_published"]=json!(comparable-published.robust_cost);
                variant["angular_change_from_published_degrees"]=json!(std::array::from_fn::<_,2,_>(|eye|
                    best.eye_gaze_directions[eye].zip(published.eye_gaze_directions[eye]).map(|(a,b)|
                        a.into_iter().zip(b).map(|(x,y)|x*y).sum::<f64>().clamp(-1.0,1.0).acos().to_degrees())));
                variant["solution"]=solution_json(Ok(best.clone()),frames,0.0);
                variant["hypotheses"]=json!(solutions.iter().map(|s|json!({
                    "modeled_eyes":s.modeled_eyes,"target_camera_mm":s.target_camera_mm,
                    "cost_with_omission_penalty":s.robust_cost+omitted.iter().sum::<f64>()})).collect::<Vec<_>>());
            }
        }
        variants.push(variant);
    }
    Ok(json!({"contract":"Separate conditional solves at the same actual source and unchanged coarse scene/history starts. Every hypothesis has one fixation. Omission costs are the native MAP engineering penalties, not normalized association evidence; different nuisance dimensions cannot be mixed using these cost gaps. No conditional output updates source history or presentation.",
        "variants":variants}))
}

#[cfg(test)]
fn current_scene_sensitivity(frames:[Option<&Frame>;2],scene:&JointScenePrior,
    published:&JointConicSolution)->Result<Value,String> {
    let mut variants=Vec::new();
    // Hypothetical sensitivity probes, deliberately not camera confidence
    // bounds or new live priors. No model average is constructed from them.
    for (name,focal_scale,principal_delta,outer_scale,alignment_scale) in [
        ("control",[1.,1.],[0.,0.],1.,1.),
        ("focal-minus-10pct",[0.9,0.9],[0.,0.],1.,1.),
        ("focal-plus-10pct",[1.1,1.1],[0.,0.],1.,1.),
        ("principal-x-minus-80px",[1.,1.],[-80.,0.],1.,1.),
        ("principal-x-plus-80px",[1.,1.],[80.,0.],1.,1.),
        ("principal-y-minus-60px",[1.,1.],[0.,-60.],1.,1.),
        ("principal-y-plus-60px",[1.,1.],[0.,60.],1.,1.),
        ("focal-aspect-x-plus-y-minus-5pct",[1.05,0.95],[0.,0.],1.,1.),
        ("focal-aspect-x-minus-y-plus-5pct",[0.95,1.05],[0.,0.],1.,1.),
        ("outer-allowance-times-1.5",[1.,1.],[0.,0.],1.5,1.),
        ("outer-allowance-times-2",[1.,1.],[0.,0.],2.,1.),
        ("axis-alignment-sigma-times-1.5",[1.,1.],[0.,0.],1.,1.5),
        ("axis-alignment-sigma-times-2",[1.,1.],[0.,0.],1.,2.),
    ] {
        let camera=PinholeCamera {focal_px:std::array::from_fn(|i|scene.camera.focal_px[i]*focal_scale[i]),
            principal_px:std::array::from_fn(|i|scene.camera.principal_px[i]+principal_delta[i])};
        let poses=frames.map(|f|f.map(|f|f.pose));
        let mut candidate=approximate_scene(camera,poses).ok_or("sensitivity camera lacks scene support")?.prior;
        candidate.target_seed_camera_mm=scene.target_seed_camera_mm;
        candidate.secondary_target_seed_camera_mm=scene.secondary_target_seed_camera_mm;
        for prior in candidate.eyes.iter_mut().flatten() {
            if let Some(alignment)=prior.surface_axis_alignment.as_mut() {
                alignment.sigma_radians=alignment.sigma_radians.map(|s|s*alignment_scale);
            }
        }
        let mut packets=frames.map(|f|f.map(|f|f.packet.clone()));
        if outer_scale != 1.0 {
            for packet in packets.iter_mut().flatten() {
                let detail=packet.detail_reliability.filter(|d|d.is_finite()).unwrap_or(0.25).clamp(0.,1.);
                let optical_sigma=0.75_f64.hypot(4.0*(1.0-detail));
                for arc in &mut packet.arcs {
                    if arc.kind == BoundaryKind::OuterLimbus {
                        arc.localization_sigma_px=Some(arc.localization_sigma_px.unwrap_or(optical_sigma)*outer_scale);
                        arc.normal_band_half_width_px*=outer_scale;
                    }
                }
            }
        }
        let prepared=packets.each_ref().map(|p|p.as_ref().map(|p|p.prepare()));
        let evidence=prepared.each_ref().map(|p|p.as_ref().map(|p|p.evidence()));
        let request=JointConicRequest {eyes:evidence,scene:&candidate,maximum_hypotheses:16,maximum_refinements:12,
            maximum_source_skew_ns:0,exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
        let result=solve_joint_conic_distribution_diagnostic(request,1,posterior::IntegrationConfig::live());
        let value=match result {
            Err(error)=>json!({"available":false,"reason":format!("{error:?}")}),
            Ok(solutions)=>{
                let best=&solutions[0];
                if name=="control" {
                    assert_eq!(best.target_camera_mm,published.target_camera_mm);
                    assert_eq!(best.eye_gaze_directions,published.eye_gaze_directions);
                    assert_eq!(best.robust_cost,published.robust_cost);
                }
                let mut row=solution_json(Ok(best.clone()),frames,0.0);
                row["native_factor_costs"]=json!(best.factor_costs);
                row
            }
        };
        variants.push(json!({"name":name,"camera":{"focal_px":camera.focal_px,"principal_px":camera.principal_px},
            "outer_localization_and_band_multiplier":outer_scale,"alignment_sigma_multiplier":alignment_scale,
            "result":value}));
    }
    Ok(json!({"contract":"Hypothetical current-read sensitivity, not calibrated uncertainty. Original source-native boundary coordinates, RAW, scale and prior source seeds are fixed; camera-dependent metric scene initialization is recomputed. Bounds and all pupil factors remain fixed. No variant updates history or becomes a live model average.","variants":variants}))
}

#[cfg(test)]
fn current_pupil_counterfactuals(frames:[Option<&Frame>;2],scene:&JointScenePrior,
    published:&JointConicSolution)->Result<Value,String> {
    let mut cases=Vec::new();
    for removed in [[false,false],[true,false],[false,true],[true,true]] {
        let mut packets=frames.map(|f|f.map(|f|f.packet.clone()));
        let mut receipts=[Value::Null,Value::Null];
        for eye in 0..2 {
            if removed[eye] {
                receipts[eye]=packets[eye].as_mut().map_or(Value::Null,remove_pupil_evidence);
            }
        }
        let prepared=packets.each_ref().map(|p|p.as_ref().map(|p|p.prepare()));
        let evidence=prepared.each_ref().map(|p|p.as_ref().map(|p|p.evidence()));
        let request=JointConicRequest {eyes:evidence,scene,maximum_hypotheses:16,maximum_refinements:12,
            maximum_source_skew_ns:0,exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
        let result=solve_joint_conic_distribution_diagnostic(request,4,posterior::IntegrationConfig::live());
        let value=match result {
            Err(error)=>json!({"available":false,"reason":format!("{error:?}")}),
            Ok(solutions)=>{
                if removed == [false,false] {
                    assert_eq!(solutions[0].target_camera_mm,published.target_camera_mm);
                    assert_eq!(solutions[0].eye_gaze_directions,published.eye_gaze_directions);
                    assert_eq!(solutions[0].robust_cost,published.robust_cost);
                }
                json!({"available":true,"solutions":solutions.iter().map(|s| {
                    let mut row=solution_json(Ok(s.clone()),frames,0.0);
                    row["native_factor_costs"]=json!(s.factor_costs);
                    row
                }).collect::<Vec<_>>()})
            }
        };
        cases.push(json!({"removed_pupil_eyes":removed,"removal_receipts":receipts,"result":value}));
    }
    Ok(json!({"contract":"Current-read causal ablation with the original full-evidence scene priors and preceding-source seeds frozen. Remove one/both eyes' pupil samples and pupil initialization hints after shared extraction. Counterfactual results never update live history, never add pixels, and are not measured gaze truth.",
        "cases":cases}))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(index:u64,eye:u32,time:u64,epoch:u64)->CachePosition {
        CachePosition {index,file:0,offset:index*50,source:ExposureKey {roi:RoiId(eye),
            sequence:index+1000,timestamp_ns:time,clock:SourceClock {domain:1,epoch}}}
    }
    #[test]
    fn delayed_arrivals_change_order_but_not_the_native_sensor_read() {
        let original=vec![entry(1,1,100,1),entry(2,2,100,1),entry(3,1,200,1),entry(4,2,200,1)];
        let mut events=original.clone();schedule(&mut events,[0,150]).unwrap();
        assert_eq!(events.iter().map(|e|e.index).collect::<Vec<_>>(),[1,3,2,4]);
        for event in events {assert_eq!(event.source,original.iter().find(|e|e.index==event.index).unwrap().source);}
    }
    #[test]
    fn unrelated_epochs_never_interleave_and_conflicting_receipts_fail() {
        let mut events=vec![entry(1,1,100,1),entry(2,2,100,1),entry(3,1,1,2),entry(4,2,1,2)];
        schedule(&mut events,[200,0]).unwrap();
        assert_eq!(events.iter().map(|e|e.source.clock.epoch).collect::<Vec<_>>(),[1,1,2,2]);
        events.push(entry(5,1,100,1));
        assert!(schedule(&mut events,[0;2]).is_err());
        assert!(schedule(&mut [entry(1,1,u64::MAX,1)],[1,0]).is_err());
    }
    #[test]
    fn disk_offsets_recover_the_exact_indexed_source_without_loading_all_contours() {
        let lines=[(11,1,100),(12,2,100),(13,1,200)].map(|(index,eye_id,time)|json!({"input":{
            "index":index,"clock_lineage":"actual-lineage","frame":{"eye_id":eye_id,
            "sequence":index+1000,"timestamp_ns":time}}}).to_string()+"\n").concat();
        let mut reader=std::io::Cursor::new(lines.as_bytes());
        let positions=index_cache(&mut reader,2,2,&mut HashMap::new()).unwrap();
        assert_eq!(positions.len(),2);
        for p in positions {
            reader.seek(SeekFrom::Start(p.offset)).unwrap();
            let mut line=String::new();reader.read_line(&mut line).unwrap();
            let row:Value=serde_json::from_str(&line).unwrap();
            assert_eq!(integer(&row["input"],"index").unwrap(),p.index);
            assert_eq!(p.file,2);
        }
    }
}
