//! Complete-corpus, bounded-memory SAM observation export. No recorded fitted
//! ellipse, gaze, target position or human label enters detector inference.
use super::*;
use std::io::{BufRead,BufReader,BufWriter,Write};

fn source_frames(input:&Path, start:usize, count:usize)
    -> Result<impl Iterator<Item=Result<(Value,Arc<sam31_outer::RawFrame>),String>>,String> {
    let mut last_file:Option<(String,File)>=None;
    let records=BufReader::new(File::open(input).map_err(|e|e.to_string())?).lines().skip(start).take(count);
    Ok(records.map(move |line| {
        let row:Value=serde_json::from_str(&line.map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let meta=&row["frame"];
        let path=row["raw_file"].as_str().ok_or("missing RAW path")?;
        if last_file.as_ref().is_none_or(|(old,_)|old!=path) {
            last_file=Some((path.to_owned(),File::open(path).map_err(|e|e.to_string())?));
        }
        let file=&mut last_file.as_mut().unwrap().1;
        file.seek(SeekFrom::Start(integer(&row,"raw_offset")?)).map_err(|e|e.to_string())?;
        let mut packed=vec![0;integer(&row,"raw_length")? as usize];
        file.read_exact(&mut packed).map_err(|e|e.to_string())?;
        let width=integer(meta,"width")? as usize;let height=integer(meta,"height")? as usize;
        let frame=Arc::new(sam31_outer::RawFrame {
            eye_index:integer(meta,"eye_id")?.checked_sub(1).filter(|eye|*eye<2).ok_or("invalid eye id")? as usize,
            sequence:integer(meta,"sequence")?,timestamp_ns:integer(meta,"timestamp_ns")?,
            sensor_x:integer(meta,"sensor_x")? as u32,sensor_y:integer(meta,"sensor_y")? as u32,
            width,height,registration_anchor:None,pupil_component_seed:None,
            pixels:Arc::new(raw10::try_unpack_raw10(&packed,width,height,integer(meta,"stride")? as usize)?),
        });
        Ok((row,frame))
    }))
}

#[test]
#[ignore = "matched native corpus motion parity and timing; explicit index/report required"]
fn native_global_patch_cache_corpus_parity() {
    let input=PathBuf::from(env::var("BUTTERCUP_PATCH_PARITY_INDEX").expect("native source index"));
    let output=PathBuf::from(env::var("BUTTERCUP_PATCH_PARITY_REPORT").expect("fresh output"));
    assert!(fs::canonicalize(output.parent().unwrap()).unwrap().starts_with(fs::canonicalize("outputs").unwrap()));
    let mut writer=BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap());
    let mut reference:[raw_motion_octrees::NativeGlobalSimilarityTracker;2]=std::array::from_fn(|_|Default::default());
    let mut candidate:[raw_motion_octrees::NativeGlobalSimilarityTracker;2]=std::array::from_fn(|_|Default::default());
    for tracker in &mut reference {tracker.use_reference_patch_cost();tracker.retain_diagnostic_correspondences(true);}
    for tracker in &mut candidate {tracker.retain_diagnostic_correspondences(true);}
    let mut lineage=Value::Null;
    let mut count=0;
    for (index,item) in source_frames(&input,0,usize::MAX).unwrap().enumerate() {
        let (row,raw)=item.unwrap();let eye=raw.eye_index;
        if row["clock_lineage"]!=lineage {
            lineage=row["clock_lineage"].clone();
            for tracker in reference.iter_mut().chain(candidate.iter_mut()) {tracker.clear();}
        }
        let observe=|tracker:&mut raw_motion_octrees::NativeGlobalSimilarityTracker| {
            let start=Instant::now();
            let evidence=tracker.observe(Arc::clone(&raw.pixels),raw.width,raw.height,raw.sensor_x,raw.sensor_y);
            (evidence,start.elapsed().as_secs_f64()*1000.0)
        };
        let (baseline,optimized)=if index%2==0 {
            (observe(&mut reference[eye]),observe(&mut candidate[eye]))
        } else {
            let optimized=observe(&mut candidate[eye]);let baseline=observe(&mut reference[eye]);
            (baseline,optimized)
        };
        assert_eq!(format!("{:?}",baseline.0),format!("{:?}",optimized.0),"whole-ROI evidence differs at {index}");
        assert_eq!(format!("{:?}",reference[eye].diagnostic_correspondences()),
            format!("{:?}",candidate[eye].diagnostic_correspondences()),"sparse correspondence differs at {index}");
        serde_json::to_writer(&mut writer,&json!({"schema":"buttercup-native-patch-cache-parity-v1",
            "index":row["index"],"capture_entry":row["capture_entry"],"clock_lineage":row["clock_lineage"],
            "eye_id":eye+1,"source_sequence":raw.sequence,"source_timestamp_ns":raw.timestamp_ns.to_string(),
            "raw_sha256":row["raw_sha256"],"reference_ms":baseline.1,"candidate_ms":optimized.1,
            "exact_motion_and_correspondences_equal":true,"reliable":optimized.0.reliable,
            "candidate_matches":optimized.0.candidate_matches,"candidate_motion":format!("{:?}",optimized.0.candidate_motion),
            "scope":"native motion only; no new segmentation/gaze accuracy claim; alternating timing order on shared host"})).unwrap();
        writeln!(writer).unwrap();count+=1;
        if count%10_000==0 {writer.flush().unwrap();eprintln!("NATIVE_PATCH_PARITY frames={count}");}
    }
    writer.flush().unwrap();
    eprintln!("NATIVE_PATCH_PARITY complete frames={count}");
}

/// Independent native-RAW motion diagnostic, with no SAM/target/label inputs.
/// It uses the live whole-ROI matcher and preserves missing/rejected transport.
/// Whole-ROI support is not a claim of pure head motion or measured 3D motion.
pub(crate) fn export_motion<I>(mut args:I)->Result<(),String> where I:Iterator<Item=String> {
    let input=PathBuf::from(args.next().ok_or("expected FRAMES.jsonl OUTPUT.jsonl [SAM.jsonl]")?);
    let output=PathBuf::from(args.next().ok_or("missing motion output")?);
    let sam=args.next().map(|path|File::open(path).map(BufReader::new).map_err(|e|e.to_string())).transpose()?;
    let mut sam=sam.map(|reader|reader.lines());
    if args.next().is_some() {return Err("unexpected motion export argument".into());}
    let allowed=fs::canonicalize("outputs").map_err(|e|e.to_string())?;
    if !fs::canonicalize(output.parent().ok_or("output needs a parent")?).map_err(|e|e.to_string())?.starts_with(allowed) {
        return Err("motion evidence output must be beneath outputs".into());
    }
    let mut writer=BufWriter::new(std::fs::OpenOptions::new().create_new(true).write(true).open(output).map_err(|e|e.to_string())?);
    let mut trackers:[raw_motion_octrees::NativeGlobalSimilarityTracker;2]=std::array::from_fn(|_|Default::default());
    for tracker in &mut trackers {tracker.retain_diagnostic_correspondences(true);}
    let mut previous=[None;2];let mut lineage=Value::Null;
    let motion_json=|m:roi_evidence::SimilarityMotion|json!({"translation_px":m.translation,
        "diagonal_coefficient_delta":m.diagonal_coefficient_delta,"rotation_coefficient":m.rotation_coefficient,
        "residual_px":m.residual,"support":m.support});
    for (index,source) in source_frames(&input,0,usize::MAX)?.enumerate() {
        let (row,raw)=source?;let eye=raw.eye_index;
        if row["clock_lineage"]!=lineage {
            lineage=row["clock_lineage"].clone();previous=[None;2];
            for tracker in &mut trackers {tracker.clear();}
        }
        if previous[eye].is_some_and(|before|raw.timestamp_ns<=before) {
            return Err("motion export requires strictly increasing sources per ROI/clock".into());
        }
        let started=Instant::now();
        let exclusion=if let Some(sam)=sam.as_mut() {
            let case:Value=serde_json::from_str(&sam.next().ok_or("SAM exclusion cache shorter than RAW sources")?
                .map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            if case["input"]!=row {return Err("SAM exclusion is not the exact current RAW receipt".into());}
            let selected=case["selected_query"].as_u64().and_then(|query|case["candidates"].as_array()?
                .iter().find(|c|c["query"].as_u64()==Some(query)));
            selected.and_then(|c| {
                let e=&c["baseline_ellipse"];
                Some(crate::geometry::Ellipse {center:(e["center"][0].as_f64()?+raw.sensor_x as f64,
                    e["center"][1].as_f64()?+raw.sensor_y as f64),major_radius:e["major_radius"].as_f64()?*1.15,
                    minor_radius:e["minor_radius"].as_f64()?*1.15,angle:e["angle"].as_f64()?})
            })
        } else {None};
        let evidence=if sam.is_some() && exclusion.is_none() {
            trackers[eye].clear();roi_evidence::NativeGlobalSimilarityEvidence::default()
        } else {
            trackers[eye].observe_excluding(Arc::clone(&raw.pixels),raw.width,raw.height,raw.sensor_x,raw.sensor_y,exclusion)
        };
        let report=json!({"schema":"buttercup-native-roi-motion-replay-v1","input":row,
            "from_source_ns":previous[eye].map(|v:u64|v.to_string()),"to_source_ns":raw.timestamp_ns.to_string(),
            "reliable":evidence.reliable,"motion":motion_json(evidence.motion),
            "candidate_motion":motion_json(evidence.candidate_motion),"motion_center_sensor_px":evidence.motion_center_sensor,
            "candidate_matches":evidence.candidate_matches,"spatial_span_px":evidence.spatial_span,
            "occupied_quadrants":evidence.occupied_quadrants,"stable_frames":evidence.stable_frames,
            "support_policy":if sam.is_some() {"outside-current-and-previous-2D-limbus-plus-15pct-and-patch-margin"} else {"whole-ROI"},
            "native_patch_correspondences":trackers[eye].diagnostic_correspondences().iter().map(|m|
                json!({"previous_sensor_px":m.previous_sensor_px,"current_sensor_px":m.current_sensor_px,
                    "photometric_score":m.photometric_score,"distinct_match_margin":m.distinct_match_margin,
                    "global_similarity_inlier":m.global_similarity_inlier})).collect::<Vec<_>>(),
            "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
            "contract":"current native RAW whole-ROI similarity; no fitted conic, screen target or identity transform for missing evidence"});
        serde_json::to_writer(&mut writer,&report).map_err(|e|e.to_string())?;writer.write_all(b"\n").map_err(|e|e.to_string())?;
        previous[eye]=Some(raw.timestamp_ns);
        if (index+1)%100==0 {writer.flush().map_err(|e|e.to_string())?;eprintln!("native motion replay frames={}",index+1);}
    }
    if sam.as_mut().is_some_and(|rows|rows.next().is_some()) {return Err("SAM exclusion cache longer than RAW sources".into());}
    writer.flush().map_err(|e|e.to_string())
}

pub(crate) fn export<I>(mut args:I)->Result<(),String> where I:Iterator<Item=String> {
    let input=PathBuf::from(args.next().ok_or("expected FRAMES.jsonl OUTPUT.jsonl [START] [COUNT]")?);
    let output=PathBuf::from(args.next().ok_or("missing output")?);
    let start=args.next().map(|s|s.parse::<usize>().map_err(|e|e.to_string())).transpose()?.unwrap_or(0);
    let count=args.next().map(|s|s.parse::<usize>().map_err(|e|e.to_string())).transpose()?.unwrap_or(usize::MAX);
    if args.next().is_some() {return Err("unexpected stereo export argument".into());}
    let allowed=fs::canonicalize("outputs").map_err(|e|e.to_string())?;
    if !fs::canonicalize(output.parent().ok_or("output needs a parent")?).map_err(|e|e.to_string())?.starts_with(allowed) {
        return Err("stereo evidence output must be beneath outputs".into());
    }
    let mut writer=BufWriter::new(std::fs::OpenOptions::new().create_new(true).write(true).open(&output).map_err(|e|e.to_string())?);
    let frames=source_frames(&input,start,count)?;
    let model=env::var_os("BUTTERCUP_SAM31_MODEL").map(PathBuf::from).unwrap_or_else(sam31_outer::default_model_path);
    let started=Instant::now();
    let mut visit=|row:Value,_:&Arc<sam31_outer::RawFrame>,mut case:Value| {
        case["input"]=row;
        // Native outline samples are retained for rejected candidates too.
        // Sparse selection after this immutable cache is shared by both arms.
        serde_json::to_writer(&mut writer,&case).map_err(|e|e.to_string())?;
        writer.write_all(b"\n").map_err(|e|e.to_string())?;
        writer.flush().map_err(|e|e.to_string())
    };
    let total=match env::var("BUTTERCUP_STEREO_LIVE_REPLAY").ok().as_deref() {
        Some("outer") => visit_live_frames(&model,frames,sam31_outer::Target::OuterLimbus,&mut visit)?,
        Some("combined") => visit_live_frames(&model,frames,sam31_outer::Target::OuterLimbusAndInnerPupilVoid,&mut visit)?,
        Some("offered-combined") => visit_offered_frames(&model,frames,sam31_outer::Target::OuterLimbusAndInnerPupilVoid,&mut visit)?,
        Some(other) => return Err(format!("unknown live replay profile {other:?}")),
        None => sam31_outer::visit_native_outline_frames(&model,frames,&mut visit)?,
    };
    writer.flush().map_err(|e|e.to_string())?;
    eprintln!("stereo SAM export completed frames={total} elapsed_seconds={:.3} output={}",started.elapsed().as_secs_f64(),output.display());
    Ok(())
}

/// Completion-paced, source-grouped replay of the actual asynchronous video
/// workers. This isolates geometry: it is NOT an offered-load latency test.
/// Only one source group is held here, and the two eyes run concurrently.
fn visit_live_frames<I,F>(model:&Path, frames:I, target:sam31_outer::Target, mut visit:F)
    ->Result<usize,String>
where I:Iterator<Item=Result<(Value,Arc<sam31_outer::RawFrame>),String>>,
    F:FnMut(Value,&Arc<sam31_outer::RawFrame>,Value)->Result<(),String> {
    let client=sam31_outer::Client::start(model)?;
    let mut frames=frames.peekable();
    let mut lineage=Value::Null;
    let mut epoch=0;
    let mut count=0;
    while let Some(first)=frames.next() {
        let first=first?;
        if lineage!=first.0["clock_lineage"] {lineage=first.0["clock_lineage"].clone();epoch+=1;}
        let time=first.1.timestamp_ns;
        let mut group=vec![first];
        while frames.peek().is_some_and(|next|next.as_ref().is_ok_and(|(row,raw)|
            row["clock_lineage"]==lineage && raw.timestamp_ns==time)) {
            group.push(frames.next().unwrap()?);
        }
        let mut present=[false;2];
        for (_,raw) in &group {
            if raw.eye_index>=2 || present[raw.eye_index] {return Err("invalid/duplicate eye in source group".into());}
            present[raw.eye_index]=true;
        }
        let started=Instant::now();
        let before=client.status().completed_batches;
        let mut proposals:[Option<Arc<sam31_outer::ProposalMasks>>;2]=[None,None];
        for (_,raw) in &group {
            let history=VecDeque::from([Arc::clone(raw)]);
            loop {
                match client.submit_history(&history,target,sam31_outer::OUTER_IRIS_PROMPT,0,epoch) {
                    sam31_outer::SubmitOutcome::Accepted => break,
                    sam31_outer::SubmitOutcome::Invalid => return Err("live SAM replay submission failed".into()),
                    sam31_outer::SubmitOutcome::DroppedBusy => {
                        if started.elapsed()>Duration::from_secs(90) {return Err("live SAM replay submission timed out".into());}
                        std::thread::sleep(Duration::from_millis(1));
                    },
                }
            }
        }
        loop {
            // Read completion counters first: publication happens before the
            // counter advances, so the final drain cannot miss its proposal.
            let status=client.status();
            for proposal in client.drain_proposal_masks() {
                let eye=proposal.eye_index;
                if eye>=2 || !present[eye] || proposal.source_timestamp_ns!=time
                    || proposal.tracking_epoch!=epoch {return Err("live replay returned a different source group".into());}
                let raw=&group.iter().find(|(_,r)|r.eye_index==eye).unwrap().1;
                if proposal.source_sequence!=raw.sequence {return Err("live replay sequence mismatch".into());}
                proposals[eye]=Some(proposal);
            }
            client.drain_results();
            if status.completed_batches>=before+group.len() as u64 {break;}
            if status.state=="error" {return Err(format!("live SAM replay: {}",status.detail));}
            if started.elapsed()>Duration::from_secs(90) {return Err("live SAM replay worker timed out".into());}
            std::thread::sleep(Duration::from_millis(1));
        }
        for (row,raw) in group {
            let proposal=proposals[raw.eye_index].as_deref();
            let mut case=live_case(&raw,proposal);
            case["elapsed_ms"]=json!(started.elapsed().as_secs_f64()*1000.0);
            case["replay"]=json!({"method":"live-video-worker-completion-paced","target":target.label(),
                    "source_group_size":present.into_iter().filter(|v|*v).count(),
                    "proposal_received":proposal.is_some(),"global_motion":"unavailable-no-RAW-thumbnail-motion-replay",
                    "initial_memory":"empty-at-capture-start","timing_validation":false});
            visit(row,&raw,case)?;
            count+=1;
        }
        if count%20<2 {eprintln!("live SAM replay frames={count} source_ns={time}");}
    }
    Ok(count)
}

fn live_case(raw:&sam31_outer::RawFrame,proposal:Option<&sam31_outer::ProposalMasks>)->Value {
    let fit=proposal.and_then(|p|p.outer_fit.as_ref());
    let ellipse=|e:sam31_outer::Ellipse|json!({"center":e.center,"major_radius":e.major_radius,
        "minor_radius":e.minor_radius,"angle":e.angle});
    let candidates=fit.map(|fit|vec![json!({"query":0,"baseline_ellipse":ellipse(fit.ellipse),
        "baseline_retained":fit.retained_points.as_ref(),"baseline_retained_segments":fit.conic_segments.as_ref(),
        "baseline_censored":fit.flat_tire_points.as_ref(),
        "baseline_raw_admitted":proposal.is_some_and(sam31_outer::proposal_raw_outer_admitted)})]).unwrap_or_default();
    json!({"sequence":raw.sequence,"timestamp_ns":raw.timestamp_ns,
        "source_group_roi_count":proposal.map(|p|p.source_group_roi_count),
        "sensor_origin":[raw.sensor_x,raw.sensor_y],"width":raw.width,"height":raw.height,
        "candidates":candidates,"selected_query":fit.map(|_|0),
        "pupil_void":proposal.and_then(|p|p.inner_pupil_fit).map(|p|json!({"ellipse":ellipse(p.ellipse)}))})
}

/// Offer one clip at its native sensor cadence, with the production atomic
/// stereo ingress. Replaced/absent jobs are not written as negative detections.
/// Cache only bounded source references; save actual completion times and the
/// latest received crop metadata so downstream acquisition sees real latency.
fn visit_offered_frames<I,F>(model:&Path,frames:I,target:sam31_outer::Target,mut visit:F)->Result<usize,String>
where I:Iterator<Item=Result<(Value,Arc<sam31_outer::RawFrame>),String>>,
    F:FnMut(Value,&Arc<sam31_outer::RawFrame>,Value)->Result<(),String> {
    let client=sam31_outer::Client::start(model)?;
    if !client.supports_source_groups() {return Err("offered stereo replay requires two pipelined worker lanes".into());}
    // Model loading/JIT warm-up precedes a live user's calibration. Use a
    // blank frame in a disposable epoch, not future eye evidence, to keep the
    // cold model start out of the offered camera clock.
    let warmup=Instant::now();
    let blank=std::array::from_fn(|eye|Arc::new(sam31_outer::RawFrame {
        eye_index:eye,sequence:1,timestamp_ns:1,sensor_x:0,sensor_y:0,width:420,height:280,
        pixels:Arc::new(vec![0;420*280]),registration_anchor:None,pupil_component_seed:None,
    }));
    if client.submit_source_group(blank,target,sam31_outer::OUTER_IRIS_PROMPT,0,[0;2],[None,None])
        !=sam31_outer::SubmitOutcome::Accepted {return Err("offered replay warmup failed".into());}
    loop {
        let status=client.status();client.drain_proposal_masks();client.drain_results();
        if status.completed_batches>=2 {break;}
        if status.state=="error" || warmup.elapsed()>Duration::from_secs(90) {return Err("offered replay warmup did not complete".into());}
        std::thread::sleep(Duration::from_millis(1));
    }
    let initial_completed=client.status().completed_batches;
    let mut frames=frames.peekable();
    let first_source=frames.peek().ok_or("empty replay")?.as_ref().map_err(Clone::clone)?.1.timestamp_ns;
    let lineage=frames.peek().unwrap().as_ref().unwrap().0["clock_lineage"].clone();
    let start=Instant::now();
    let mut pending=BTreeMap::<(usize,u64,u64),(Value,Arc<sam31_outer::RawFrame>)>::new();
    let mut latest=[Value::Null,Value::Null];
    let mut count=0;let mut offered=0;let mut incomplete=0;
    let mut drain=|pending:&mut BTreeMap<(usize,u64,u64),(Value,Arc<sam31_outer::RawFrame>)>,latest:&[Value;2]|->Result<(),String> {
        for proposal in client.drain_proposal_masks() {
            let key=(proposal.eye_index,proposal.source_sequence,proposal.source_timestamp_ns);
            let (row,raw)=pending.remove(&key).ok_or("completion outside bounded offered source ledger")?;
            if proposal.tracking_epoch!=1 || proposal.prompt_generation!=0 {return Err("offered replay lineage mismatch".into());}
            let mut case=live_case(&raw,Some(&proposal));
            case["replay"]=json!({"method":"live-video-worker-offered-load","target":target.label(),
                "ready_elapsed_ns":start.elapsed().as_nanos().to_string(),"first_source_ns":first_source.to_string(),
                "presentation_inputs":latest,"global_motion":"unavailable-no-RAW-thumbnail-motion-replay",
                "initial_memory":"blank-warmup-discarded-by-new-tracking-epoch","timing_validation":true});
            visit(row,&raw,case)?;count+=1;
        }
        client.drain_results();
        if client.status().state=="error" {return Err(format!("offered SAM replay: {}",client.status().detail));}
        Ok(())
    };
    while let Some(first)=frames.next() {
        let first=first?;let time=first.1.timestamp_ns;
        if first.0["clock_lineage"]!=lineage || time<first_source {return Err("offered replay requires one increasing sensor lineage".into());}
        let mut group=vec![first];
        while frames.peek().is_some_and(|next|next.as_ref().is_ok_and(|(row,raw)|
            row["clock_lineage"]==lineage && raw.timestamp_ns==time)) {group.push(frames.next().unwrap()?);}
        while start.elapsed()<Duration::from_nanos(time-first_source) {
            drain(&mut pending,&latest)?;std::thread::sleep(Duration::from_millis(1));
        }
        drain(&mut pending,&latest)?;
        offered+=group.len();
        for (row,raw) in &group {
            if raw.eye_index>=2 || row["clock_attested"]!=true {return Err("unattested offered source".into());}
            latest[raw.eye_index]=row.clone();
        }
        let stereo=group[0].0["frame"]["region"]["active_mask"]==3;
        let accepted=if stereo {
            if group.len()!=2 || group[0].1.eye_index!=0 || group[1].1.eye_index!=1 {
                incomplete+=group.len();false
            } else {client.submit_source_group([Arc::clone(&group[0].1),Arc::clone(&group[1].1)],
                target,sam31_outer::OUTER_IRIS_PROMPT,0,[1;2],[None,None])==sam31_outer::SubmitOutcome::Accepted}
        } else if group.len()==1 {
            client.submit_history(&VecDeque::from([Arc::clone(&group[0].1)]),target,
                sam31_outer::OUTER_IRIS_PROMPT,0,1)==sam31_outer::SubmitOutcome::Accepted
        } else {return Err("unexpected offered source group".into());};
        if accepted {for (row,raw) in group {pending.insert((raw.eye_index,raw.sequence,raw.timestamp_ns),(row,raw));}}
        pending.retain(|(_,_,source),_|time.saturating_sub(*source)<5_000_000_000);
        if pending.len()>128 {return Err("offered source ledger exceeded bounded capacity".into());}
    }
    let finished=Instant::now();
    loop {
        drain(&mut pending,&latest)?;
        let status=client.status();
        if status.completed_batches-initial_completed+status.dropped_batches>=(offered-incomplete) as u64 {
            drain(&mut pending,&latest)?;break;
        }
        if finished.elapsed()>Duration::from_secs(10) {return Err("offered replay did not drain".into());}
        std::thread::sleep(Duration::from_millis(1));
    }
    drop(drain);
    let status=client.status();
    eprintln!("OFFERED_SAM offered={offered} incomplete={incomplete} completed={} proposals={count} dropped={} replaced={}",
        status.completed_batches-initial_completed,status.dropped_batches,status.replaced_batches);
    Ok(count)
}
