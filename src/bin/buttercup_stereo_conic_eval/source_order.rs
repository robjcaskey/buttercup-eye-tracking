//! Replay immutable native evidence through the SAME bounded live tracker.
//!
//! Only arrival order is perturbed. Source timestamps, clocks, crop geometry,
//! RAW bytes and training/withheld samples are never changed. A previous joint
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

pub(super) fn run(files:&[String],limit:usize,partial_outlines:bool,delay:[u64;2],
    writer:&mut impl Write)->Result<(),String> {
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
    let mut clock=None;
    let mut generation=0;
    let mut newest=0u64;
    let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
    let mut line=String::new();
    for (event,position) in positions.into_iter().enumerate() {
        let reader=&mut readers[position.file];
        reader.seek(SeekFrom::Start(position.offset)).map_err(|e|e.to_string())?;
        line.clear();reader.read_line(&mut line).map_err(|e|e.to_string())?;
        let frame=Arc::new(prepare(serde_json::from_str(&line).map_err(|e|e.to_string())?,partial_outlines)?);
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
            "generation":generation,"arrival_delay_ns":delay.map(|v|v.to_string()),
            "logical_arrival_timestamp_ns":(source.timestamp_ns+delay[eye]).to_string(),
            "source_now_ns":newest.to_string(),"native_roi_reframe":reframe,
            "contract":"Fresh native evidence through JointTracker. Delays are synthetic scheduling stress, not measured latency. Repeated publications are not additional RAW exposures."});
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
