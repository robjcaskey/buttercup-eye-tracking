//! Complete-corpus, bounded-memory SAM observation export. No recorded fitted
//! ellipse, gaze, target position or human label enters detector inference.
use super::*;
use std::io::{BufRead,BufReader,BufWriter,Write};

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
    let mut last_file:Option<(String,File)>=None;
    let records=BufReader::new(File::open(&input).map_err(|e|e.to_string())?).lines().skip(start).take(count);
    let frames=records.map(|line| {
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
            eye_index:integer(meta,"eye_id")?.checked_sub(1).ok_or("invalid eye id")? as usize,
            sequence:integer(meta,"sequence")?,timestamp_ns:integer(meta,"timestamp_ns")?,
            sensor_x:integer(meta,"sensor_x")? as u32,sensor_y:integer(meta,"sensor_y")? as u32,
            width,height,registration_anchor:None,pupil_component_seed:None,
            pixels:Arc::new(raw10::try_unpack_raw10(&packed,width,height,integer(meta,"stride")? as usize)?),
        });
        Ok((row,frame))
    });
    let model=env::var_os("BUTTERCUP_SAM31_MODEL").map(PathBuf::from).unwrap_or_else(sam31_outer::default_model_path);
    let started=Instant::now();
    let total=sam31_outer::visit_native_outline_frames(&model,frames,|row,_,mut case| {
        case["input"]=row;
        // Native outline samples are retained for rejected candidates too.
        // Sparse selection after this immutable cache is shared by both arms.
        serde_json::to_writer(&mut writer,&case).map_err(|e|e.to_string())?;
        writer.write_all(b"\n").map_err(|e|e.to_string())?;
        writer.flush().map_err(|e|e.to_string())
    })?;
    writer.flush().map_err(|e|e.to_string())?;
    eprintln!("stereo SAM export completed frames={total} elapsed_seconds={:.3} output={}",started.elapsed().as_secs_f64(),output.display());
    Ok(())
}
