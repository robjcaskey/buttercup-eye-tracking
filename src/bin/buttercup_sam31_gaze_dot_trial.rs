//! Bounded evaluation only: SAM segmentation of two projected gaze candidates.
#![allow(dead_code)]
#[cfg(feature = "sam31")]
#[path = "../sam31_text.rs"]
mod sam31_text;

#[cfg(feature = "sam31")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        io::{Read, Seek, SeekFrom},
        path::Path,
    };
    use tch::{CModule, Device, IValue, Kind, Tensor};
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a=="--select-targets") {
        if args.len()!=4 {return Err("usage: --select-targets ELLIPSE_EXPERIMENT_DIR METADATA_OUTPUT_DIR".into());}
        return select_targets(Path::new(&args[2]),Path::new(&args[3]));
    }
    if !(3..=5).contains(&args.len()) {
        return Err(
            "usage: buttercup_sam31_gaze_dot_trial ELLIPSE_EXPERIMENT_DIR NEW_OUTPUT_DIR [GAZE_PROMPT] [TARGET_SELECTION_JSON]".into(),
        );
    }
    let base = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    fs::create_dir(out)?;
    let read = |name: &str| -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        fs::read_to_string(base.join(name))?
            .lines()
            .map(|s| Ok(serde_json::from_str(s)?))
            .collect()
    };
    let shapes = read("ellipse-only.jsonl")?;
    let estimates = read("estimates.jsonl")?;
    let sides = read("evaluation-sidecar.jsonl")?;
    let gaze_prompt: &'static str = args.get(3).map(|text| &*Box::leak(text.clone().into_boxed_str()))
        .unwrap_or("the yellow dot the eye is looking at");
    let prompts = [
        "yellow dots",
        gaze_prompt,
        "the yellow dot the eye is not looking at",
    ];
    let texts: Vec<_> = prompts
        .iter()
        .map(|&text| sam31_text::SemanticPrompt {
            key: "dot",
            label: "DOT",
            text,
        })
        .collect();
    tch::set_num_threads(4);
    let bundle = sam31_text::encode_prompt_set(
        Path::new("data/models/sam31_multiplex.pt"),
        Path::new("data/models/sam31_bpe_simple_vocab_16e6.txt.gz"),
        &texts,
    )?;
    sam31_text::save_prompt_bundle(&bundle, &out.join("prompts.pt"))?;
    let device = Device::Cuda(0);
    let lf = bundle
        .language_features
        .to_device(device)
        .to_kind(Kind::BFloat16);
    let lm = bundle.language_mask.to_device(device).to_kind(Kind::Bool);
    let mut model = CModule::load_on_device("data/models/sam31_semantic_dynamic_u8.pt", device)?;
    model.set_eval();
    const W: usize = 384;
    const H: usize = 256;
    let mut rows = Vec::new();
    let selection: Vec<Value> = if let Some(path)=args.get(4) {
        serde_json::from_str(&fs::read_to_string(path)?)?
    } else { [914,1073,1195].iter().map(|seq|json!({"capture":"later-recording","sequence":seq})).collect() };
    for selected in selection {
        let seq=selected["sequence"].as_u64().ok_or("missing selected sequence")?;
        for eye in [1, 2] {
            let s = shapes
                .iter()
                .find(|r| {
                    r["capture"] == selected["capture"] && r["sequence"] == seq && r["eye"] == eye
                })
                .ok_or("missing requested frame")?;
            let id = s["id"].as_u64().ok_or("missing id")?;
            let e = estimates
                .iter()
                .find(|r| r["id"] == id)
                .ok_or("missing estimate")?;
            let inp = &sides
                .iter()
                .find(|r| r["id"] == id)
                .ok_or("missing source")?["input"];
            let f = &inp["frame"];
            let w = f["width"].as_u64().unwrap() as usize;
            let h = f["height"].as_u64().unwrap() as usize;
            let mut file = fs::File::open(inp["raw_file"].as_str().unwrap())?;
            file.seek(SeekFrom::Start(inp["raw_offset"].as_u64().unwrap()))?;
            let mut bytes = vec![0; inp["raw_length"].as_u64().unwrap() as usize];
            file.read_exact(&mut bytes)?;
            assert_eq!(bytes.len(), w * h * 5 / 4);
            let mut raw = Vec::with_capacity(w * h);
            for g in bytes.chunks_exact(5) {
                let v = g
                    .iter()
                    .enumerate()
                    .fold(0u64, |a, (i, b)| a | ((*b as u64) << (8 * i)));
                for k in 0..4 {
                    raw.push(((v >> (10 * k)) & 1023) as f64);
                }
            }
            // Average complete 4x4 sensor cells before display to remove CFA phase texture.
            assert_eq!(w % 4, 0);
            assert_eq!(h % 4, 0);
            let dw = w / 4;
            let dh = h / 4;
            let mut gray = vec![0.; dw * dh];
            for gy in 0..dh {
                for gx in 0..dw {
                    for yy in 0..4 {
                        for xx in 0..4 {
                            gray[gy * dw + gx] += raw[(gy * 4 + yy) * w + gx * 4 + xx] / 16.;
                        }
                    }
                }
            }
            let mut sorted = gray.clone();
            sorted.sort_by(f64::total_cmp);
            let lo = sorted[sorted.len() / 200];
            let hi = sorted[sorted.len() * 199 / 200];
            let poses = e["poses"].as_array().ok_or("no candidate poses")?;
            let mut dots = Vec::new();
            for p in poses {
                let p: Vec<f64> = p
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let q = [p[0] + 15. * p[3], p[1] + 15. * p[4], p[2] + 15. * p[5]];
                dots.push([
                    4000. * q[0] / q[2] + 4000. - f["sensor_x"].as_f64().unwrap(),
                    4000. * q[1] / q[2] + 3000. - f["sensor_y"].as_f64().unwrap(),
                ]);
            }
            let minx = dots.iter().map(|p| p[0] - 24.).fold(0., f64::min);
            let miny = dots.iter().map(|p| p[1] - 24.).fold(0., f64::min);
            let maxx = dots.iter().map(|p| p[0] + 24.).fold(w as f64, f64::max);
            let maxy = dots.iter().map(|p| p[1] + 24.).fold(h as f64, f64::max);
            let scale = (W as f64 / (maxx - minx)).min(H as f64 / (maxy - miny));
            let ox = (W as f64 - (maxx - minx) * scale) / 2. - minx * scale;
            let oy = (H as f64 - (maxy - miny) * scale) / 2. - miny * scale;
            let points: Vec<_> = dots
                .iter()
                .map(|p| [p[0] * scale + ox, p[1] * scale + oy])
                .collect();
            let variants = if args.len()==5 {vec!["original", "mirror", "vertical-mirror", "eye-erased"]} else {vec!["original", "mirror", "eye-erased"]};
            for variant in variants {
                let pp: Vec<_> = points
                    .iter()
                    .map(|p| {
                        [
                            if variant == "mirror" {
                                (W - 1) as f64 - p[0]
                            } else {
                                p[0]
                            },
                            if variant=="vertical-mirror" {(H-1)as f64-p[1]} else {p[1]},
                        ]
                    })
                    .collect();
                let mut rgb = vec![24u8; W * H * 3];
                for y in 0..H {
                    for x in 0..W {
                        let ux = if variant == "mirror" { W - 1 - x } else { x };
                        let rx = (ux as f64 - ox) / scale;
                        let uy=if variant=="vertical-mirror" {H-1-y}else{y};
                        let ry = (uy as f64 - oy) / scale;
                        if variant != "eye-erased"
                            && rx >= 0.
                            && ry >= 0.
                            && rx < w as f64
                            && ry < h as f64
                        {
                            let gx = (rx / 4. - 0.5).clamp(0., (dw - 1) as f64);
                            let gy = (ry / 4. - 0.5).clamp(0., (dh - 1) as f64);
                            let ix = gx as usize;
                            let iy = gy as usize;
                            let tx = gx - ix as f64;
                            let ty = gy - iy as f64;
                            let ix1 = (ix + 1).min(dw - 1);
                            let iy1 = (iy + 1).min(dh - 1);
                            let sample = (1. - ty)
                                * ((1. - tx) * gray[iy * dw + ix] + tx * gray[iy * dw + ix1])
                                + ty * ((1. - tx) * gray[iy1 * dw + ix]
                                    + tx * gray[iy1 * dw + ix1]);
                            let v = (((sample - lo) / (hi - lo).max(1.)).clamp(0., 1.).powf(0.7)
                                * 255.) as u8;
                            rgb[(y * W + x) * 3..(y * W + x) * 3 + 3].fill(v);
                        }
                        if pp.iter().any(|p| {
                            (x as f64 - p[0]).powi(2) + (y as f64 - p[1]).powi(2) <= 6f64.powi(2)
                        }) {
                            rgb[(y * W + x) * 3..(y * W + x) * 3 + 3]
                                .copy_from_slice(&[255, 255, 0]);
                        }
                    }
                }
                let name = format!("source-{seq}-eye-{eye}-{variant}");
                let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
                ppm.extend(&rgb);
                fs::write(out.join(format!("{name}.ppm")), ppm)?;
                // Repeat the same still five times to satisfy the installed graph's filmstrip contract.
                let tile = Tensor::from_slice(&rgb)
                    .reshape([H as i64, W as i64, 3])
                    .permute([2, 0, 1])
                    .unsqueeze(0);
                let input = tile.repeat([1, 1, 1, 5]).to_device(device);
                let mut answers = Vec::new();
                for (pi, prompt) in prompts.iter().enumerate() {
                    let result = tch::no_grad(|| {
                        model.forward_is(&[
                            IValue::Tensor(input.shallow_clone()),
                            IValue::Tensor(lf.shallow_clone()),
                            IValue::Tensor(lm.shallow_clone()),
                            IValue::Tensor(Tensor::zeros([1], (Kind::Int64, device))),
                            IValue::Tensor(Tensor::from_slice(&[pi as i64]).to_device(device)),
                        ])
                    })?;
                    let IValue::Tuple(mut v) = result else {
                        return Err("unexpected SAM output".into());
                    };
                    let scores: Tensor = v.remove(0).try_into()?;
                    let masks: Tensor = v.remove(0).try_into()?;
                    let sz = masks.size();
                    let mh = sz[2] as usize;
                    let mw = sz[3] as usize;
                    let nq = sz[1] as usize;
                    let scores = scores
                        .to_device(Device::Cpu)
                        .to_kind(Kind::Float)
                        .contiguous();
                    let mut sv = vec![0f32; nq];
                    scores.copy_data(&mut sv, nq);
                    let masks = masks
                        .gt(0.)
                        .to_device(Device::Cpu)
                        .to_kind(Kind::Uint8)
                        .contiguous();
                    let mut mb = vec![0u8; nq * mw * mh];
                    let n = mb.len();
                    masks.copy_data_u8(&mut mb, n);
                    let mut candidates = Vec::new();
                    let mut support = [0f32; 2];
                    for q in 0..nq {
                        let mut hit = [0usize; 2];
                        let mut total = [0usize; 2];
                        for y in 0..mh {
                            for x in mw * 4 / 5..mw {
                                let px = (x as f64 + 0.5) * ((W * 5) as f64) / mw as f64
                                    - (W * 4) as f64;
                                let py = (y as f64 + 0.5) * H as f64 / mh as f64;
                                for j in 0..2 {
                                    if (px - pp[j][0]).powi(2) + (py - pp[j][1]).powi(2) <= 36. {
                                        total[j] += 1;
                                        hit[j] += mb[(q * mh + y) * mw + x] as usize;
                                    }
                                }
                            }
                        }
                        let fractions = [
                            hit[0] as f32 / total[0].max(1) as f32,
                            hit[1] as f32 / total[1].max(1) as f32,
                        ];
                        for j in 0..2 {
                            if fractions[j] >= 0.5 {
                                support[j] = support[j].max(sv[q]);
                            }
                        }
                        if hit.iter().any(|&n| n > 0) {
                            candidates
                                .push(json!({"query":q,"score":sv[q],"dot_coverage":fractions}));
                        }
                    }
                    answers.push(json!({"prompt":prompt,"dot_scores":support,"dot_candidates":candidates,"mask_size":[mw,mh]}));
                }
                eprintln!(
                    "{name}: {}",
                    json!(answers.iter().map(|a| &a["dot_scores"]).collect::<Vec<_>>())
                );
            rows.push(json!({"name":name,"source_id":id,"selection":selected,"sequence":seq,"eye":eye,"variant":variant,"input":inp,"raw_sha256":format!("{:x}",Sha256::digest(&bytes)),"poses":poses,"dot_pixels":pp,"answers":answers}));
                fs::write(out.join("results.json"), serde_json::to_vec_pretty(&rows)?)?;
            }
        }
    }
    fs::write(
        out.join("scope.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"evaluation only; legacy cached ellipse ancestry not approved for training", "geometry":"project C+15mm*n with legacy K fx=fy=4000,cx=4000,cy=3000; dots depict projected directions, not calibrated screen targets", "truth":"no independently labeled sign truth", "image":"RAW10 unpack; 4x4 phase average then bilinear grayscale percentile contrast gamma0.7; uniform resize/pad; equal 6px yellow dots; five identical tiles", "controls":["yellow dots","negative gaze phrase","horizontal mirror retaining candidate identity","eye erased retaining dot positions"]}),
        )?,
    )?;
    Ok(())
}
#[cfg(not(feature = "sam31"))]
fn main() {
    eprintln!("requires --features sam31");
    std::process::exit(2);
}

#[cfg(feature="sam31")]
fn select_targets(base:&std::path::Path, output:&std::path::Path)->Result<(),Box<dyn std::error::Error>> {
    use std::{fs,io::{BufRead,BufReader}};
    use serde_json::{Value,json};
    let load=|name:&str|->Result<Vec<Value>,Box<dyn std::error::Error>> {
        fs::read_to_string(base.join(name))?.lines().map(|s|Ok(serde_json::from_str(s)?)).collect()
    };
    let shapes=load("ellipse-only.jsonl")?;let side=load("evaluation-sidecar.jsonl")?;let estimates=load("estimates.jsonl")?;
    let num=|v:&Value|v.as_u64().or_else(||v.as_str()?.parse().ok()).unwrap();
    let mut selected=Vec::new();let mut windows=Vec::new();
    for capture in ["complete-nine","later-recording"] {
        let mut spans:Vec<(u64,u64,Value)>=Vec::new();let mut current:Option<(u64,Value)>=None;let mut last=0;
        for line in BufReader::new(fs::File::open(output.join(format!("{capture}-metadata.jsonl")))?).lines() {
            let r:Value=serde_json::from_str(&line?)?;
            if r["event"]!="presentation" {continue;}
            let t=num(&r["host_submit_end_unix_ns"]);last=t;
            let target=r["active_targets"].as_array().and_then(|a|a.iter().find(|t|t["role"]=="calibration" && t["visible"]==true)).cloned();
            if current.as_ref().map(|(_,v)|&v["id"])!=target.as_ref().map(|v|&v["id"]) {
                if let Some((start,v))=current.take(){spans.push((start,t,v));}
                current=target.map(|v|(t,v));
            }
        }
        if let Some((start,v))=current {spans.push((start,last,v));}
        for target_index in 0..5 {
            let target_id=format!("calibration-{target_index}");
            let mut chosen=None;
            for (start,end,target) in spans.iter().filter(|(_,_,v)|v["id"]==target_id) {
                let mut candidates=Vec::new();
                for s in shapes.iter().filter(|r|r["capture"]==capture && r["eye"]==1) {
                    let id=s["id"].as_u64().unwrap();
                    let Some(partner)=shapes.iter().find(|r|r["capture"]==capture && r["sequence"]==s["sequence"] && r["eye"]==2) else {continue};
                    if s["clock"]!=partner["clock"] || s["source_ns"]!=partner["source_ns"] {continue;}
                    let ids=[id,partner["id"].as_u64().unwrap()];
                    let mut arrivals=Vec::new();let mut good=true;
                    for id in ids {
                        let inp=&side.iter().find(|r|r["id"]==id).unwrap()["input"];
                        let f=&inp["frame"];let t=num(&f["host_arrival_unix_ns"]);arrivals.push(t);
                        if t<start+2_100_000_000 || t+200_000_000>*end {good=false;}
                        if !estimates.iter().any(|r|r["id"]==id && r["poses"].as_array().is_some_and(|p|p.len()==2)) {good=false;}
                    }
                    if good {candidates.push((s.clone(),arrivals));}
                }
                candidates.sort_by_key(|(_,a)|a[0]);
                windows.push(json!({"capture":capture,"target":target,"start_host_ns":start,"end_host_ns":end,"eligible_pairs":candidates.len()}));
                if !candidates.is_empty() {
                    let (s,a)=&candidates[candidates.len()/2];
                    chosen=Some(json!({"capture":capture,"sequence":s["sequence"],"clock":s["clock"],"source_ns":s["source_ns"],"target":target["normalized"],"target_index":target_index,"target_id":target_id,"target_start_host_ns":start,"target_end_host_ns":end,"raw_arrival_host_ns":a,"seconds_after_target_submit":a.iter().map(|t|(*t-start)as f64/1e9).collect::<Vec<_>>(),"label_kind":"intended fixation cue, not measured gaze or independent sign truth","selection":"middle eligible pair after 2.1 seconds, at least 0.2 seconds before transition; requires both candidate poses, no SAM score filtering"}));
                    break;
                }
            }
            if let Some(c)=chosen {selected.push(c);}
        }
    }
    fs::write(output.join("selection.json"),serde_json::to_vec_pretty(&selected)?)?;
    fs::write(output.join("target-windows.json"),serde_json::to_vec_pretty(&windows)?)?;
    println!("selected {} source pairs across two sessions",selected.len());
    Ok(())
}
