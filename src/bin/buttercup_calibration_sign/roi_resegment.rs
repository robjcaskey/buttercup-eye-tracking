//! Matched fresh SAM/RAW16 Obelisk masks on the frozen ambiguous exposure set.
//! Historical diagnostic only: no training, target labels or sign selection.
use super::{bootstrapability as boot, data, native, sam_export, sam_native, Result};
#[path = "../buttercup_roi_focus/archive.rs"]
#[allow(dead_code)]
mod archive;
#[path = "canvas.rs"]
#[allow(dead_code)]
mod canvas;
#[path = "../../raw_preview.rs"]
#[allow(dead_code)]
mod preview;
use buttercup_eye_tracking::{recorded_bundle::BundleSource, sam31_outer as sam};
use canvas::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

pub fn lines(p: impl AsRef<Path>) -> Result<Vec<Value>> {
    BufReader::new(fs::File::open(p)?)
        .lines()
        .map(|r| Ok(serde_json::from_str(&r?)?))
        .collect()
}
fn student_segment(
    model: &sam::student::Model,
    input: &sam_native::Input,
) -> Result<(Option<native::NativeFit>, Vec<u8>)> {
    let f = &input.frame;
    let n = |k: &str| data::num(&f[k]).unwrap();
    let raw = Arc::new(sam::RawFrame {
        eye_index: n("eye_id") as usize - 1,
        sequence: n("sequence"),
        timestamp_ns: n("timestamp_ns"),
        sensor_x: n("sensor_x") as u32,
        sensor_y: n("sensor_y") as u32,
        width: n("width") as usize,
        height: n("height") as usize,
        registration_anchor: None,
        pupil_component_seed: None,
        pixels: Arc::new(input.raw.clone()),
    });
    let (logits, _) = model.infer_source(&raw, false)?;
    if logits.size() != [1, 6, sam::FRAME_HEIGHT as i64, sam::FRAME_WIDTH as i64]
        || logits.isfinite().all().int64_value(&[]) != 1
    {
        return Err("Obelisk nonfinite or wrong mask shape".into());
    }
    let mut probabilities = vec![0f32; 6 * sam::FRAME_WIDTH * sam::FRAME_HEIGHT];
    let size = probabilities.len();
    logits
        .sigmoid()
        .contiguous()
        .copy_data(&mut probabilities, size);
    let mask = probabilities[..sam::FRAME_WIDTH * sam::FRAME_HEIGHT]
        .iter()
        .map(|v| u8::from(*v >= 0.5))
        .collect::<Vec<_>>();
    let fit = sam::diagnostic_fit_single_frame_mask(&mask, sam::FRAME_WIDTH, sam::FRAME_HEIGHT)
        .and_then(|r| {
            let scale = n("width") as f64 / sam::FRAME_WIDTH as f64;
            let mut e = r.ellipse;
            e.center = (
                (e.center.0 + 0.5) * scale - 0.5,
                (e.center.1 + 0.5) * scale - 0.5,
            );
            e.major_radius *= scale;
            e.minor_radius *= scale;
            let points = r
                .retained_points
                .iter()
                .map(|&(x, y)| [(x + 0.5) * scale - 0.5, (y + 0.5) * scale - 0.5])
                .collect();
            native::from_segmentation(&input.raw, f, e, points).map(|mut v| {
                v.provider = "butter-obelisk-raw16".into();
                v
            })
        });
    Ok((
        fit,
        probabilities
            .into_iter()
            .map(|v| (v * 255.).round().clamp(0., 255.) as u8)
            .collect(),
    ))
}
fn render(
    out: &Path,
    id: usize,
    f: &Value,
    raw: &[u16],
    old: &archive::Frame,
    sam_fit: Option<&native::NativeFit>,
    student_fit: Option<&native::NativeFit>,
    student_probs: &[u8],
) -> Result<()> {
    let w = data::num(&f["width"]).unwrap() as usize;
    let h = data::num(&f["height"]).unwrap() as usize;
    let origin = [
        data::num(&f["sensor_x"]).unwrap() as u32,
        data::num(&f["sensor_y"]).unwrap() as u32,
    ];
    let rgb = preview::color_preview(raw, w, h, origin[0], origin[1], 100, None);
    let bgra = rgb
        .iter()
        .flat_map(|p| {
            [
                (*p & 255) as u8,
                ((*p >> 8) & 255) as u8,
                ((*p >> 16) & 255) as u8,
                255,
            ]
        })
        .collect::<Vec<_>>();
    let mut c = Canvas::new(1800, 670)?;
    c.clear();
    c.text(
        20.,
        32.,
        24.,
        WHITE,
        &format!(
            "Same RAW | record {id} | source {} | eye {} | sequence {}",
            old.source, old.eye, old.sequence
        ),
    );
    c.text(20.,61.,16.,MUTED,"Fresh single-image segmentation; contours are model predictions, not human truth. Fixed shared native fit and RAW-support gates.");
    for k in 0..3 {
        let x = 20. + k as f64 * 594.;
        let y = 105.;
        let scale = 570. / w as f64;
        c.text(
            x,
            92.,
            22.,
            WHITE,
            [
                "Archived ellipse",
                "Fresh SAM3.1",
                "Fresh Butter Obelisk RAW16",
            ][k],
        );
        c.image(&bgra, w, h, x, y, 570., 570. * h as f64 / w as f64);
        c.clipped(x, y, 570., 570. * h as f64 / w as f64, |c| {
            let fit = if k == 1 { sam_fit } else { student_fit };
            let e = if k == 0 {
                old.shape().map(|mut e| {
                    e.center.0 -= origin[0] as f64;
                    e.center.1 -= origin[1] as f64;
                    e
                })
            } else {
                fit.map(|f| f.ellipse())
            };
            if let Some(e) = e {
                c.path(
                    &e.dense_points(200)
                        .iter()
                        .map(|&(u, v)| [x + u * scale, y + v * scale])
                        .collect::<Vec<_>>(),
                    2.,
                    WHITE,
                );
            }
            if k > 0 {
                if let Some(f) = fit {
                    for p in f.points.iter().step_by(3) {
                        c.dot(x + p[0] * scale, y + p[1] * scale, 2., PINK, true);
                    }
                }
            }
            if k == 2 {
                for yy in (0..sam::FRAME_HEIGHT).step_by(4) {
                    for xx in (0..sam::FRAME_WIDTH).step_by(4) {
                        if student_probs
                            [3 * sam::FRAME_WIDTH * sam::FRAME_HEIGHT + yy * sam::FRAME_WIDTH + xx]
                            >= 179
                        {
                            c.dot(
                                x + (xx as f64 + 0.5) * 570. / sam::FRAME_WIDTH as f64,
                                y + (yy as f64 + 0.5) * 570. / sam::FRAME_WIDTH as f64,
                                1.4,
                                CYAN,
                                true,
                            );
                        }
                    }
                }
            }
        });
        if k > 0 {
            let fit = if k == 1 { sam_fit } else { student_fit };
            let txt = fit
                .map(|f| {
                    format!(
                        "RAW gate {} | rim support {:.2}/{:.2}/{:.2}",
                        f.admissible(),
                        f.outward_support_left_bottom_right[0],
                        f.outward_support_left_bottom_right[1],
                        f.outward_support_left_bottom_right[2]
                    )
                })
                .unwrap_or("No shared fitted ellipse".into());
            c.text(
                x,
                525.,
                16.,
                if fit.is_some_and(|f| f.admissible()) {
                    GREEN
                } else {
                    ORANGE
                },
                &txt,
            );
        }
    }
    c.text(20.,590.,18.,MUTED,"White: fitted ellipse. Pink: retained predicted contour samples. Cyan: Obelisk sclera probability >=0.7 (unverified mask).");
    c.text(20.,627.,17.,MUTED,"Color is display-only demosaicing; stored RAW and physical source coordinates are unchanged. Missing fits remain missing.");
    c.png(&out.join(format!("raw-comparison-{id}.png")))
}
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 6 && args.len() != 7 {
        return Err("roi-resegment BINARY NEURAL_EVAL SAM_EXPORT OBELISK NEW_OUT [LIMIT]".into());
    }
    let start = Instant::now();
    let out = data::output(&args[5])?;
    fs::create_dir(out.join("masks"))?;
    let source = boot::current_source(Path::new("."))?;
    let binary_hash = sam_export::hash(Path::new(&args[1]))?;
    let eval: Value = serde_json::from_slice(&fs::read(Path::new(&args[2]).join("summary.json"))?)?;
    if eval["binary_sha256"] != binary_hash {
        return Err("ambiguous cohort binary mismatch".into());
    }
    let (manifest, frames) = archive::read(Path::new(&args[1]))?;
    let mut chosen = lines(Path::new(&args[2]).join("predictions.jsonl"))?;
    if let Some(limit) = args.get(6) {
        chosen.truncate(limit.parse()?);
    }
    let mut groups = BTreeMap::<u32, Vec<Value>>::new();
    for r in chosen {
        groups
            .entry(r["source"].as_u64().unwrap() as u32)
            .or_default()
            .push(r);
    }
    let weights = Path::new(&args[4]);
    let metadata = sam::student::validate_model(weights)?;
    if metadata["architecture"] != sam::student::RAW_ARCHITECTURE {
        return Err("require Butter Obelisk RAW16, not old RGB student".into());
    }
    let model_hash = sam_export::hash(weights)?;
    if metadata["weights_sha256"] != model_hash {
        return Err("Obelisk weights/manifest mismatch".into());
    }
    let sam_receipt: Value =
        serde_json::from_slice(&fs::read(Path::new(&args[3]).join("export.json"))?)?;
    data::write(
        out.join("provenance.json"),
        &json!({"source":source,"binary_sha256":binary_hash,"cohort_summary_sha256":sam_export::hash(&Path::new(&args[2]).join("summary.json"))?,"executable_sha256":sam_export::hash(&std::env::current_exe()?)?,"sam_export":sam_receipt,"obelisk_model":weights,"obelisk_sha256":model_hash,"obelisk_metadata":metadata,"scope":"Explicit existing-model diagnostic comparison permitted by bootstrapability.md. No custom-model training, pseudo-label training, model promotion or current-checkout bootstrap proof. SAM uses its pinned verified export; Obelisk runs on CPU.","mask_format":{"sam":[420,280],"obelisk":[6,sam::FRAME_HEIGHT,sam::FRAME_WIDTH],"obelisk_encoding":"uint8 round(sigmoid(logit)*255); channel order matches six semantic prompts"}}),
    )?;
    let teacher = sam_native::Teacher::open(Path::new(&args[3]))?;
    let student = sam::student::Model::load_on_device(weights, tch::Device::Cpu)?;
    tch::set_num_threads(4);
    let mut writer = BufWriter::new(fs::File::create(out.join("frames.jsonl"))?);
    let mut counts = [0usize; 5];
    let mut sources = vec![];
    for (sid, rows) in groups {
        let s = &manifest.sources[sid as usize];
        let native_path = rows[0]["raw_source"].as_str().ok_or("RAW source path")?;
        let bundle = BundleSource::open(Path::new(native_path))?;
        let index = bundle.read_entry(&format!("{}frames.jsonl", s.prefix))?;
        if data::digest(&index) != s.frames_sha256 {
            return Err("native index hash changed".into());
        }
        let wanted = rows
            .iter()
            .map(|r| {
                let id = r["record"].as_u64().unwrap() as usize;
                (frames[id].index as usize, id)
            })
            .collect::<BTreeMap<_, _>>();
        let mut native = BTreeMap::new();
        for (i, b) in index
            .split(|b| *b == b'\n')
            .filter(|b| !b.is_empty())
            .enumerate()
        {
            if let Some(&id) = wanted.get(&i) {
                native.insert(id, serde_json::from_slice::<Value>(b)?);
            }
        }
        let before = counts;
        for (pos, r) in rows.iter().enumerate() {
            let id = r["record"].as_u64().unwrap() as usize;
            let f = &frames[id];
            let row = native.get(&id).ok_or("native row missing")?;
            for (key, want) in [
                ("eye_id", f.eye as u64),
                ("sequence", f.sequence),
                ("timestamp_ns", f.ns),
                ("offset", f.offset),
                ("length", f.length as u64),
                ("width", f.width as u64),
                ("height", f.height as u64),
                ("stride", f.stride as u64),
                ("sensor_x", f.origin[0] as u64),
                ("sensor_y", f.origin[1] as u64),
            ] {
                if data::num(&row[key]) != Some(want) {
                    return Err(format!("RAW identity mismatch {id} {key}").into());
                }
            }
            let raw = bundle.read_range(
                &format!("{}{}", s.prefix, s.streams[f.stream as usize]),
                f.offset,
                f.length as usize,
            )?;
            let hash = data::digest(&raw);
            if r["raw_sha256"] != hash {
                return Err("RAW differs from frozen evaluated exposure".into());
            }
            let input = sam_native::Input {
                raw: native::unpack(&raw, row)?,
                frame: row.clone(),
                hash: hash.clone(),
            };
            let st = Instant::now();
            let (student_fit, probs) = student_segment(&student, &input)?;
            let student_ms = st.elapsed().as_secs_f64() * 1000.;
            let st = Instant::now();
            let seg = teacher
                .segment(std::slice::from_ref(&input), None)?
                .remove(0);
            let sam_ms = st.elapsed().as_secs_f64() * 1000.;
            let sam_mask = format!("masks/{id}-sam.bin");
            let student_mask = format!("masks/{id}-obelisk.bin");
            fs::write(out.join(&sam_mask), &seg.mask)?;
            fs::write(out.join(&student_mask), &probs)?;
            let output = json!({"record":id,"source":sid,"epoch":f.epoch,"eye":f.eye,"sequence":f.sequence,"source_ns":f.ns.to_string(),"raw_source":native_path,"stream_entry":format!("{}{}",s.prefix,s.streams[f.stream as usize]),"frame":row,"raw_sha256":hash,"original_provider":f.provider,"original_ellipse_sensor":f.ellipse,"sam":{"fit":seg.fit,"admissible":seg.fit.as_ref().is_some_and(|f|f.admissible()),"score":seg.score,"mask":sam_mask,"mask_sha256":data::digest(&seg.mask),"milliseconds":sam_ms},"obelisk":{"fit":student_fit,"admissible":student_fit.as_ref().is_some_and(|f|f.admissible()),"masks":student_mask,"masks_sha256":data::digest(&probs),"milliseconds":student_ms},"previous_evaluation":r,"physical_sign_truth":null});
            serde_json::to_writer(&mut writer, &output)?;
            writeln!(writer)?;
            writer.flush()?;
            counts[0] += 1;
            counts[1] += usize::from(seg.fit.is_some());
            counts[2] += usize::from(seg.fit.as_ref().is_some_and(|f| f.admissible()));
            counts[3] += usize::from(student_fit.is_some());
            counts[4] += usize::from(student_fit.as_ref().is_some_and(|f| f.admissible()));
            if pos < 2 || pos == rows.len() / 2 {
                render(
                    &out,
                    id,
                    row,
                    &input.raw,
                    f,
                    seg.fit.as_ref(),
                    student_fit.as_ref(),
                    &probs,
                )?;
            }
            if counts[0] % 25 == 0 || pos + 1 == rows.len() {
                eprintln!(
                    "RESEGMENT {counts:?} seconds={:.1}",
                    start.elapsed().as_secs_f64()
                );
                data::write(
                    out.join("progress.json"),
                    &json!({"counts_total_sam_fit_admitted_obelisk_fit_admitted":counts,"seconds":start.elapsed().as_secs_f64()}),
                )?;
            }
        }
        sources.push(json!({"source":sid,"raw_source":native_path,"counts":std::array::from_fn::<_,5,_>(|i|counts[i]-before[i])}));
    }
    data::write(
        out.join("summary.json"),
        &json!({"complete":true,"counts_total_sam_fit_admitted_obelisk_fit_admitted":counts,"by_source":sources,"seconds":start.elapsed().as_secs_f64(),"source_start":source,"source_finish":boot::current_source(Path::new("."))?,"physical_sign_accuracy":null,"source_note":"The executable hash and source at launch identify this fixed inference run; later source edits cannot change the running executable."}),
    )?;
    Ok(())
}
