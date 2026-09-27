//! Fresh official SAM anatomy prompts on area-admitted RAW, without training.
use super::{data, sam_export, sam_native, Result};
#[path = "canvas.rs"]
mod canvas;
#[path = "../../raw_preview.rs"]
#[allow(dead_code)]
mod preview;
use buttercup_eye_tracking::{raw10, recorded_bundle::BundleSource};
use canvas::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};
use tch::{CModule, Device, IValue, Kind, Tensor};
#[path = "contact_sheet.rs"]
pub mod contact_sheet;
#[path = "pupil_prompts.rs"]
pub mod pupil_prompts;
#[path = "sclera_splat_inputs.rs"]
pub mod sclera_splat_inputs;

fn load_cpu_export(export: &Path) -> Result<(Value, Vec<String>, CModule)> {
    let receipt: Value = serde_json::from_slice(&fs::read(export.join("export.json"))?)?;
    let prompts: Vec<String> = serde_json::from_value(receipt["prompts"].clone())?;
    if receipt["schema"] != "buttercup-sam31-cold-export-v1"
        || receipt["checkpoint_sha256"] != sam_export::CHECKPOINT_SHA
        || receipt["upstream_revision"] != sam_export::REVISION
        || receipt["adapter_sha256"] != sam_export::cpu_adapter_hash()
        || receipt["upstream"]["device"] != "cpu"
        || receipt["upstream"]["exported_dtype"] != "float32"
        || prompts.len() != 6
        || prompts.iter().any(|s| s.trim().is_empty())
        || receipt["upstream"]["prompts"] != receipt["prompts"]
        || receipt["model_sha256"] != sam_export::hash(&export.join("detector.pt"))?
    {
        return Err("anatomy SAM export does not match pinned recipe/assets/prompts".into());
    }
    tch::set_num_threads(8);
    tch::set_num_interop_threads(1);
    tch::jit::set_tensor_expr_fuser_enabled(false);
    tch::jit::set_graph_executor_optimize(false);
    let mut model = CModule::load_on_device(export.join("detector.pt"), Device::Cpu)?;
    model.set_eval();
    Ok((receipt, prompts, model))
}

fn infer_scores_masks(model: &CModule, rgb: &[u8]) -> Result<(Tensor, Tensor)> {
    if rgb.len() != 3 * sam_native::TW * sam_native::TH {
        return Err("SAM input shape mismatch".into());
    }
    let image = Tensor::from_slice(rgb)
        .view([1, 3, sam_native::TH as i64, sam_native::TW as i64])
        .to_kind(Kind::Float)
        / 255.;
    let image = image.internal_upsample_bilinear2d_aa([1008, 1008], false, None, None);
    let image = (image - 0.5) / 0.5;
    let IValue::Tuple(mut answer) = model.forward_is(&[IValue::Tensor(image)])? else {
        return Err("SAM answer must be tuple".into());
    };
    if answer.len() != 2 {
        return Err("SAM answer arity".into());
    }
    let IValue::Tensor(logits) = answer.pop().unwrap() else {
        return Err("SAM masks tensor".into());
    };
    let IValue::Tensor(scores) = answer.pop().unwrap() else {
        return Err("SAM scores tensor".into());
    };
    let shape = logits.size();
    if shape.len() != 4 || shape[0] != 6 || scores.size() != shape[..2] {
        return Err("SAM anatomy output shape mismatch".into());
    }
    Ok((scores, logits))
}
fn rows(p: &Path) -> Result<Vec<Value>> {
    BufReader::new(fs::File::open(p)?)
        .lines()
        .map(|s| Ok(serde_json::from_str(&s?)?))
        .collect()
}
fn n(v: &Value) -> u64 {
    data::num(v).expect("native identity")
}

/// Optional, non-learned diagnostic inputs. These do not alter native RAW
/// photometry, live inference, training, or the established default adapter.
fn model_input(input: &sam_native::Input, adapter: &str) -> Result<Vec<u8>> {
    let rgb = sam_native::input_rgb(input)?;
    if adapter == "quad_rgb" {
        return Ok(rgb);
    }
    let (w, h) = (sam_native::TW, sam_native::TH);
    let len = w * h;
    let gray = (0..len)
        .map(|i| {
            ((77 * rgb[i] as u32 + 150 * rgb[len + i] as u32 + 29 * rgb[2 * len + i] as u32 + 128)
                >> 8) as u8
        })
        .collect::<Vec<_>>();
    let gray = match adapter {
        "gray" => gray,
        "gray-bilateral-v1" => {
            // Fixed 11x11 bilateral kernel: spatial sigma 3 native pixels,
            // intensity sigma 20 quantized adapter codes. No learned prior,
            // temporal input, candidate conic, or sign enters this transform.
            let offsets = (-5i32..=5)
                .flat_map(|dy| {
                    (-5i32..=5).map(move |dx| (dx, dy, (-((dx * dx + dy * dy) as f64) / 18.).exp()))
                })
                .collect::<Vec<_>>();
            let range = (0..=255)
                .map(|d| (-(d * d) as f64 / 800.).exp())
                .collect::<Vec<_>>();
            (0..len)
                .map(|i| {
                    let (x, y) = ((i % w) as i32, (i / w) as i32);
                    let mut sum = 0.;
                    let mut mass = 0.;
                    for &(dx, dy, spatial) in &offsets {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                            continue;
                        }
                        let sample = gray[yy as usize * w + xx as usize];
                        let weight = spatial * range[sample.abs_diff(gray[i]) as usize];
                        sum += weight * sample as f64;
                        mass += weight;
                    }
                    (sum / mass).round() as u8
                })
                .collect::<Vec<_>>()
        }
        _ => {
            return Err(
                "unknown anatomy input adapter; use quad_rgb, gray or gray-bilateral-v1".into(),
            )
        }
    };
    Ok(gray.repeat(3))
}
fn read_input(
    row: &Value,
    bundles: &mut BTreeMap<String, BundleSource>,
) -> Result<sam_native::Input> {
    let path = row["raw_source"].as_str().ok_or("RAW source")?;
    if !bundles.contains_key(path) {
        bundles.insert(path.to_string(), BundleSource::open(Path::new(path))?);
    }
    let f = &row["frame"];
    let bytes = bundles[path].read_range(
        row["stream_entry"].as_str().ok_or("stream entry")?,
        n(&f["offset"]),
        n(&f["length"]) as usize,
    )?;
    let hash = data::digest(&bytes);
    if hash != row["raw_sha256"] {
        return Err("RAW byte hash mismatch".into());
    }
    Ok(sam_native::Input {
        raw: raw10::try_unpack_raw10(
            &bytes,
            n(&f["width"]) as usize,
            n(&f["height"]) as usize,
            n(&f["stride"]) as usize,
        )?,
        frame: f.clone(),
        hash,
    })
}
fn render_input(out: &Path, id: u64, rgb: &[u8]) -> Result<()> {
    let (w, h) = (sam_native::TW, sam_native::TH);
    let plane = w * h;
    if rgb.len() != 3 * plane {
        return Err("RGB audit shape mismatch".into());
    }
    let bgra = (0..plane)
        .flat_map(|i| [rgb[2 * plane + i], rgb[plane + i], rgb[i], 255])
        .collect::<Vec<_>>();
    let mut c = Canvas::new(w * 2 + 40, h * 2 + 120)?;
    c.clear();
    c.text(
        20.,
        32.,
        21.,
        WHITE,
        &format!("Exact SAM RGB input | RAW record {id}"),
    );
    c.text(
        20.,
        64.,
        15.,
        MUTED,
        "Before model resize/normalization. Hashed adapter bytes, without display enhancement.",
    );
    c.image(&bgra, w, h, 20., 82., (w * 2) as f64, (h * 2) as f64);
    c.png(&out.join(format!("sam-input-{id}.png")))
}

fn raw_bgra(input: &sam_native::Input) -> (usize, usize, Vec<u8>) {
    let (w, h) = (
        n(&input.frame["width"]) as usize,
        n(&input.frame["height"]) as usize,
    );
    let color = preview::color_preview(
        &input.raw,
        w,
        h,
        n(&input.frame["sensor_x"]) as u32,
        n(&input.frame["sensor_y"]) as u32,
        100,
        None,
    );
    let bgra = color
        .into_iter()
        .flat_map(|v| {
            [
                (v & 255) as u8,
                ((v >> 8) & 255) as u8,
                ((v >> 16) & 255) as u8,
                255,
            ]
        })
        .collect::<Vec<_>>();
    (w, h, bgra)
}
fn render(
    out: &Path,
    id: u64,
    input: &sam_native::Input,
    masks: &[Option<Vec<u8>>],
    details: &[Value],
    rank: usize,
) -> Result<()> {
    let (w, h, bgra) = raw_bgra(input);
    let mut c = Canvas::new(1800, 1170)?;
    c.clear();
    c.text(
        20.,
        35.,
        25.,
        WHITE,
        &format!(
            "Fresh SAM3.1 anatomy proposals | native RAW record {id} | rank {}",
            rank + 1
        ),
    );
    c.text(20.,71.,18.,MUTED,"Saved nonempty proposals, ranked by SAM score. No iris-sign, anatomical-center prior or recorded prediction selects these masks.");
    for (k, detail) in details.iter().enumerate() {
        let prompt = detail["prompt"].as_str().unwrap();
        let x = 20. + (k % 3) as f64 * 594.;
        let y = 125. + (k / 3) as f64 * 470.;
        let s = 570. / w as f64;
        c.text(
            x,
            y - 12.,
            22.,
            WHITE,
            &format!(
                "{prompt} | score {:.3}",
                details[k]["candidates"][rank]["score"]
                    .as_f64()
                    .unwrap_or(0.)
            ),
        );
        c.image(&bgra, w, h, x, y, 570., h as f64 * s);
        if let Some(mask) = &masks[k] {
            let (mw, mh) = (sam_native::TW, sam_native::TH);
            for yy in 1..mh - 1 {
                for xx in 1..mw - 1 {
                    let i = yy * mw + xx;
                    if mask[i] >= 128
                        && [i - 1, i + 1, i - mw, i + mw]
                            .into_iter()
                            .any(|j| mask[j] < 128)
                    {
                        c.dot(
                            x + (xx as f64 + 0.5) * 570. / mw as f64,
                            y + (yy as f64 + 0.5) * h as f64 * s / mh as f64,
                            1.2,
                            CYAN,
                            true,
                        );
                    }
                }
            }
        }
    }
    c.text(20.,1080.,18.,MUTED,"SAM query scores are not calibrated anatomical correctness. A semantic answer may still be the iris, skin or whole eye instead of the requested lid.");
    c.text(20.,1116.,18.,MUTED,"All alternative saved proposals retain exact RAW identities. Display demosaicing only; no model training or promotion.");
    c.png(&out.join(if rank == 0 {
        format!("anatomy-{id}.png")
    } else {
        format!("anatomy-{id}-rank-{}.png", rank + 1)
    }))
}
pub fn review(args: &[String]) -> Result<()> {
    if args.len() != 3 {
        return Err("roi-anatomy-review COMPLETED_ANATOMY_RUN NEW_OUT".into());
    }
    let source = Path::new(&args[1]);
    let summary: Value = serde_json::from_slice(&fs::read(source.join("summary.json"))?)?;
    if summary["complete"] != true {
        return Err("completed anatomy run required".into());
    }
    let out = data::output(&args[2])?;
    let mut bundles = BTreeMap::new();
    let rows = rows(&source.join("frames.jsonl"))?;
    for row in &rows {
        let input = read_input(row, &mut bundles)?;
        let rgb = model_input(&input, row["input_adapter"].as_str().unwrap_or("quad_rgb"))?;
        if data::digest(&rgb) != row["sam_input_rgb_sha256"] {
            return Err("RAW-to-SAM adapter changed since saved inference".into());
        }
        render_input(&out, n(&row["record"]), &rgb)?;
        let details = row["prompts"].as_array().ok_or("prompt records")?;
        for rank in 0..3 {
            let mut masks = vec![];
            for detail in details {
                let m = &detail["candidates"][rank];
                let mask = if let Some(path) = m["mask"].as_str() {
                    let bytes = fs::read(source.join(path))?;
                    if data::digest(&bytes) != m["mask_sha256"]
                        || bytes.len() != sam_native::TW * sam_native::TH
                    {
                        return Err("saved anatomy mask hash/shape mismatch".into());
                    }
                    Some(bytes)
                } else {
                    None
                };
                masks.push(mask);
            }
            render(&out, n(&row["record"]), &input, &masks, details, rank)?;
        }
    }
    data::write(
        out.join("review.json"),
        &json!({"complete":true,"source":source,"frames":rows.len(),"ranks_per_frame":3,"raw_mask_and_adapter_hashes_verified":true}),
    )?;
    Ok(())
}

/// A fixed-prompt, fixed-rank matched view. No sign or model score chooses a
/// different prompt between images. The source RAW, frame metadata and model
/// inputs are checked before any mask can enter the comparison.
pub fn compare(args: &[String]) -> Result<()> {
    if args.len() != 6 {
        return Err("roi-anatomy-compare RUN_A RUN_B RUN_C NEW_OUT PROMPT_INDEX".into());
    }
    let prompt_index: usize = args[5].parse()?;
    if prompt_index >= 6 {
        return Err("prompt index must be 0..5".into());
    }
    let mut runs = vec![];
    let mut metadata = vec![];
    for path in &args[1..4] {
        let p = Path::new(path);
        let summary: Value = serde_json::from_slice(&fs::read(p.join("summary.json"))?)?;
        if summary["complete"] != true {
            return Err("completed anatomy runs required".into());
        }
        let bytes = fs::read(p.join("frames.jsonl"))?;
        let records = rows(&p.join("frames.jsonl"))?;
        metadata.push(json!({"path":path,"summary":summary,"frames_sha256":data::digest(&bytes)}));
        runs.push(records);
    }
    if runs[0].is_empty() || runs.iter().any(|r| r.len() != runs[0].len()) {
        return Err("comparison needs identical nonempty source subsets".into());
    }
    let prompt = runs[0][0]["prompts"][prompt_index]["prompt"]
        .as_str()
        .ok_or("prompt")?;
    if metadata.iter().any(|m| {
        m["summary"]["teacher"]["model_sha256"] != metadata[0]["summary"]["teacher"]["model_sha256"]
    }) {
        return Err("comparison requires identical SAM weights/export".into());
    }
    let out = data::output(&args[4])?;
    let mut bundles = BTreeMap::new();
    let mut manifest = vec![];
    for page in 0..runs[0].len().div_ceil(4) {
        let mut c = Canvas::new(1800, 1820)?;
        c.clear();
        c.text(
            20.,
            32.,
            25.,
            WHITE,
            &format!("Matched SAM anatomy inputs | prompt: {prompt} | highest-score proposal"),
        );
        c.text(20., 65., 18., MUTED, "Identical area-admitted RAW and pinned model. Cyan is a predicted boundary, not a human label or sign decision.");
        for (k, m) in metadata.iter().enumerate() {
            c.text(
                20. + k as f64 * 594.,
                105.,
                22.,
                WHITE,
                m["summary"]["input_adapter"].as_str().unwrap_or("quad_rgb"),
            );
        }
        for row_index in page * 4..((page + 1) * 4).min(runs[0].len()) {
            let source = &runs[0][row_index];
            let input = read_input(source, &mut bundles)?;
            let (w, h, bgra) = raw_bgra(&input);
            for (k, run) in runs.iter().enumerate() {
                let row = &run[row_index];
                for key in [
                    "record",
                    "raw_sha256",
                    "frame",
                    "source",
                    "epoch",
                    "eye",
                    "sequence",
                    "source_ns",
                    "area_admitted_providers",
                ] {
                    if row[key] != source[key] {
                        return Err(format!("comparison source mismatch: {key}").into());
                    }
                }
                let adapter = row["input_adapter"].as_str().unwrap_or("quad_rgb");
                let rgb = model_input(&input, adapter)?;
                if data::digest(&rgb) != row["sam_input_rgb_sha256"] {
                    return Err("input transform hash mismatch".into());
                }
                let detail = &row["prompts"][prompt_index];
                if detail["prompt"] != prompt {
                    return Err("comparison prompt mismatch".into());
                }
                let candidate = &detail["candidates"][0];
                let x = 20. + k as f64 * 594.;
                let y = 160. + (row_index % 4) as f64 * 411.;
                c.text(
                    x,
                    y - 12.,
                    18.,
                    WHITE,
                    &format!(
                        "RAW {} | eye {} | score {:.3}",
                        row["record"],
                        row["eye"],
                        candidate["score"].as_f64().unwrap_or(0.)
                    ),
                );
                c.image(&bgra, w, h, x, y, 570., 380.);
                if let Some(path) = candidate["mask"].as_str() {
                    let mask = fs::read(Path::new(&args[k + 1]).join(path))?;
                    let (mw, mh) = (sam_native::TW, sam_native::TH);
                    if data::digest(&mask) != candidate["mask_sha256"] || mask.len() != mw * mh {
                        return Err("mask hash/shape mismatch".into());
                    }
                    for yy in 1..mh - 1 {
                        for xx in 1..mw - 1 {
                            let i = yy * mw + xx;
                            if mask[i] >= 128
                                && [i - 1, i + 1, i - mw, i + mw]
                                    .into_iter()
                                    .any(|j| mask[j] < 128)
                            {
                                c.dot(
                                    x + (xx as f64 + 0.5) * 570. / mw as f64,
                                    y + (yy as f64 + 0.5) * 380. / mh as f64,
                                    1.2,
                                    CYAN,
                                    true,
                                );
                            }
                        }
                    }
                }
                manifest.push(json!({"page":page,"record":row["record"],"raw_sha256":row["raw_sha256"],"adapter":adapter,"candidate":candidate,"sam_input_rgb_sha256":row["sam_input_rgb_sha256"]}));
            }
        }
        c.png(&out.join(format!("comparison-{}.png", page + 1)))?;
    }
    data::write(
        out.join("comparison.json"),
        &json!({"complete":true,"metadata":metadata,"frames":runs[0].len(),"rows":manifest,"prompt_index":prompt_index,"prompt":prompt,"anatomical_accuracy":null,"physical_sign_truth":null}),
    )?;
    Ok(())
}
pub fn run(args: &[String]) -> Result<()> {
    if !(4..=6).contains(&args.len()) {
        return Err("roi-anatomy AREA_FIRST_DIR ANATOMY_SAM_EXPORT NEW_OUT [PILOT_LIMIT] [quad_rgb|gray|gray-bilateral-v1]".into());
    }
    let area = Path::new(&args[1]);
    let export = Path::new(&args[2]);
    let out = data::output(&args[3])?;
    let limit = args
        .get(4)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let adapter = args.get(5).map(String::as_str).unwrap_or("quad_rgb");
    if !["quad_rgb", "gray", "gray-bilateral-v1"].contains(&adapter) {
        return Err("unknown anatomy input adapter".into());
    }
    let summary: Value = serde_json::from_slice(&fs::read(area.join("summary.json"))?)?;
    if summary["schema"] != "buttercup-area-first-focus-v1" || summary["complete"] != true {
        return Err("complete area-first run required".into());
    }
    let classes = rows(&area.join("classifications.jsonl"))?;
    let sources = classes
        .iter()
        .filter(|r| r["class"] == "multiple")
        .map(|r| n(&r["source"]))
        .collect::<BTreeSet<_>>();
    let retained = rows(&area.join("retained-inputs.jsonl"))?;
    let mut unique = BTreeMap::new();
    let mut providers = BTreeMap::<u64, Vec<String>>::new();
    for row in retained
        .into_iter()
        .filter(|r| sources.contains(&n(&r["source"])))
    {
        if row["area_admission"]["accepted"] != true {
            return Err("non-admitted RAW in anatomy input".into());
        }
        let id = n(&row["record"]);
        providers
            .entry(id)
            .or_default()
            .push(row["provider"].as_str().unwrap().to_string());
        if let Some(old) = unique.insert(id, row.clone()) {
            let old: Value = old;
            if old["raw_sha256"] != row["raw_sha256"] || old["frame"] != row["frame"] {
                return Err("providers disagree about source identity".into());
            }
        }
    }
    let (receipt, prompts, model) = load_cpu_export(export)?;
    data::write(
        out.join("inputs.json"),
        &json!({"area_summary_sha256":data::digest(&fs::read(area.join("summary.json"))?),"teacher":receipt,"device":"cpu","input_adapter":adapter,"scope":"fresh permitted SAM diagnostic; area-admitted source selection, no custom model training or promotion"}),
    )?;
    fs::create_dir(out.join("masks"))?;
    let step = if limit > 0 {
        unique.len().div_ceil(limit)
    } else {
        1
    };
    let mut bundles = BTreeMap::new();
    let mut writer = BufWriter::new(fs::File::create(out.join("frames.jsonl"))?);
    let mut index = vec![];
    let started = Instant::now();
    let _guard = tch::no_grad_guard();
    for (i, (id, row)) in unique.iter().enumerate().filter(|(i, _)| i % step == 0) {
        let t = Instant::now();
        let input = read_input(row, &mut bundles)?;
        let f = &row["frame"];
        let hash = &input.hash;
        let rgb = model_input(&input, adapter)?;
        let rgb_hash = data::digest(&rgb);
        if index.len() < 12 {
            render_input(&out, *id, &rgb)?;
        }
        let (scores, logits) = infer_scores_masks(&model, &rgb)?;
        let mut details = vec![];
        let mut displayed = vec![];
        for (k, prompt) in prompts.iter().enumerate() {
            let values = Vec::<f32>::try_from(scores.get(k as i64).to_kind(Kind::Float))?;
            let mut ranked = (0..values.len())
                .filter(|&q| values[q].is_finite() && values[q] >= 0.05)
                .collect::<Vec<_>>();
            ranked.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
            let mut candidates = vec![];
            let mut top = None;
            for q in ranked.into_iter().take(12) {
                let p = logits
                    .get(k as i64)
                    .get(q as i64)
                    .to_kind(Kind::Float)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .upsample_bilinear2d(
                        [sam_native::TH as i64, sam_native::TW as i64],
                        false,
                        None,
                        None,
                    )
                    .sigmoid();
                let p = (p * 255.).round().to_kind(Kind::Uint8).contiguous();
                let mut mask = vec![0u8; sam_native::TW * sam_native::TH];
                let size = mask.len();
                p.copy_data(&mut mask, size);
                let area = mask.iter().filter(|&&v| v >= 128).count();
                if area < 64 {
                    continue;
                }
                let name = format!("masks/{id}-p{k}-q{q}.u8");
                fs::write(out.join(&name), &mask)?;
                candidates.push(json!({"query":q,"score":values[q],"mask":name,"mask_sha256":data::digest(&mask),"width":sam_native::TW,"height":sam_native::TH,"area_fraction":area as f64/mask.len() as f64}));
                if top.is_none() {
                    top = Some(mask);
                }
                if candidates.len() == 3 {
                    break;
                }
            }
            displayed.push(top);
            details.push(json!({"prompt":prompt,"candidates":candidates}));
        }
        let record = json!({"record":id,"source":row["source"],"epoch":row["epoch"],"eye":row["eye"],"sequence":row["sequence"],"source_ns":row["source_ns"],"raw_source":row["raw_source"],"stream_entry":row["stream_entry"],"raw_sha256":hash,"input_adapter":adapter,"sam_input_rgb_sha256":rgb_hash,"frame":f,"area_admitted_providers":providers[id],"prompts":details,"seconds":t.elapsed().as_secs_f64()});
        serde_json::to_writer(&mut writer, &record)?;
        writeln!(writer)?;
        writer.flush()?;
        if index.len() < 12 {
            render(&out, *id, &input, &displayed, &details, 0)?;
            index.push(json!({"record":id,"image":format!("anatomy-{id}.png")}));
        }
        eprintln!(
            "SAM ANATOMY record={id} source-row={i}/{} seconds={:.2}",
            unique.len(),
            t.elapsed().as_secs_f64()
        );
    }
    data::write(out.join("review.json"), &index)?;
    data::write(
        out.join("summary.json"),
        &json!({"complete":true,"unique_admitted_sources":unique.len(),"sampling_stride":step,"evaluated":unique.len().div_ceil(step),"seconds":started.elapsed().as_secs_f64(),"device":"cpu","input_adapter":adapter,"prompts":prompts,"teacher":receipt,"executable_sha256":sam_export::hash(Path::new("/proc/self/exe"))?,"no_new_training":true,"anatomical_accuracy":null}),
    )?;
    Ok(())
}
