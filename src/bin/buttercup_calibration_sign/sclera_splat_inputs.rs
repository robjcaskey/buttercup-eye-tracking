//! Fresh, source-bound SAM proposals for a bounded CPU surface-map experiment.
//! Selection reads acquisition metadata only, never archived model predictions.
use super::*;
use crate::bootstrapability as boot;
#[path = "../../sclera_splat_input_recipe.rs"]
mod recipe;

pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 6 {
        return Err(
            "sclera-splat-inputs BUNDLE CPU_EXPORT NEW_OUT SKIP_PER_EYE COUNT_PER_EYE (2..24)"
                .into(),
        );
    }
    let skip: usize = args[4].parse()?;
    let count: usize = args[5].parse()?;
    if !(2..=24).contains(&count) {
        return Err("bounded count must be 2..24 per eye".into());
    }
    let bundle = BundleSource::open(Path::new(&args[1]))?;
    let mut seen = [0; 2];
    let mut inputs = Vec::new();
    for line in String::from_utf8(bundle.read_entry("frames.jsonl")?)?.lines() {
        let frame: Value = serde_json::from_str(line)?;
        let eye = n(&frame["eye_id"]);
        if !(1..=2).contains(&eye) {
            continue;
        }
        let index = seen[eye as usize - 1];
        seen[eye as usize - 1] += 1;
        if index < skip || index >= skip + count {
            continue;
        }
        if n(&frame["width"]) != 420
            || n(&frame["height"]) != 280
            || frame["pixel_format"] != "RAW10_LE40_1X1"
        {
            return Err("pilot requires native 420x280 RAW10 ROIs".into());
        }
        let bytes = bundle.read_range(
            frame["stream"].as_str().ok_or("stream")?,
            n(&frame["offset"]),
            n(&frame["length"]) as usize,
        )?;
        inputs.push(sam_native::Input {
            raw: raw10::try_unpack_raw10(&bytes, 420, 280, n(&frame["stride"]) as usize)?,
            frame,
            hash: data::digest(&bytes),
        });
    }
    if inputs.len() != count * 2 {
        return Err("requested complete paired interval unavailable".into());
    }
    let out = data::output(&args[3])?;
    let source = boot::current_source(Path::new("."))?;
    let (teacher, prompts, model) = load_cpu_export(Path::new(&args[2]))?;
    let k = prompts
        .iter()
        .position(|s| s == "exposed white sclera")
        .ok_or("export must include literal exposed white sclera prompt")?;
    let inventory = json!({"bundle":fs::canonicalize(&args[1])?,"skip_per_eye":skip,"count_per_eye":count,
        "frames":inputs.iter().map(|i| json!({"source":i.frame,"raw_sha256":i.hash})).collect::<Vec<_>>()});
    data::write(out.join("raw-inventory.json"), &inventory)?;
    let graph = json!({"schema":boot::SCHEMA,"source":source,"targets":["masks"],"nodes":[
        {"id":"source","kind":"source","sha256":source.tree_sha256,"dependencies":[]},
        {"id":"raw","kind":"raw","sha256":data::digest(&fs::read(out.join("raw-inventory.json"))?),"dependencies":[]},
        {"id":"sam31","kind":"sam3","sha256":sam_export::CHECKPOINT_SHA,"dependencies":[]},
        {"id":"export","kind":"export","sha256":teacher["model_sha256"],"dependencies":["source","sam31"]},
        {"id":"masks","kind":"derived_data","planned":true,"sha256":null,"dependencies":["source","raw","export"]}]});
    let manifest: boot::Manifest = serde_json::from_value(graph.clone())?;
    let certificate =
        boot::validate(&manifest, &source).map_err(|e| format!("sclera preflight: {e:?}"))?;
    data::write(out.join("bootstrap-graph.json"), &graph)?;
    data::write(out.join("bootstrap-preflight.json"), &certificate)?;
    data::write(
        out.join("inputs.json"),
        &json!({"teacher":teacher,"source":source,"device":"cpu","bundle":inventory["bundle"],"selection":"fixed contiguous acquisition interval, both eyes, no prediction-based sample selection","prompt":prompts[k],"input_adapter":"native-preview-100","custom_ancestors":[],"cache_role":"verified pinned SAM export; not a new cold export proof"}),
    )?;
    let started = Instant::now();
    let _guard = tch::no_grad_guard();
    let mut rows = BufWriter::new(fs::File::create(out.join("frames.jsonl"))?);
    for (i, input) in inputs.iter().enumerate() {
        let pixels = preview::color_preview(
            &input.raw,
            420,
            280,
            n(&input.frame["sensor_x"]) as u32,
            n(&input.frame["sensor_y"]) as u32,
            100,
            None,
        );
        let rgb = (0..3)
            .flat_map(|ch| pixels.iter().map(move |p| (p >> (16 - ch * 8)) as u8))
            .collect::<Vec<_>>();
        let (scores, logits) = infer_scores_masks(&model, &rgb)?;
        let mut alternatives = vec![];
        let bgra = (0..420 * 280)
            .flat_map(|j| [rgb[2 * 420 * 280 + j], rgb[420 * 280 + j], rgb[j], 255])
            .collect::<Vec<_>>();
        let mut comparison = Canvas::new(1300, 680)?;
        comparison.clear();
        comparison.text(
            15.,
            28.,
            21.,
            WHITE,
            "Six fixed sclera prompts | highest query each | same native RAW",
        );
        for (j, prompt) in prompts.iter().enumerate() {
            let values = Vec::<f32>::try_from(scores.get(j as i64).to_kind(Kind::Float))?;
            let q = (0..values.len())
                .filter(|&q| values[q].is_finite())
                .max_by(|&a, &b| values[a].total_cmp(&values[b]))
                .ok_or("no finite query")?;
            let p = logits
                .get(j as i64)
                .get(q as i64)
                .unsqueeze(0)
                .unsqueeze(0)
                .upsample_bilinear2d([280, 420], false, None, None)
                .sigmoid();
            let p = (p * 255.).round().to_kind(Kind::Uint8).contiguous();
            let mut mask = vec![0u8; 420 * 280];
            p.copy_data(&mut mask, 420 * 280);
            let name = format!("{i:03}-prompt-{j}.u8");
            fs::write(out.join(&name), &mask)?;
            alternatives.push(json!({"prompt":prompt,"query":q,"score":values[q],"mask":name,"sha256":data::digest(&mask),"area":mask.iter().filter(|&&p|p>=179).count()}));
            let mut view = bgra.clone();
            for l in 0..mask.len() {
                if mask[l] < 179 {
                    for ch in 0..3 {
                        view[4 * l + ch] /= 6;
                    }
                }
            }
            let x = 10. + (j % 3) as f64 * 430.;
            let y = 70. + (j / 3) as f64 * 310.;
            comparison.text(
                x,
                y - 12.,
                15.,
                WHITE,
                &format!("{prompt} | {:.3}", values[q]),
            );
            comparison.image(&view, 420, 280, x, y, 420., 280.);
        }
        comparison.png(&out.join(format!("{i:03}-prompts.png")))?;
        let scores = Vec::<f32>::try_from(scores.get(k as i64).to_kind(Kind::Float))?;
        let q = (0..scores.len())
            .filter(|&q| scores[q].is_finite())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .ok_or("no finite SAM score")?;
        let probability = logits
            .get(k as i64)
            .get(q as i64)
            .unsqueeze(0)
            .unsqueeze(0)
            .upsample_bilinear2d([280, 420], false, None, None)
            .sigmoid();
        let probability = (probability * 255.)
            .round()
            .to_kind(Kind::Uint8)
            .contiguous();
        let mut mask = vec![0; 420 * 280];
        probability.copy_data(&mut mask, 420 * 280);
        let mask_name = format!("{i:03}-mask.u8");
        let rgb_name = format!("{i:03}-rgb.u8");
        fs::write(out.join(&mask_name), &mask)?;
        fs::write(out.join(&rgb_name), &rgb)?;
        let row = json!({"source":input.frame,"raw_sha256":input.hash,"mask":mask_name,"mask_sha256":data::digest(&mask),"rgb":rgb_name,"rgb_sha256":data::digest(&rgb),"prompt":prompts[k],"query":q,"score":scores[q],"mask_area":mask.iter().filter(|&&m|m>=179).count(),"mask_role":"unreviewed SAM proposal, not an anatomical label","input_adapter":"native-preview-100","alternatives":alternatives});
        serde_json::to_writer(&mut rows, &row)?;
        writeln!(rows)?;
        rows.flush()?;
        let mut c = Canvas::new(880, 390)?;
        c.clear();
        c.text(
            18.,
            29.,
            21.,
            WHITE,
            &format!(
                "Sclera map inputs | eye {} | sequence {}",
                input.frame["eye_id"], input.frame["sequence"]
            ),
        );
        c.text(
            18.,
            56.,
            15.,
            MUTED,
            "Native RAW preview | fixed SAM sclera proposal; dark regions excluded",
        );
        let bgra = (0..420 * 280)
            .flat_map(|j| [rgb[2 * 420 * 280 + j], rgb[420 * 280 + j], rgb[j], 255])
            .collect::<Vec<_>>();
        let mut masked = bgra.clone();
        for j in 0..mask.len() {
            if mask[j] < 179 {
                for ch in 0..3 {
                    masked[4 * j + ch] /= 6;
                }
            }
        }
        c.image(&bgra, 420, 280, 15., 75., 420., 280.);
        c.image(&masked, 420, 280, 445., 75., 420., 280.);
        c.text(
            18.,
            378.,
            14.,
            MUTED,
            &format!(
                "Fixed highest-score query {q}, score {:.3}; score is not anatomical accuracy.",
                scores[q]
            ),
        );
        c.png(&out.join(format!("{i:03}-input.png")))?;
        eprintln!(
            "SCLERA INPUT {}/{} {:.1}s",
            i + 1,
            inputs.len(),
            started.elapsed().as_secs_f64()
        );
    }
    if boot::current_source(Path::new("."))? != source {
        return Err("source changed during sclera preparation".into());
    }
    data::write(
        out.join("summary.json"),
        &json!({"schema":"buttercup-sclera-splat-inputs-v1","complete":true,"frames":inputs.len(),"seconds":started.elapsed().as_secs_f64(),"source":source,"preparation_recipe_sha256":recipe::stamp()?,"frames_sha256":data::digest(&fs::read(out.join("frames.jsonl"))?),"device":"cpu","executable_sha256":sam_export::hash(Path::new("/proc/self/exe"))?}),
    )?;
    Ok(())
}
