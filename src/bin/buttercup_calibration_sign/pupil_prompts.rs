//! Matched RAW/blur prompt diagnostics: no training or anatomical oracle.
use super::{canvas::*, data, n, rows, sam_export, sam_native, Result};
use buttercup_eye_tracking::recorded_bundle::BundleSource;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
use tch::Kind;
#[path = "sclera_arcs.rs"]
pub mod sclera_arcs;

const SEED: &str = "pupil-prompt-blur-20260918-v1";
const W: usize = sam_native::TW;
const H: usize = sam_native::TH;
const PROMPTS: [&str; 6] = [
    "exact center point of the pupil",
    "center of the pupil",
    "pupil center",
    "black center of the eye",
    "pupil",
    "dark circular pupil",
];
const BLACK_PUPIL_PROMPTS: [&str; 6] = [
    "normally black part of the pupil",
    "black pupil",
    "dark pupil opening",
    "pupil aperture",
    "black opening inside the iris",
    "pupil excluding light reflections",
];
const SKIN_PROMPTS: [&str; 6] = [
    "skin",
    "skin around the eye",
    "facial skin",
    "eyelid skin",
    "skin below the eye",
    "skin above the eye",
];
const EYE_CORNER_PROMPTS: [&str; 6] = [
    "outer corner of the eye",
    "lateral canthus",
    "eye corner farthest from the nose",
    "inner corner of the eye",
    "medial canthus",
    "eye corner nearest the nose",
];
const VEIN_REGION_PROMPTS: [&str; 6] = [
    "area around tiny tiny veins on the eyeball",
    "sclera around tiny blood vessels",
    "white of the eye with tiny veins",
    "eye surface around fine red veins",
    "sclera containing fine blood vessels",
    "white tissue around tiny eye veins",
];
const LOWER_LID_PROMPTS: [&str; 6] = [
    "lower eyelid",
    "lower eyelid margin",
    "lower eyelid arc",
    "lower eyelid waterline",
    "curved edge of the lower eyelid",
    "boundary between the eyeball and lower eyelid",
];
const SCLERA_PROMPTS: [&str; 6] = [
    "white of the sclera",
    "white of the eye",
    "sclera",
    "visible sclera",
    "exposed white sclera",
    "white sclera between iris and eyelids",
];

fn gray(rgb: &[u8]) -> Vec<f64> {
    (0..W * H)
        .map(|i| {
            (77. * rgb[i] as f64 + 150. * rgb[W * H + i] as f64 + 29. * rgb[2 * W * H + i] as f64)
                / 256.
        })
        .collect()
}

fn smooth(p: &[f64], sigma: f64) -> Vec<f64> {
    let radius = (sigma * 3.).ceil() as isize;
    let kernel: Vec<_> = (-radius..=radius)
        .map(|d| (-(d * d) as f64 / (2. * sigma * sigma)).exp())
        .collect();
    let mass: f64 = kernel.iter().sum();
    let mut mid = vec![0.; W * H];
    let mut out = mid.clone();
    for y in 0..H {
        for x in 0..W {
            for (k, &v) in kernel.iter().enumerate() {
                let xx = (x as isize + k as isize - radius).clamp(0, W as isize - 1) as usize;
                mid[y * W + x] += p[y * W + xx] * v / mass;
            }
        }
    }
    for y in 0..H {
        for x in 0..W {
            for (k, &v) in kernel.iter().enumerate() {
                let yy = (y as isize + k as isize - radius).clamp(0, H as isize - 1) as usize;
                out[y * W + x] += mid[yy * W + x] * v / mass;
            }
        }
    }
    out
}

/// Contrast-normalized gradient energy on a fixed central crop. Smoothing
/// suppresses sensor/CFA grain; this is a ranking proxy, not an optical blur label.
fn sharpness(rgb: &[u8]) -> f64 {
    let p = smooth(&gray(rgb), 1.5);
    let (mut sum, mut sum2, mut gradient, mut count) = (0., 0., 0., 0.);
    for y in H / 8..H * 7 / 8 {
        for x in W / 8..W * 7 / 8 {
            let i = y * W + x;
            sum += p[i];
            sum2 += p[i] * p[i];
            count += 1.;
            gradient += ((p[i + 1] - p[i - 1]).powi(2) + (p[i + W] - p[i - W]).powi(2)) / 4.;
        }
    }
    (gradient / count) / (sum2 / count - (sum / count).powi(2)).max(1.)
}

fn blur_rgb(rgb: &[u8]) -> Vec<u8> {
    (0..3)
        .flat_map(|c| {
            let p: Vec<_> = rgb[c * W * H..(c + 1) * W * H]
                .iter()
                .map(|&v| v as f64)
                .collect();
            smooth(&p, 3.)
                .into_iter()
                .map(|v| v.round().clamp(0., 255.) as u8)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn bgra(rgb: &[u8]) -> Vec<u8> {
    (0..W * H)
        .flat_map(|i| [rgb[2 * W * H + i], rgb[W * H + i], rgb[i], 255])
        .collect()
}

fn input_for(row: &Value, bundles: &mut BTreeMap<String, BundleSource>) -> Result<Vec<u8>> {
    let input = super::read_input(row, bundles)?;
    let rgb = super::model_input(&input, "quad_rgb")?;
    Ok(if row["added_blur_sigma_px"] == 3. {
        blur_rgb(&rgb)
    } else {
        rgb
    })
}

fn random_key(row: &Value, group: &str, seed: &str) -> String {
    data::digest(format!("{seed}:{group}:{}", row["raw_sha256"]).as_bytes())
}

fn sample(
    pool: &[Value],
    group: &str,
    used: &mut BTreeSet<String>,
    seed: &str,
) -> Result<Vec<Value>> {
    let mut pool = pool.to_vec();
    pool.sort_by_cached_key(|r| random_key(r, group, seed));
    let mut sources = BTreeSet::new();
    let mut out = vec![];
    // Prefer different recordings. A second pass permits another exposure
    // from a source only if fewer than six distinct sources are available.
    for distinct in [true, false] {
        for row in &pool {
            let hash = row["raw_sha256"].as_str().ok_or("RAW hash")?.to_string();
            if used.contains(&hash) || (distinct && sources.contains(&n(&row["source"]))) {
                continue;
            }
            used.insert(hash);
            sources.insert(n(&row["source"]));
            let mut row = row.clone();
            row["group"] = json!(group);
            row["sample_id"] = json!(format!("{group}-{}", out.len() + 1));
            out.push(row);
            if out.len() == 6 {
                return Ok(out);
            }
        }
    }
    Err("fewer than six distinct RAW exposures in sampling pool".into())
}

pub fn select(args: &[String]) -> Result<()> {
    if !(4..=5).contains(&args.len())
        || !["natural", "matched", "matched-random"].contains(&args[3].as_str())
    {
        return Err(
            "pupil-prompt-select AREA_FIRST NEW_OUT natural|matched|matched-random [SEED]".into(),
        );
    }
    let seed = args.get(4).map(String::as_str).unwrap_or(SEED);
    if seed.trim().is_empty() {
        return Err("sample seed must not be empty".into());
    }
    let area = Path::new(&args[1]);
    let summary: Value = serde_json::from_slice(&fs::read(area.join("summary.json"))?)?;
    if summary["schema"] != "buttercup-area-first-focus-v1" || summary["complete"] != true {
        return Err("completed area-first input required".into());
    }
    let out = data::output(&args[2])?;
    let mut unique = BTreeMap::new();
    for mut row in rows(&area.join("retained-inputs.jsonl"))? {
        // Pinned SAM is the only model dependency of sample admission.
        if row["provider"] != "sam" || row["area_admission"]["accepted"] != true {
            continue;
        }
        row.as_object_mut().ok_or("row")?.remove("fit");
        unique
            .entry(row["raw_sha256"].as_str().ok_or("RAW hash")?.to_string())
            .or_insert(row);
    }
    let mut bundles = BTreeMap::new();
    let mut scored = vec![];
    for (i, mut row) in unique.into_values().enumerate() {
        if args[3] != "matched-random" {
            let rgb = input_for(&row, &mut bundles)?;
            row["sharpness_proxy"] = json!(sharpness(&rgb));
        }
        scored.push(row);
        if args[3] != "matched-random" && i % 200 == 0 {
            eprintln!("PUPIL SAMPLE scored={}", i + 1);
        }
    }
    if args[3] != "matched-random" {
        scored.sort_by(|a, b| {
            a["sharpness_proxy"]
                .as_f64()
                .unwrap()
                .total_cmp(&b["sharpness_proxy"].as_f64().unwrap())
        });
    }
    if scored.len() < 30 {
        return Err("at least thirty area-admitted RAW inputs required".into());
    }
    let quintile = scored.len() / 5;
    let mut used = BTreeSet::new();
    let mut chosen = if args[3] == "natural" {
        let mut low = sample(&scored[..quintile], "blurry", &mut used, seed)?;
        low.extend(sample(
            &scored[scored.len() - quintile..],
            "sharp",
            &mut used,
            seed,
        )?);
        low
    } else {
        let pool = if args[3] == "matched-random" {
            &scored[..]
        } else {
            &scored[scored.len() - quintile..]
        };
        let high = sample(pool, "sharp", &mut used, seed)?;
        let mut low = high.clone();
        for row in &mut low {
            row["group"] = json!("blurry");
            row["sample_id"] = json!(row["sample_id"]
                .as_str()
                .unwrap()
                .replace("sharp", "blurry"));
            row["added_blur_sigma_px"] = json!(3.);
        }
        low.extend(high);
        low
    };
    for row in &mut chosen {
        let rgb = input_for(row, &mut bundles)?;
        row["sam_input_rgb_sha256"] = json!(data::digest(&rgb));
        fs::write(
            out.join(format!("{}.rgb8", row["sample_id"].as_str().unwrap())),
            rgb,
        )?;
    }
    data::write(
        out.join("selection.json"),
        &json!({
            "schema":"buttercup-pupil-prompt-selection-v1", "complete":true, "mode":args[3],
            "seed":seed, "pool_count":scored.len(), "quintile_count":quintile,
            "area_inputs_sha256":sam_export::hash(&area.join("retained-inputs.jsonl"))?,
            "executable_sha256":sam_export::hash(Path::new("/proc/self/exe"))?,
            "sampling":if args[3] == "matched-random" { "SHA256 shuffle across all SAM area-admitted unique RAW inputs; prefer six distinct recordings; originals paired with Gaussian blur; before prompt inference" } else { "SHA256 shuffle within bottom/top quintile of fixed-crop contrast-normalized gradient energy; prefer six distinct recordings per group; before prompt inference" },
            "sharpness_caveat":if args[3] == "matched-random" { "not computed: random sampling does not require decoding or scoring the unselected pool" } else { "relative image sharpness proxy, not measured optical blur; pose, noise and lighting can also change this score" },
            "input_adapter":"existing quad_rgb, same single RAW repeated only for adapter shape; 420x280 RGB, antialiased bilinear 1008 square, normalize (x-.5)/.5",
            "rows":chosen, "scores":scored
        }),
    )?;
    data::write(out.join("prompts.json"), &PROMPTS)?;
    let mut c = Canvas::new(3 * 440 + 20, 4 * 335 + 100)?;
    c.clear();
    c.text(
        20.,
        32.,
        24.,
        WHITE,
        "Frozen random selection | rows 1-2 softer, rows 3-4 originals",
    );
    c.text(20., 60., 17., MUTED, if args[3] == "natural" { "Unchanged recorded images; bottom/top sharpness quintiles, no SAM prompt results used." } else { "Matched originals / Gaussian sigma 3 px copies; exact model inputs before its 1008 px resize." });
    for (i, row) in chosen.iter().enumerate() {
        let x = 20. + (i % 3) as f64 * 440.;
        let y = 112. + (i / 3) as f64 * 335.;
        let rgb = input_for(row, &mut bundles)?;
        c.text(
            x,
            y - 12.,
            17.,
            WHITE,
            &format!(
                "{} | source {} | eye {} | seq {}",
                row["sample_id"].as_str().unwrap(),
                n(&row["source"]),
                n(&row["eye"]),
                n(&row["sequence"])
            ),
        );
        c.image(&bgra(&rgb), W, H, x, y, W as f64, H as f64);
    }
    c.png(&out.join("selection.png"))?;
    eprintln!(
        "PUPIL SELECT DONE: {} RAW candidates, {} samples",
        scored.len(),
        chosen.len()
    );
    Ok(())
}

fn render_sheet(out: &Path, records: &[Value], title: &str, filename: &str) -> Result<()> {
    let diagnostic = records
        .first()
        .and_then(|r| r["diagnostic"].as_str())
        .unwrap_or("pupil");
    let region_mask = diagnostic != "pupil";
    let arcs = diagnostic == "sclera-arcs";
    let mut headers = vec![vec![
        "Source / model input".to_string(),
        "no extra display processing".to_string(),
    ]];
    for p in records.first().ok_or("empty contact sheet")?["prompts"]
        .as_array()
        .ok_or("prompt headers")?
    {
        let prompt = p["prompt"].as_str().ok_or("prompt text")?;
        let mut lines = vec![String::new()];
        for word in prompt.split_whitespace() {
            // Also split unusually long tokens so columns never share text.
            let chars: Vec<_> = word.chars().collect();
            for part in chars.chunks(29) {
                if !lines.last().unwrap().is_empty()
                    && lines.last().unwrap().chars().count() + part.len() + 1 > 29
                {
                    lines.push(String::new());
                }
                let line = lines.last_mut().unwrap();
                if !line.is_empty() {
                    line.push(' ');
                }
                line.extend(part.iter().copied());
            }
        }
        headers.push(lines);
    }
    let extra = headers
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(2)
        .saturating_sub(2)
        * 28;
    let mut c = Canvas::new(7 * 432 + 24, records.len() * 348 + 190 + extra)?;
    c.clear();
    c.text(18., 35., 30., WHITE, title);
    c.text(18., 66., 21., MUTED, if arcs {
        "Gray: SAM boundary. Amber upper / cyan lower: dots are mask samples, lines are fitted arcs; dashed = no nearby support. Gray dots rejected."
    } else if region_mask {
        "Cyan: SAM mask boundary. Highest-score query, including failures. No fitted anatomical curve. Score is not an accuracy estimate."
    } else {
        "Cyan: SAM mask boundary. Pink +: computed mask centroid (not a SAM point output). Score is not an accuracy estimate."
    });
    for (col, lines) in headers.iter().enumerate() {
        let x = 18. + col as f64 * 432.;
        for (line, text) in lines.iter().enumerate() {
            c.text(x, 104. + line as f64 * 28., 23., WHITE, text);
        }
    }
    for (r, row) in records.iter().enumerate() {
        let rgb = fs::read(out.join(row["rgb"].as_str().ok_or("rgb")?))?;
        if data::digest(&rgb) != row["sam_input_rgb_sha256"] {
            return Err("contact sheet RGB changed".into());
        }
        let image = bgra(&rgb);
        let y = 175. + extra as f64 + r as f64 * 348.;
        c.text(
            18.,
            y - 14.,
            18.,
            WHITE,
            &format!(
                "{} | S{} eye{} seq{}",
                row["sample_id"].as_str().unwrap(),
                n(&row["source"]),
                n(&row["eye"]),
                n(&row["sequence"])
            ),
        );
        for col in 0..7 {
            let x = 18. + col as f64 * 432.;
            c.image(&image, W, H, x, y, W as f64, H as f64);
            if col == 0 {
                continue;
            }
            let p = &row["prompts"][col - 1];
            if let Some(mask_path) = p["mask"].as_str() {
                let mask = fs::read(out.join(mask_path))?;
                if mask.len() != W * H || data::digest(&mask) != p["mask_sha256"] {
                    return Err("mask changed".into());
                }
                for yy in 0..H {
                    for xx in 0..W {
                        let i = yy * W + xx;
                        if mask[i] >= 128
                            && (xx == 0
                                || yy == 0
                                || xx + 1 == W
                                || yy + 1 == H
                                || [i - 1, i + 1, i - W, i + W].iter().any(|&j| mask[j] < 128))
                        {
                            c.dot(
                                x + xx as f64 + 0.5,
                                y + yy as f64 + 0.5,
                                0.8,
                                if arcs { MUTED } else { CYAN },
                                true,
                            );
                        }
                    }
                }
                if let Some(center) = p["centroid_xy"].as_array().filter(|_| !region_mask) {
                    let (cx, cy) = (
                        x + center[0].as_f64().unwrap() + 0.5,
                        y + center[1].as_f64().unwrap() + 0.5,
                    );
                    c.dot(cx, cy, 7., [0., 0., 0.], true);
                    c.cross(cx, cy, 6., PINK);
                }
            }
            if arcs {
                c.clipped(x, y, W as f64, H as f64, |c| {
                    sclera_arcs::draw(c, x, y, &p["arcs"])
                });
                c.text(
                    x,
                    y + H as f64 + 49.,
                    16.,
                    MUTED,
                    &sclera_arcs::status(&p["arcs"]),
                );
            }
            let score = p["score"].as_f64().unwrap();
            let label = if p["area_pixels"] == 0 {
                format!("No mask | top score {score:.3}")
            } else {
                format!(
                    "{}score {score:.3} | {} px",
                    if score < 0.5 { "LOW " } else { "" },
                    n(&p["area_pixels"])
                )
            };
            c.text(
                x,
                y + H as f64 + 27.,
                19.,
                if score < 0.5 { ORANGE } else { WHITE },
                &label,
            );
        }
    }
    c.png(&out.join(filename))
}

pub fn run(args: &[String]) -> Result<()> {
    if !(4..=5).contains(&args.len()) {
        return Err(
            "roi-prompt-run SELECTION_DIR CPU_EXPORT NEW_OUT [pupil|black-pupil|skin|eye-corners|vein-regions|lower-eyelid-arc|sclera]"
                .into(),
        );
    }
    let diagnostic = args.get(4).map(String::as_str).unwrap_or("pupil");
    let (expected, subject) = match diagnostic {
        "pupil" => (&PROMPTS, "pupil"),
        "black-pupil" => (&BLACK_PUPIL_PROMPTS, "black pupil region"),
        "skin" => (&SKIN_PROMPTS, "skin"),
        "eye-corners" => (&EYE_CORNER_PROMPTS, "outer / inner eye corners"),
        "vein-regions" => (&VEIN_REGION_PROMPTS, "areas around tiny eyeball veins"),
        "lower-eyelid-arc" => (&LOWER_LID_PROMPTS, "lower-eyelid arc"),
        "sclera" => (&SCLERA_PROMPTS, "sclera"),
        _ => return Err(
            "unknown prompt diagnostic; use pupil, black-pupil, skin, eye-corners, vein-regions, lower-eyelid-arc or sclera"
                .into(),
        ),
    };
    let selection_dir = Path::new(&args[1]);
    run_prompts(
        selection_dir,
        Path::new(&args[2]),
        &args[3],
        expected,
        subject,
        diagnostic,
    )
}

pub(super) fn run_prompts(
    selection_dir: &Path,
    export_dir: &Path,
    new_out: &str,
    expected: &[&str],
    subject: &str,
    diagnostic: &str,
) -> Result<()> {
    let selection: Value =
        serde_json::from_slice(&fs::read(selection_dir.join("selection.json"))?)?;
    if selection["schema"] != "buttercup-pupil-prompt-selection-v1" || selection["complete"] != true
    {
        return Err("completed selection required".into());
    }
    let chosen = selection["rows"].as_array().ok_or("selection rows")?;
    if chosen.len() != 12 {
        return Err("exactly twelve selected inputs required".into());
    }
    let (receipt, prompts, model) = super::load_cpu_export(export_dir)?;
    if prompts != expected {
        return Err("literal diagnostic prompt wording mismatch".into());
    }
    let out = data::output(new_out)?;
    fs::create_dir(out.join("masks"))?;
    let mut bundles = BTreeMap::new();
    let mut records = vec![];
    let mut writer = BufWriter::new(fs::File::create(out.join("frames.jsonl"))?);
    let started = Instant::now();
    let _guard = tch::no_grad_guard();
    for row in chosen {
        let t = Instant::now();
        let id = row["sample_id"].as_str().ok_or("sample id")?;
        let rgb = input_for(row, &mut bundles)?;
        if data::digest(&rgb) != row["sam_input_rgb_sha256"] {
            return Err("frozen model input changed".into());
        }
        fs::write(out.join(format!("{id}.rgb8")), &rgb)?;
        let (scores, logits) = super::infer_scores_masks(&model, &rgb)?;
        let mut details = vec![];
        for (k, prompt) in prompts.iter().enumerate() {
            let values = Vec::<f32>::try_from(scores.get(k as i64).to_kind(Kind::Float))?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite SAM score".into());
            }
            // Always show the highest-scored query, even if empty or low score.
            // Do not choose a nicer-looking alternative, threshold tiny masks,
            // or turn a mask centroid into a model-native point prediction.
            let q = (0..values.len())
                .max_by(|&a, &b| values[a].total_cmp(&values[b]))
                .ok_or("empty SAM queries")?;
            let p = logits
                .get(k as i64)
                .get(q as i64)
                .to_kind(Kind::Float)
                .unsqueeze(0)
                .unsqueeze(0)
                .upsample_bilinear2d([H as i64, W as i64], false, None, None)
                .sigmoid();
            if p.isfinite().all().int64_value(&[]) != 1 {
                return Err("nonfinite SAM mask".into());
            }
            let p = (p * 255.).round().to_kind(Kind::Uint8).contiguous();
            let mut mask = vec![0u8; W * H];
            let count = mask.len();
            p.copy_data(&mut mask, count);
            let mut area = 0usize;
            let mut center = [0.; 2];
            for (i, &v) in mask.iter().enumerate() {
                if v >= 128 {
                    area += 1;
                    center[0] += (i % W) as f64;
                    center[1] += (i / W) as f64;
                }
            }
            let centroid = (area > 0).then(|| [center[0] / area as f64, center[1] / area as f64]);
            let filename = format!("masks/{id}-p{k}-q{q}.u8");
            fs::write(out.join(&filename), &mask)?;
            details.push(json!({"prompt":prompt,"query":q,"score":values[q],"all_query_scores":values,"mask":filename,"mask_sha256":data::digest(&mask),"area_pixels":area,"centroid_xy":centroid,"centroid_role":"unweighted arithmetic mean of thresholded SAM mask pixel centers, not SAM point output or anatomical truth"}));
        }
        let mut record = row.clone();
        record["diagnostic"] = json!(diagnostic);
        record["prompts"] = json!(details);
        record["rgb"] = json!(format!("{id}.rgb8"));
        record["seconds"] = json!(t.elapsed().as_secs_f64());
        serde_json::to_writer(&mut writer, &record)?;
        writeln!(writer)?;
        writer.flush()?;
        records.push(record);
        eprintln!(
            "ROI PROMPTS {}/12 {id} seconds={:.2}",
            records.len(),
            t.elapsed().as_secs_f64()
        );
    }
    render_all(&out, &records, &selection, subject)?;
    let mut table = vec![];
    for (k, prompt) in prompts.iter().enumerate() {
        for group in ["blurry", "sharp"] {
            let p: Vec<_> = records
                .iter()
                .filter(|r| r["group"] == group)
                .map(|r| &r["prompts"][k])
                .collect();
            table.push(json!({"prompt":prompt,"group":group,"nonempty":p.iter().filter(|v| n(&v["area_pixels"]) > 0).count(),"score_at_least_05_nonempty":p.iter().filter(|v| n(&v["area_pixels"]) > 0 && v["score"].as_f64().unwrap() >= 0.5).count(),"mean_score":p.iter().map(|v| v["score"].as_f64().unwrap()).sum::<f64>() / p.len() as f64}));
        }
    }
    data::write(
        out.join("summary.json"),
        &json!({"complete":true,"schema":"buttercup-roi-prompt-diagnostic-v1","diagnostic":diagnostic,"selection_sha256":sam_export::hash(&selection_dir.join("selection.json"))?,"selection":selection,"prompts":prompts,"teacher":receipt,"executable_sha256":sam_export::hash(Path::new("/proc/self/exe"))?,"device":"cpu","seconds":started.elapsed().as_secs_f64(),"results":table,"anatomical_accuracy":null,"new_training":false}),
    )?;
    let legend = if diagnostic == "pupil" {
        "Pink crosses are mask centroids computed afterward, not model-native point predictions."
    } else {
        "No centroid marker or fitted arc is drawn: a SAM region boundary does not establish anatomical identity."
    };
    fs::write(
        out.join("README.md"),
        format!(
            r#"# SAM3.1 {subject} wording comparison

Twelve images from the frozen ROI comparison selection (seed `{seed}`); matched modes pair six originals with Gaussian-blurred copies (sigma 3 adapter pixels). Selection uses SAM area-consistent inputs and prefers distinct recordings. This corpus contains one person's recordings. Mode: {mode}. Sampling policy: {sampling}.

The six exact prompts are recorded in summary.json and column headings. The highest-scored query is shown for each prompt, including empty and low-score results. Cyan is its threshold-0.5 mask boundary. {legend} Scores are uncalibrated model scores, not localization accuracy. No reviewed anatomical labels are available for this comparison.

Images are the exact existing quad_rgb adapter inputs before model resize to 1008 square and normalization. Their hashes must match the frozen selection. No point prompt, gaze target, fitted ellipse or calibration selects a response. Fresh inference uses the pinned official SAM3.1 checkpoint's documented FP32 CPU export, without claiming bit-equivalence to BF16 CUDA. No training.

`contact-sheet.png` contains all twelve rows, each original immediately above its blurred copy in matched mode. `contact-sheet-page-1/2.png` split this into two pages; `pair-1..6.png` provide closeups. Separate blurry and sharp sheets are retained. `frames.jsonl` records all 200 query scores per prompt, top-query masks, mask centroids, original RAW locations/hashes and input hashes.

Reproduce with new output paths and the documented LibTorch/resource-coordination setup:

    buttercup_calibration_sign sam-export-anatomy-cpu data/models/sam31_multiplex.pt NEW_EXPORT SIX_PROMPTS_JSON
    buttercup_calibration_sign roi-prompt-run SELECTION_DIR NEW_EXPORT NEW_RESULTS {diagnostic}
"#,
            mode = selection["mode"].as_str().unwrap_or("unknown"),
            seed = selection["seed"].as_str().unwrap_or("unknown"),
            sampling = selection["sampling"].as_str().unwrap_or("unknown")
        ),
    )?;
    eprintln!("ROI PROMPTS DONE {}", out.display());
    Ok(())
}

fn render_all(out: &Path, records: &[Value], selection: &Value, subject: &str) -> Result<()> {
    render_sheet(
        out,
        &records[..6],
        &if selection["mode"] == "natural" {
            format!("SAM3.1 {subject} prompts | six naturally softer recorded ROIs")
        } else {
            format!("SAM3.1 {subject} prompts | six matched blurred copies (Gaussian sigma 3 px)")
        },
        "blurry-contact-sheet.png",
    )?;
    render_sheet(
        out,
        &records[6..],
        &format!("SAM3.1 {subject} prompts | six original ROIs"),
        "sharp-contact-sheet.png",
    )?;
    let display = if selection["mode"] != "natural" {
        let mut paired = vec![];
        for i in 0..6 {
            let pair = [records[6 + i].clone(), records[i].clone()];
            render_sheet(
                out,
                &pair,
                &format!(
                    "SAM3.1 {subject} prompts | pair {} | original above, blurred below",
                    i + 1
                ),
                &format!("pair-{}.png", i + 1),
            )?;
            paired.extend(pair);
        }
        paired
    } else {
        records.to_vec()
    };
    render_sheet(
        out,
        &display,
        &if selection["mode"] != "natural" {
            format!(
                "SAM3.1 {subject} prompts | six pairs, original then blurred | fixed random sample"
            )
        } else {
            format!("SAM3.1 {subject} prompts | blurry first, then sharp | fixed random sample")
        },
        "contact-sheet.png",
    )?;
    for (page, chunk) in display.chunks(6).enumerate() {
        render_sheet(
            out,
            chunk,
            &format!(
                "SAM3.1 {subject} prompts | {} | page {}",
                if selection["mode"] == "natural" {
                    "natural softer / sharper"
                } else {
                    "matched original / blur"
                },
                page + 1
            ),
            &format!("contact-sheet-page-{}.png", page + 1),
        )?;
    }
    Ok(())
}
