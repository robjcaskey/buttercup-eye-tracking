#[cfg(feature = "sam31")]
#[path = "../sam31_text.rs"]
mod sam31_text;

#[cfg(feature = "sam31")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use tch::{Kind, Tensor};

    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() < 3 || arguments.len() > 5 {
        return Err(
            "usage: buttercup_sam31_prompt_bundle CHECKPOINT BPE_GZ OUTPUT [EXPECTED_OUTER_BF16|--arc-trial|--prompt-file FILE|--replace-outer-file FILE]"
                .into(),
        );
    }
    let checkpoint = PathBuf::from(&arguments[0]);
    let bpe = PathBuf::from(&arguments[1]);
    let output = PathBuf::from(&arguments[2]);
    let arc_trial = arguments
        .get(3)
        .is_some_and(|argument| argument == "--arc-trial");
    let prompt_file = arguments
        .get(3)
        .is_some_and(|argument| argument == "--prompt-file")
        .then(|| arguments.get(4))
        .flatten();
    let replacement_outer_file = arguments
        .get(3)
        .is_some_and(|argument| argument == "--replace-outer-file")
        .then(|| arguments.get(4))
        .flatten();
    if arguments.len() == 5 && prompt_file.is_none() && replacement_outer_file.is_none() {
        return Err("five arguments require --prompt-file or --replace-outer-file FILE".into());
    }
    let custom_prompts = if let Some(path) = prompt_file {
        let source = std::fs::read_to_string(path)?;
        let lines = source
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect::<Vec<_>>();
        if lines.is_empty() {
            return Err("the prompt file has no non-comment prompt lines".into());
        }
        let mut prompts = Vec::with_capacity(lines.len() + 1);
        prompts.push(sam31_text::SEMANTIC_PROMPTS[0]);
        for (index, line) in lines.into_iter().enumerate() {
            let key = Box::leak(format!("prompt_step_{}", index + 1).into_boxed_str());
            let label = Box::leak(format!("PROMPT STEP {}", index + 1).into_boxed_str());
            let text = Box::leak(line.to_string().into_boxed_str());
            prompts.push(sam31_text::SemanticPrompt { key, label, text });
        }
        Some(prompts)
    } else {
        None
    };
    let replacement_prompts = if let Some(path) = replacement_outer_file {
        let text = std::fs::read_to_string(path)?.trim().to_string();
        if text.is_empty() {
            return Err("replacement outer prompt is empty".into());
        }
        let replacement = sam31_text::SemanticPrompt {
            key: "outer_iris_custom",
            label: "CUSTOM OUTER IRIS PROMPT",
            text: Box::leak(text.into_boxed_str()),
        };
        let mut prompts = sam31_text::SEMANTIC_PROMPTS.to_vec();
        prompts[0] = replacement;
        Some(prompts)
    } else {
        None
    };
    let prompts = if let Some(prompts) = replacement_prompts.as_deref() {
        prompts
    } else if let Some(prompts) = custom_prompts.as_deref() {
        prompts
    } else if arc_trial {
        &sam31_text::ARC_TRIAL_PROMPTS[..]
    } else {
        &sam31_text::SEMANTIC_PROMPTS[..]
    };
    let mut bundle = sam31_text::encode_prompt_set(&checkpoint, &bpe, prompts)?;
    let preserve_outer = arc_trial || prompt_file.is_some();
    if preserve_outer {
        // CUDA BF16 GEMMs can round very slightly differently when the text
        // batch grows from the six live prompts to the larger trial batch.
        // Preserve the already-reviewed live anchor exactly so only follow-on
        // wording changes between the two pipelines.
        let canonical_path = output.with_file_name("sam31_semantic_prompts_cuda_bf16.pt");
        let canonical = sam31_text::load_prompt_bundle(&canonical_path)?;
        bundle.language_features = Tensor::cat(
            &[
                canonical.language_features.narrow(1, 0, 1),
                bundle
                    .language_features
                    .narrow(1, 1, prompts.len() as i64 - 1),
            ],
            1,
        );
        bundle.language_mask = Tensor::cat(
            &[
                canonical.language_mask.narrow(0, 0, 1),
                bundle.language_mask.narrow(0, 1, prompts.len() as i64 - 1),
            ],
            0,
        );
        bundle.token_ids = Tensor::cat(
            &[
                canonical.token_ids.narrow(0, 0, 1),
                bundle.token_ids.narrow(0, 1, prompts.len() as i64 - 1),
            ],
            0,
        );
        println!("outer_anchor=exact canonical live feature");
    }
    for (index, prompt) in prompts.iter().enumerate() {
        let tokens = bundle.token_ids.get(index as i64);
        let non_padding = tokens.ne(0).sum(Kind::Int64).int64_value(&[]);
        println!(
            "prompt={index} key={} tokens={non_padding} text={:?}",
            prompt.key, prompt.text
        );
    }
    if let Some(expected_path) = arguments
        .get(3)
        .filter(|_| !preserve_outer && replacement_outer_file.is_none())
    {
        let bytes = std::fs::read(expected_path)?;
        let expected_bytes = 32 * 256 * 2;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "expected outer feature tensor is {} bytes, not {expected_bytes}",
                bytes.len()
            )
            .into());
        }
        let expected = Tensor::from_data_size(&bytes, &[32, 1, 256], Kind::BFloat16);
        let actual = bundle.language_features.narrow(1, 0, 1);
        let actual_f32 = actual.to_kind(Kind::Float);
        let expected_f32 = expected.to_kind(Kind::Float);
        let absolute = (&actual_f32 - &expected_f32).abs();
        let cosine = (&actual_f32 * &expected_f32).sum(Kind::Float)
            / (actual_f32.square().sum(Kind::Float).sqrt()
                * expected_f32.square().sum(Kind::Float).sqrt());
        println!(
            "outer_reference mean_abs={:.9} max_abs={:.9} cosine={:.9} actual_mean={:.9} actual_std={:.9} expected_mean={:.9} expected_std={:.9}",
            absolute.mean(Kind::Float).double_value(&[]),
            absolute.max().double_value(&[]),
            cosine.double_value(&[]),
            actual_f32.mean(Kind::Float).double_value(&[]),
            actual_f32.std(true).double_value(&[]),
            expected_f32.mean(Kind::Float).double_value(&[]),
            expected_f32.std(true).double_value(&[]),
        );
    }
    sam31_text::save_prompt_bundle(&bundle, &output)?;
    println!("wrote={}", output.display());
    Ok(())
}

#[cfg(not(feature = "sam31"))]
fn main() {
    eprintln!("rebuild with --features sam31");
    std::process::exit(2);
}
