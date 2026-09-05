#![allow(dead_code)]
#![recursion_limit = "256"]

#[cfg(feature = "sam31")]
#[path = "../raw10.rs"]
mod raw10;
#[cfg(feature = "sam31")]
#[path = "../sam31_outer.rs"]
mod sam31_outer;
#[cfg(feature = "sam31")]
#[path = "../sam31_text.rs"]
mod sam31_text;

#[cfg(feature = "sam31")]
mod enabled {
    use super::{raw10, sam31_outer, sam31_text};
    use sam31_outer::{Ellipse, ProposalMask, RawFrame, SemanticProposalMasks};
    use serde_json::{json, Value};
    use std::collections::{BTreeSet, HashSet};
    use std::ffi::OsStr;
    use std::fs;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::Arc;

    const OUTER: usize = 0;
    const VISIBLE_IRIS: usize = 1;
    const ADJACENT_SCLERA: usize = 2;
    const VISIBLE_ARCS: usize = 3;
    const OCCLUDERS: usize = 4;
    const LEFT_SLICE: usize = 5;
    const RIGHT_SLICE: usize = 6;
    const UPPER_OCCLUSION: usize = 7;
    const LOWER_OCCLUSION: usize = 8;
    const OUTPUT_WIDTH: usize = 1536;
    const OUTPUT_HEIGHT: usize = 640;
    const PANEL_SCALE: usize = 2;
    const PANEL_TOP: usize = 64;
    const FRAMES_PER_STRATEGY: usize = 48;
    const FRAMES_PER_REJECTION: usize = 36;
    const OUTPUT_FPS: usize = 30;
    const PROMPT_LAB_DISPLAY_CANDIDATES: usize = 4;
    const VIDEO_FEATURE_DEFAULT_MAX_FRAMES: usize = 192;

    #[derive(Clone)]
    struct LoadedRaw {
        source: PathBuf,
        frame: Arc<RawFrame>,
    }

    #[derive(Clone)]
    struct RejectedSamMaskReview {
        sequence: u64,
        preview: Vec<[u8; 3]>,
        width: usize,
        height: usize,
        mask_width: usize,
        mask_height: usize,
        mask: ProposalMask,
        geometry_accepted: bool,
        elapsed_ms: u64,
    }

    #[derive(Clone)]
    struct PromptLabCandidate {
        query: usize,
        score: f32,
        area_fraction: f64,
        segment_roundness: f64,
        crust_score: f64,
        /// May independently enter the disk/ellipse fitter. Evidence stages
        /// deliberately never receive this capability.
        eligible: bool,
        /// May be shown as supporting boundary evidence. Stages two and three
        /// have no lower area floor, while retaining the 50% upper bound.
        display_eligible: bool,
        evidence_only: bool,
        mask_width: usize,
        mask_height: usize,
        mask: ProposalMask,
        fitted: Option<Ellipse>,
        fit_points: Arc<Vec<(f64, f64)>>,
        label_boundary_mean_px: Option<f64>,
        label_boundary_rms_px: Option<f64>,
        label_boundary_max_px: Option<f64>,
        label_fit_mean_px: Option<f64>,
        label_fit_rms_px: Option<f64>,
        label_fit_max_px: Option<f64>,
    }

    #[derive(Clone)]
    struct PromptLabStep {
        prompt_index: usize,
        selected_query: Option<usize>,
        returned_candidates: usize,
        candidates: Vec<PromptLabCandidate>,
        all_candidates: Vec<PromptLabCandidate>,
    }

    #[derive(Clone)]
    struct PromptLabRow {
        source: String,
        sequence: u64,
        sensor_origin: (u32, u32),
        logical_history_sequences: Vec<u64>,
        preview: Vec<[u8; 3]>,
        conditioning_preview: Option<Vec<[u8; 3]>>,
        width: usize,
        height: usize,
        steps: Vec<PromptLabStep>,
        video_feature_shapes: Option<sam31_outer::VideoFeatureShapes>,
        elapsed_ms: u64,
        label_path: Option<String>,
        visible_label_points: usize,
    }

    #[derive(Clone, Debug)]
    struct RankedMask {
        query: usize,
        objective: f64,
        model_score: f32,
        area_fraction: f64,
    }

    #[derive(Clone)]
    struct MaskField {
        width: usize,
        height: usize,
        pixels: Vec<u8>,
        selected: Vec<RankedMask>,
    }

    impl MaskField {
        fn empty() -> Self {
            Self {
                width: 0,
                height: 0,
                pixels: Vec::new(),
                selected: Vec::new(),
            }
        }

        fn sample(&self, x: f64, y: f64, source_width: usize, source_height: usize) -> f64 {
            if self.width == 0
                || self.height == 0
                || self.pixels.len() != self.width * self.height
                || !x.is_finite()
                || !y.is_finite()
                || x < 0.0
                || y < 0.0
                || x >= source_width as f64
                || y >= source_height as f64
            {
                return 0.0;
            }
            let low_x = ((x + 0.5) * self.width as f64 / source_width as f64 - 0.5)
                .round()
                .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
            let low_y = ((y + 0.5) * self.height as f64 / source_height as f64 - 0.5)
                .round()
                .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;
            f64::from(self.pixels[low_y * self.width + low_x] != 0)
        }
    }

    #[derive(Clone, Debug)]
    struct ArcEvidence {
        point: (f64, f64),
        phase: f64,
        raw_order: f64,
        visible_iris: f64,
        adjacent_sclera: f64,
        direct_arc: f64,
        occluder: f64,
        pizza_slice: f64,
        upper_lower_occlusion: f64,
    }

    #[derive(Clone)]
    struct Strategy {
        name: &'static str,
        follow_on_prompts: &'static [usize],
        scores: Vec<f64>,
        trusted: Vec<bool>,
        fitted: Option<Ellipse>,
        coverage: f64,
        sector_coverage: f64,
        mean_score: f64,
        fit_residual_px: Option<f64>,
        internal_quality: f64,
        label_mean_px: Option<f64>,
        label_rms_px: Option<f64>,
        label_max_px: Option<f64>,
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
        if arguments
            .first()
            .is_some_and(|value| value == "--rejected-review")
        {
            if arguments.len() < 3 {
                return Err(
                    "usage: buttercup-sam31-arc-trial --rejected-review OUTPUT.mkv \
                     RAW10_OR_CAPTURE@SEQUENCE@EYE_LABEL [...]"
                        .into(),
                );
            }
            arguments.remove(0);
            let output = PathBuf::from(arguments.remove(0));
            let frames = arguments
                .iter()
                .map(|argument| load_rejection_review_input(argument))
                .collect::<Result<Vec<_>, _>>()?;
            render_rejection_review_video(&output, &frames)?;
            println!("rejected_review={}", output.display());
            return Ok(());
        }
        if arguments
            .first()
            .is_some_and(|value| value == "--rejected-sam-review")
        {
            if arguments.len() < 3 {
                return Err(
                    "usage: buttercup-sam31-arc-trial --rejected-sam-review OUTPUT.mkv \
                     RAW10_OR_CAPTURE@SEQUENCE@EYE_LABEL [...]"
                        .into(),
                );
            }
            arguments.remove(0);
            let output = PathBuf::from(arguments.remove(0));
            let model = environment_path(
                "BUTTERCUP_SAM31_MODEL",
                "data/models/sam31_semantic_dynamic_u8.pt",
            );
            let prompt_bundle = environment_path(
                "BUTTERCUP_SAM31_ARC_PROMPT_BUNDLE",
                "data/models/sam31_arc_trial_prompts_cuda_bf16.pt",
            );
            let outer_prompt_bundle = environment_path(
                "BUTTERCUP_SAM31_PROMPT_BUNDLE",
                "data/models/sam31_semantic_prompts_cuda_bf16.pt",
            );
            for required in [&model, &prompt_bundle, &outer_prompt_bundle] {
                if !required.is_file() {
                    return Err(format!(
                        "SAM31 runtime input is unavailable: {}",
                        required.display()
                    )
                    .into());
                }
            }
            let mut reviews = Vec::with_capacity(arguments.len());
            for argument in &arguments {
                reviews.push(run_rejected_sam_mask(
                    &model,
                    &outer_prompt_bundle,
                    &prompt_bundle,
                    argument,
                )?);
            }
            render_rejected_sam_review_video(&output, &reviews)?;
            println!("rejected_sam_review={}", output.display());
            return Ok(());
        }
        if arguments
            .first()
            .is_some_and(|value| value == "--prompt-lab")
        {
            if arguments.len() != 5 {
                return Err(
                    "usage: buttercup-sam31-arc-trial --prompt-lab OUTPUT_DIR PROMPT_FILE MANIFEST PROMPT_BUNDLE"
                        .into(),
                );
            }
            arguments.remove(0);
            let output = PathBuf::from(arguments.remove(0));
            let prompt_file = PathBuf::from(arguments.remove(0));
            let manifest = PathBuf::from(arguments.remove(0));
            let prompt_bundle = PathBuf::from(arguments.remove(0));
            run_prompt_lab(&output, &prompt_file, &manifest, &prompt_bundle)?;
            return Ok(());
        }
        if arguments
            .first()
            .is_some_and(|value| value == "--video-features")
        {
            if arguments.len() != 3 {
                return Err(
                    "usage: buttercup-sam31-arc-trial --video-features REPORT.json RAW10_OR_CAPTURE@SEQUENCE@EYE_LABEL"
                        .into(),
                );
            }
            arguments.remove(0);
            let report = PathBuf::from(arguments.remove(0));
            let input = arguments.remove(0);
            run_video_feature_report(&report, &input)?;
            return Ok(());
        }
        if arguments.len() < 2 {
            return Err(
                "usage: buttercup-sam31-arc-trial OUTPUT_DIR TARGET.raw10 [OLDER.raw10 ... TARGET.raw10]\n\
                 or: buttercup-sam31-arc-trial OUTPUT_DIR --capture CAPTURE_DIR TARGET_SEQUENCE [EYE_LABEL]\n\
                 or: buttercup-sam31-arc-trial --rejected-review OUTPUT.mkv RAW10_OR_CAPTURE@SEQUENCE@EYE_LABEL [...]\n\
                 or: buttercup-sam31-arc-trial --rejected-sam-review OUTPUT.mkv RAW10_OR_CAPTURE@SEQUENCE@EYE_LABEL [...]\n\
                 or: buttercup-sam31-arc-trial --prompt-lab OUTPUT_DIR PROMPT_FILE MANIFEST PROMPT_BUNDLE\n\
                 Individual inputs are chronological native 384x256 RAW10 frames; a short history is left-padded."
                    .into(),
            );
        }
        let output = PathBuf::from(arguments.remove(0));
        fs::create_dir_all(&output)?;

        let model = environment_path(
            "BUTTERCUP_SAM31_MODEL",
            "data/models/sam31_semantic_dynamic_u8.pt",
        );
        let prompt_bundle = environment_path(
            "BUTTERCUP_SAM31_ARC_PROMPT_BUNDLE",
            "data/models/sam31_arc_trial_prompts_cuda_bf16.pt",
        );
        let outer_prompt_bundle = environment_path(
            "BUTTERCUP_SAM31_PROMPT_BUNDLE",
            "data/models/sam31_semantic_prompts_cuda_bf16.pt",
        );
        if !model.is_file() {
            return Err(format!("SAM31 model is unavailable: {}", model.display()).into());
        }
        if !prompt_bundle.is_file() {
            return Err(format!(
                "arc-trial prompt bundle is unavailable: {} (run scripts/run-sam31-arc-trial.sh so it can be generated natively)",
                prompt_bundle.display()
            )
            .into());
        }
        if !outer_prompt_bundle.is_file() {
            return Err(format!(
                "canonical live prompt bundle is unavailable: {}",
                outer_prompt_bundle.display()
            )
            .into());
        }

        let mut loaded = if arguments.first().is_some_and(|value| value == "--capture") {
            if !(3..=4).contains(&arguments.len()) {
                return Err("--capture requires CAPTURE_DIR TARGET_SEQUENCE [EYE_LABEL]".into());
            }
            let capture = PathBuf::from(&arguments[1]);
            let target_sequence = arguments[2]
                .to_str()
                .ok_or("target sequence is not UTF-8")?
                .parse::<u64>()?;
            let eye_label = arguments
                .get(3)
                .and_then(|value| value.to_str())
                .unwrap_or("subject-right");
            load_capture_history(&capture, target_sequence, eye_label)?
        } else {
            arguments
                .iter()
                .map(|path| load_raw(Path::new(path)))
                .collect::<Result<Vec<_>, _>>()?
        };
        if loaded.len() > sam31_outer::HISTORY_FRAMES {
            loaded = loaded.split_off(loaded.len() - sam31_outer::HISTORY_FRAMES);
        }
        while loaded.len() < sam31_outer::HISTORY_FRAMES {
            let first = loaded
                .first()
                .cloned()
                .ok_or("the RAW10 input list is empty")?;
            loaded.insert(0, first);
        }
        let frames = loaded
            .iter()
            .map(|loaded| Arc::clone(&loaded.frame))
            .collect::<Vec<_>>();
        let target = loaded.last().ok_or("missing target RAW10 frame")?;
        eprintln!(
            "SAM31 arc trial: target={} sequence={} history={} prompts={}",
            target.source.display(),
            target.frame.sequence,
            frames.len(),
            sam31_text::ARC_TRIAL_PROMPTS.len(),
        );

        let probe_only = std::env::var_os("BUTTERCUP_SAM31_ARC_PROBE").is_some();
        let prompt_indices = if probe_only {
            vec![OUTER]
        } else {
            (0..sam31_text::ARC_TRIAL_PROMPTS.len()).collect::<Vec<_>>()
        };
        let suite = sam31_outer::run_offline_semantic_suite(
            &model,
            &outer_prompt_bundle,
            &prompt_bundle,
            sam31_text::ARC_TRIAL_PROMPTS.len(),
            &frames,
            &prompt_indices,
        )?;
        if probe_only {
            let contextual_requested =
                std::env::var_os("BUTTERCUP_SAM31_FLAT_TIRE_RADIUS_SUPPORT").is_some();
            let contextual_fit = std::env::var("BUTTERCUP_SAM31_FLAT_TIRE_RADIUS_SUPPORT")
                .ok()
                .and_then(|support| {
                    let values = support
                        .split(',')
                        .map(str::trim)
                        .map(str::parse::<f64>)
                        .collect::<Result<Vec<_>, _>>()
                        .ok()?;
                    if values.len() != 3 {
                        return None;
                    }
                    let context = sam31_outer::OuterContourScaleContext::new(
                        values[0], values[1], values[2],
                    )?;
                    let outer = suite
                        .passes
                        .iter()
                        .find(|pass| pass.prompt_index == OUTER)?;
                    sam31_outer::contextual_outer_fit(outer, context)
                });
            let selected_fit = if contextual_requested {
                contextual_fit.as_ref()
            } else {
                suite.outer_fit.as_ref()
            };
            if let Some(fit) = selected_fit {
                println!(
                    "probe=accepted sequence={} center=({:.3},{:.3}) radii=({:.3},{:.3}) retained={} censored={}",
                    target.frame.sequence,
                    fit.ellipse.center.0,
                    fit.ellipse.center.1,
                    fit.ellipse.major_radius,
                    fit.ellipse.minor_radius,
                    fit.retained_points.len(),
                    fit.flat_tire_points.len(),
                );
            } else {
                println!("probe=rejected sequence={}", target.frame.sequence);
            }
            return Ok(());
        }
        let outer_fit = suite
            .outer_fit
            .as_ref()
            .ok_or("SAM31 did not produce a geometrically plausible outer-iris component")?;
        let outer = outer_fit.ellipse;

        let mut fields = Vec::with_capacity(sam31_text::ARC_TRIAL_PROMPTS.len());
        for prompt_index in 0..sam31_text::ARC_TRIAL_PROMPTS.len() {
            let pass = suite
                .passes
                .iter()
                .find(|pass| pass.prompt_index == prompt_index)
                .ok_or_else(|| format!("missing SAM31 answer for prompt {prompt_index}"))?;
            let field = build_mask_field(
                pass,
                prompt_index,
                outer,
                suite.source_width,
                suite.source_height,
            );
            eprintln!(
                "prompt={prompt_index} {:<24} selected={:?}",
                sam31_text::ARC_TRIAL_PROMPTS[prompt_index].label,
                field
                    .selected
                    .iter()
                    .map(|ranked| (
                        ranked.query,
                        ranked.objective,
                        ranked.model_score,
                        ranked.area_fraction
                    ))
                    .collect::<Vec<_>>()
            );
            fields.push(field);
        }

        let evidence = outer_fit
            .retained_points
            .iter()
            .copied()
            .map(|point| {
                arc_evidence(
                    point,
                    outer,
                    &fields,
                    &suite.source_raw,
                    suite.source_width,
                    suite.source_height,
                )
            })
            .collect::<Vec<_>>();
        let labels_path = locate_labels(&target.source);
        let labels = labels_path
            .as_deref()
            .map(read_visible_labels)
            .transpose()?
            .unwrap_or_default();
        let mut strategies = build_strategies(&evidence, outer, &labels);
        strategies.sort_by(|left, right| {
            right
                .internal_quality
                .total_cmp(&left.internal_quality)
                .then_with(|| left.name.cmp(right.name))
        });

        for strategy in &strategies {
            eprintln!(
                "strategy={:<21} follow_ons={} trusted={}/{} coverage={:.3} sectors={:.3} quality={:.3} label_mean={:?}",
                strategy.name,
                strategy.follow_on_prompts.len(),
                strategy.trusted.iter().filter(|&&trusted| trusted).count(),
                strategy.trusted.len(),
                strategy.coverage,
                strategy.sector_coverage,
                strategy.internal_quality,
                strategy.label_mean_px,
            );
        }

        let adapters = sam31_outer::diagnostic_quantized_adapters(&frames)?;
        let quad = adapters
            .iter()
            .find(|(name, _)| *name == "quad_rgb")
            .map(|(_, bytes)| bytes)
            .ok_or("SAM31 diagnostic adapter did not return Quad-RGB")?;
        let sam_color_preview = smooth_preview_chroma(
            &latest_quad_preview(quad, suite.source_width, suite.source_height)?,
            suite.source_width,
            suite.source_height,
        );
        let preview = clean_raw_display_preview(
            &suite.source_raw,
            &sam_color_preview,
            suite.source_width,
            suite.source_height,
        );
        let video_path = output.join("sam31-arc-trial-affine.mkv");
        render_review_video(
            &video_path,
            &preview,
            suite.source_width,
            suite.source_height,
            outer_fit,
            &fields,
            &evidence,
            &strategies[..1],
        )?;
        let comparison_video_path = output.join("sam31-arc-trial-comparison.mkv");
        render_review_video(
            &comparison_video_path,
            &preview,
            suite.source_width,
            suite.source_height,
            outer_fit,
            &fields,
            &evidence,
            &strategies,
        )?;

        let report = json!({
            "schema": "buttercup-sam31-arc-trial-v1",
            "source": target.source,
            "sequence": target.frame.sequence,
            "raw_contract": "native lossless RAW10 input; Quad-Bayer adapter is produced directly in Rust; no Python, JPEG, desktop capture, or downsampled source image",
            "history_sources": loaded.iter().map(|loaded| loaded.source.display().to_string()).collect::<Vec<_>>(),
            "model": model,
            "canonical_outer_prompt_bundle": outer_prompt_bundle,
            "prompt_bundle": prompt_bundle,
            "elapsed_ms": suite.elapsed_ms,
            "outer_fit": ellipse_json(outer),
            "outer_evidence": {
                "retained_points": outer_fit.retained_points.len(),
                "curvature_censored_points": outer_fit.flat_tire_points.len(),
                "upper_flat_tire": outer_fit.upper_flat_tire,
                "lower_flat_tire": outer_fit.lower_flat_tire,
            },
            "labels": labels_path,
            "prompts": sam31_text::ARC_TRIAL_PROMPTS.iter().enumerate().map(|(index, prompt)| json!({
                "index": index,
                "key": prompt.key,
                "label": prompt.label,
                "text": prompt.text,
                "selected_masks": fields[index].selected.iter().map(|mask| json!({
                    "query": mask.query,
                    "objective": mask.objective,
                    "model_score": mask.model_score,
                    "area_fraction": mask.area_fraction,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "strategies": strategies.iter().map(strategy_json).collect::<Vec<_>>(),
            "best_internal_strategy": strategies.first().map(|strategy| strategy.name),
            "review_video": video_path,
            "comparison_video": comparison_video_path,
        });
        let report_path = output.join("report.json");
        fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
        println!("report={}", report_path.display());
        println!("review={}", video_path.display());
        println!("comparison={}", comparison_video_path.display());
        Ok(())
    }

    fn run_video_feature_report(report: &Path, input: &OsStr) -> Result<(), String> {
        let model = environment_path(
            "BUTTERCUP_SAM31_MODEL",
            "data/models/sam31_semantic_video_features_u8.pt",
        );
        let prompt_bundle = environment_path(
            "BUTTERCUP_SAM31_PROMPT_BUNDLE",
            "data/models/sam31_semantic_prompts_cuda_bf16.pt",
        );
        let regime = std::env::var("BUTTERCUP_SAM31_PREPROCESS")
            .ok()
            .and_then(|value| sam31_outer::PreprocessRegime::parse(&value))
            .unwrap_or_default();
        let video_prompt_count = std::env::var("BUTTERCUP_SAM31_VIDEO_PROMPT_COUNT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(sam31_outer::SEMANTIC_PROMPT_COUNT);
        let video_prompt_index = std::env::var("BUTTERCUP_SAM31_VIDEO_PROMPT_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(sam31_outer::OUTER_IRIS_PROMPT);
        let mut loaded = load_rejection_review_history(input)?;
        while loaded.len() < sam31_outer::HISTORY_FRAMES {
            let first = loaded
                .first()
                .cloned()
                .ok_or("video feature history is empty")?;
            loaded.insert(0, first);
        }
        let sequences = loaded
            .iter()
            .map(|loaded| loaded.frame.sequence)
            .collect::<Vec<_>>();
        let frames = loaded
            .iter()
            .map(|loaded| Arc::clone(&loaded.frame))
            .collect::<Vec<_>>();
        let result = sam31_outer::run_offline_video_feature_sequence(
            &model,
            &prompt_bundle,
            &frames,
            regime,
        )?;
        let affine_review = report.with_extension("affine-review.mkv");
        render_video_feature_affine_review(&affine_review, &result.review_frames, regime.label())?;
        let human_limbus_evidence = result
            .review_frames
            .iter()
            .zip(loaded.iter())
            .filter_map(|(review, loaded)| {
                let label_path = locate_labels(&loaded.source)?;
                let labels = read_visible_labels(&label_path).ok()?;
                let metrics = review.mask.as_ref().and_then(|mask| {
                    mask_label_boundary_metrics(
                        mask,
                        review.mask_width,
                        review.mask_height,
                        review.source.width,
                        review.source.height,
                        &labels,
                    )
                });
                Some(json!({
                    "sequence": review.source.sequence,
                    "label_path": label_path,
                    "visible_label_points": labels.len(),
                    "mask_boundary_mean_error_px": metrics.map(|value| value.0),
                    "mask_boundary_rms_error_px": metrics.map(|value| value.1),
                    "mask_boundary_max_error_px": metrics.map(|value| value.2),
                    "production_fit_accepted": review.fit.is_some(),
                    "production_fit_mean_error_px": review.fit.as_ref().map(|fit| labels.iter().map(|&point| ellipse_residual_px(fit.ellipse, point)).sum::<f64>() / labels.len().max(1) as f64),
                }))
            })
            .collect::<Vec<_>>();
        let document = json!({
            "execution_contract": {
                "input": "independent native-aspect RAW10 ROI frames",
                "preprocess": regime.label(),
                "prompt_count": video_prompt_count,
                "prompt_index": video_prompt_index,
                "native_backbone_pyramid": true,
                "native_decoder_queries": true,
                "native_mask_memory_encoder": true,
                "native_temporal_memory_attention": true,
                "native_temporal_memory_slots": result.frame_count.saturating_sub(1).min(7),
                "native_propagation_mask_decoder": true,
                "native_object_pointer_projection": true,
                "native_object_pointers_in_temporal_attention": true,
                "native_object_pointer_slots": result.frame_count.saturating_sub(2).min(16),
                "native_occlusion_state": true,
                "native_conditioning_memory_retained": true,
                "native_recent_nonconditioning_memories": 6,
                "native_multiplex_lifecycle": "single tracked eye in slot zero plus fifteen learned empty slots",
                "scope": "The full native single-eye propagation tensor path is active, including detector conditioning, learned mask-memory encoding, temporal memory selection, object-pointer tokens, propagation decoding, hard absence, and bounded detector reconditioning. The eye occupies multiplex slot zero; the other fifteen slots use SAM3.1's learned empty-object embeddings because this diagnostic intentionally tracks one eye."
            },
            "source_sequences": sequences,
            "frame_count": result.frame_count,
            "elapsed_ms": result.elapsed_ms,
            "affine_review": affine_review,
            "flat_tire_reviews": result.review_frames.iter().map(|review| json!({
                "sequence": review.source.sequence,
                "accepted": review.fit.is_some(),
                "center": review.fit.as_ref().map(|fit| fit.ellipse.center),
                "major_radius_px": review.fit.as_ref().map(|fit| fit.ellipse.major_radius),
                "minor_radius_px": review.fit.as_ref().map(|fit| fit.ellipse.minor_radius),
                "fronto_parallel_area_px2": review.fit.as_ref().map(|fit| std::f64::consts::PI * fit.ellipse.major_radius.powi(2)),
                "projected_area_px2": review.fit.as_ref().map(|fit| std::f64::consts::PI * fit.ellipse.major_radius * fit.ellipse.minor_radius),
                "hypothesis_center": review.contour_hypothesis.as_ref().map(|fit| fit.ellipse.center),
                "hypothesis_radii": review.contour_hypothesis.as_ref().map(|fit| [fit.ellipse.major_radius, fit.ellipse.minor_radius]),
                "usable_contour_points": review.contour_hypothesis.as_ref().map(|fit| fit.retained_points.len()),
                "occlusion_contour_points": review.contour_hypothesis.as_ref().map(|fit| fit.flat_tire_points.len()),
                "raw_contour_points": review.raw_contour_points.len(),
                "persistent_hot_pixels": review.hot_pixels.iter().map(|&(x, y)| json!({
                    "roi_x": x,
                    "roi_y": y,
                    "sensor_x": review.source.sensor_x + x as u32,
                    "sensor_y": review.source.sensor_y + y as u32,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "human_limbus_evidence": human_limbus_evidence,
            "feature_shapes": {
                "pyramid": result.feature_shapes.pyramid,
                "decoder_queries": result.feature_shapes.decoder_queries,
                "mask_memory": result.mask_memory_shape,
                "mask_memory_spatial_position": result.mask_memory_position_shape,
                "temporal_conditioned": result.temporal_conditioned_shape,
            },
            "transitions": result.transitions.iter().map(|transition| json!({
                "from_frame": transition.from_frame,
                "to_frame": transition.to_frame,
                "from_sequence": sequences[transition.from_frame],
                "to_sequence": sequences[transition.to_frame],
                "pyramid_cosine": transition.pyramid_cosine,
                "pyramid_normalized_rms_change": transition.pyramid_normalized_rms_change,
                "decoder_query_cosine": transition.decoder_query_cosine,
                "decoder_query_normalized_rms_change": transition.decoder_query_normalized_rms_change,
                "source_selected_query": transition.source_selected_query,
                "target_detector_query": transition.target_detector_query,
                "matched_query": transition.matched_query,
                "matched_query_cosine": transition.matched_query_cosine,
                "matched_mask_iou_sensor": transition.matched_mask_iou_sensor,
                "mask_memory_cosine": transition.mask_memory_cosine,
                "mask_memory_normalized_rms_change": transition.mask_memory_normalized_rms_change,
                "temporal_conditioned_cosine": transition.temporal_conditioned_cosine,
                "temporal_conditioned_normalized_rms_change": transition.temporal_conditioned_normalized_rms_change,
                "temporal_update_cosine": transition.temporal_update_cosine,
                "temporal_update_normalized_rms_change": transition.temporal_update_normalized_rms_change,
                "tracker_object_score_logit": transition.tracker_object_score_logit,
                "tracker_object_present": transition.tracker_object_present,
                "tracker_reconditioned_from_detector": transition.tracker_reconditioned_from_detector,
                "detector_recondition_area_fraction": transition.detector_recondition_area_fraction,
                "tracker_selected_iou_score": transition.tracker_selected_iou_score,
                "tracker_mask_iou_sensor": transition.tracker_mask_iou_sensor,
                "tracker_vs_detector_mask_iou": transition.tracker_vs_detector_mask_iou,
                "tracker_area_fraction": transition.tracker_area_fraction,
                "tracker_equivalent_radius_px": transition.tracker_equivalent_radius_px,
                "tracker_radius_ratio_from_prior": transition.tracker_radius_ratio_from_prior,
                "tracker_centroid_motion_sensor_px": transition.tracker_centroid_motion_sensor_px,
            })).collect::<Vec<_>>(),
        });
        if let Some(parent) = report.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        fs::write(
            report,
            serde_json::to_vec_pretty(&document)
                .map_err(|error| format!("encode video feature report: {error}"))?,
        )
        .map_err(|error| format!("write {}: {error}", report.display()))?;
        println!("video_feature_report={}", report.display());
        println!("affine_review={}", affine_review.display());
        Ok(())
    }

    /// Lossless side-by-side temporal review. The left panel is the untouched
    /// RAW10 source with the admitted memory mask; the right panel de-affines
    /// the fitted ellipse to its equivalent-area circle and gently yaws/pitches
    /// that canonical plane. The motion makes the 3D affine assumption visible
    /// instead of presenting a potentially persuasive static fit.
    fn render_video_feature_affine_review(
        path: &Path,
        frames: &[sam31_outer::OfflineVideoReviewFrame],
        regime: &str,
    ) -> Result<(), String> {
        if frames.is_empty() {
            return Err("SAM31 video review has no frames".to_string());
        }
        let panel_width = sam31_outer::FRAME_WIDTH * 2;
        let width = panel_width * 3;
        let height = OUTPUT_HEIGHT;
        let header = 96usize;
        let scale = 2usize;
        let geometry = format!("{width}x{height}");
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &geometry,
                "-framerate",
                &OUTPUT_FPS.to_string(),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "rgb24",
                path.to_str()
                    .ok_or("video feature review path is not UTF-8")?,
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start video feature review ffmpeg: {error}"))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or("video feature review ffmpeg has no stdin")?;
        let display_frames_per_source = 8usize;
        for (source_index, review) in frames.iter().enumerate() {
            let blank = vec![[0u8; 3]; review.source.pixels.len()];
            let preview = clean_raw_display_preview(
                &review.source.pixels,
                &blank,
                review.source.width,
                review.source.height,
            );
            for local_frame in 0..display_frames_per_source {
                let constrained_review =
                    std::env::var_os("BUTTERCUP_SAM31_FLAT_TIRE_RADIUS_SUPPORT").is_some();
                let review_ellipse = review.fit.as_ref().map(|fit| fit.ellipse).or_else(|| {
                    (!constrained_review).then(|| {
                        review.mask.as_ref().and_then(|mask| {
                            mask_moment_review_ellipse(
                                mask,
                                review.mask_width,
                                review.mask_height,
                                review.source.width,
                                review.source.height,
                            )
                        })
                    })?
                });
                let phase =
                    std::f64::consts::TAU * local_frame as f64 / display_frames_per_source as f64;
                let view = [
                    [(0.38 * phase.sin()).cos(), 0.10 * phase.cos()],
                    [0.05 * phase.sin(), (0.22 * (phase + 0.6).sin()).cos()],
                ];
                let inverse_view = invert_matrix(view);
                let mut output = vec![[5u8, 8u8, 12u8]; width * height];
                draw_text_dynamic(
                    &mut output,
                    width,
                    height,
                    12,
                    10,
                    &format!(
                        "NATIVE SAM3.1 VIDEO  {}  SEQ {}  {}/{}",
                        regime,
                        review.source.sequence,
                        source_index + 1,
                        frames.len()
                    ),
                    2,
                    [235, 240, 245],
                );
                draw_text_dynamic(
                    &mut output,
                    width,
                    height,
                    12,
                    45,
                    "UNTOUCHED RAW10 SOURCE",
                    1,
                    [40, 235, 255],
                );
                draw_text_dynamic(
                    &mut output,
                    width,
                    height,
                    panel_width as isize + 12,
                    45,
                    if review.fit.is_some() {
                        "FITTED EQUIVALENT-AREA CIRCLE / ANIMATED 3D AFFINE"
                    } else if constrained_review {
                        "REJECTED: INSUFFICIENT DISTRIBUTED LIMBUS SUPPORT"
                    } else {
                        "MASK-MOMENT AFFINE (REVIEW ONLY; FIT REJECTED)"
                    },
                    1,
                    [255, 205, 65],
                );
                draw_text_dynamic(
                    &mut output,
                    width,
                    height,
                    (panel_width * 2) as isize + 12,
                    66,
                    &format!("PERSISTENT RAW10 HOT PIXELS: {}", review.hot_pixels.len()),
                    1,
                    if review.hot_pixels.is_empty() {
                        [120, 180, 130]
                    } else {
                        [255, 80, 65]
                    },
                );
                draw_text_dynamic(
                    &mut output,
                    width,
                    height,
                    (panel_width * 2) as isize + 12,
                    45,
                    if review.fit.is_some() {
                        "RAW10 + ACCEPT/IGNORE POINTS (FIT HIDDEN)"
                    } else if constrained_review {
                        "RAW10 / NO ELLIPSE PUBLISHED"
                    } else {
                        "RAW10 + REVIEW-MOMENT ELLIPSE (FIT REJECTED)"
                    },
                    1,
                    [255, 205, 65],
                );
                for y in 0..review.source.height {
                    for x in 0..review.source.width {
                        let pixel = preview[y * review.source.width + x];
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let ox = x * scale + dx;
                                let oy = header + y * scale + dy;
                                if ox < panel_width && oy < height {
                                    output[oy * width + ox] = pixel;
                                }
                                let third_x = panel_width * 2 + x * scale + dx;
                                if third_x < width && oy < height {
                                    output[oy * width + third_x] = pixel;
                                }
                            }
                        }
                    }
                }
                if let (Some(mask), Some(ellipse)) = (review.mask.as_ref(), review_ellipse) {
                    let center = ellipse.center;
                    let radius = (ellipse.major_radius * ellipse.minor_radius).sqrt();
                    let (sin_angle, cos_angle) = ellipse.angle.sin_cos();
                    for y in 0..review.source.height {
                        for x in 0..review.source.width {
                            let d = (x as f64 + 0.5 - center.0, y as f64 + 0.5 - center.1);
                            let canonical = (
                                inverse_view[0][0] * d.0 + inverse_view[0][1] * d.1,
                                inverse_view[1][0] * d.0 + inverse_view[1][1] * d.1,
                            );
                            let local = (
                                canonical.0 * ellipse.major_radius / radius.max(1.0),
                                canonical.1 * ellipse.minor_radius / radius.max(1.0),
                            );
                            let source = (
                                center.0 + cos_angle * local.0 - sin_angle * local.1,
                                center.1 + sin_angle * local.0 + cos_angle * local.1,
                            );
                            let mut pixel = preview_sample(
                                &preview,
                                review.source.width,
                                review.source.height,
                                source.0,
                                source.1,
                            );
                            let mx = (source.0.max(0.0) as usize * review.mask_width
                                / review.source.width)
                                .min(review.mask_width.saturating_sub(1));
                            let my = (source.1.max(0.0) as usize * review.mask_height
                                / review.source.height)
                                .min(review.mask_height.saturating_sub(1));
                            if review.mask_width != 0
                                && review.mask_height != 0
                                && mask
                                    .get(my * review.mask_width + mx)
                                    .is_some_and(|&value| value != 0)
                            {
                                pixel = [
                                    (pixel[0] as f64 * 0.62).round() as u8,
                                    (pixel[1] as f64 * 0.62 + 91.0).min(255.0).round() as u8,
                                    (pixel[2] as f64 * 0.62 + 98.0).min(255.0).round() as u8,
                                ];
                            }
                            for dy in 0..scale {
                                for dx in 0..scale {
                                    let ox = panel_width + x * scale + dx;
                                    let oy = header + y * scale + dy;
                                    if ox < width && oy < height {
                                        output[oy * width + ox] = pixel;
                                    }
                                }
                            }
                        }
                    }
                    for sample in 0..720usize {
                        let theta = std::f64::consts::TAU * sample as f64 / 720.0;
                        let canonical = (radius * theta.cos(), radius * theta.sin());
                        let projected = (
                            view[0][0] * canonical.0 + view[0][1] * canonical.1,
                            view[1][0] * canonical.0 + view[1][1] * canonical.1,
                        );
                        let x = panel_width as isize
                            + ((center.0 + projected.0) * scale as f64).round() as isize;
                        let y = header as isize
                            + ((center.1 + projected.1) * scale as f64).round() as isize;
                        for oy in -1..=1 {
                            for ox in -1..=1 {
                                if x + ox >= 0
                                    && y + oy >= 0
                                    && (x + ox) < width as isize
                                    && (y + oy) < height as isize
                                {
                                    output[(y + oy) as usize * width + (x + ox) as usize] =
                                        [255, 205, 65];
                                }
                            }
                        }
                    }
                }
                if let Some(fit) = review.contour_hypothesis.as_ref() {
                    for (&point, color, radius) in fit
                        .flat_tire_points
                        .iter()
                        .map(|point| (point, [255, 65, 190], 3isize))
                        .chain(
                            fit.retained_points
                                .iter()
                                .map(|point| (point, [70, 255, 115], 2isize)),
                        )
                    {
                        let x =
                            (panel_width * 2) as isize + (point.0 * scale as f64).round() as isize;
                        let y = header as isize + (point.1 * scale as f64).round() as isize;
                        for oy in -radius..=radius {
                            for ox in -radius..=radius {
                                if ox.abs() + oy.abs() > radius
                                    || x + ox < 0
                                    || y + oy < 0
                                    || (x + ox) >= width as isize
                                    || (y + oy) >= height as isize
                                {
                                    continue;
                                }
                                output[(y + oy) as usize * width + (x + ox) as usize] = color;
                            }
                        }
                    }
                } else {
                    for &point in review.raw_contour_points.iter() {
                        let x =
                            (panel_width * 2) as isize + (point.0 * scale as f64).round() as isize;
                        let y = header as isize + (point.1 * scale as f64).round() as isize;
                        for oy in -2isize..=2 {
                            for ox in -2isize..=2 {
                                if ox.abs() + oy.abs() > 2
                                    || x + ox < 0
                                    || y + oy < 0
                                    || (x + ox) >= width as isize
                                    || (y + oy) >= height as isize
                                {
                                    continue;
                                }
                                output[(y + oy) as usize * width + (x + ox) as usize] =
                                    [255, 65, 190];
                            }
                        }
                    }
                }
                // Flash a high-contrast cross around every confirmed defect in
                // both untouched RAW panels. The center sample remains the
                // original RAW value; this is a review overlay, not repair.
                let hot_color = if local_frame < display_frames_per_source / 2 {
                    [255, 45, 30]
                } else {
                    [255, 255, 255]
                };
                for &(hot_x, hot_y) in review.hot_pixels.iter() {
                    for panel_x in [0usize, panel_width * 2] {
                        let cx = panel_x as isize + (hot_x * scale + scale / 2) as isize;
                        let cy = header as isize + (hot_y * scale + scale / 2) as isize;
                        for offset in -7isize..=7 {
                            if offset.abs() <= 2 {
                                continue;
                            }
                            for (x, y) in [(cx + offset, cy), (cx, cy + offset)] {
                                if x >= 0 && y >= 0 && x < width as isize && y < height as isize {
                                    output[y as usize * width + x as usize] = hot_color;
                                }
                            }
                        }
                    }
                }
                let bytes = unsafe {
                    std::slice::from_raw_parts(output.as_ptr() as *const u8, output.len() * 3)
                };
                stdin
                    .write_all(bytes)
                    .map_err(|error| format!("write video feature review frame: {error}"))?;
            }
        }
        drop(stdin);
        let status = child
            .wait()
            .map_err(|error| format!("wait for video feature review ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("video feature review ffmpeg exited with {status}"));
        }
        Ok(())
    }

    /// A visualization-only affine derived from the occupied mask's area and
    /// second moments. It exists so a rejected production contour remains
    /// visible in review; it is never returned to tracking or ellipse fitting.
    fn mask_moment_review_ellipse(
        mask: &[u8],
        mask_width: usize,
        mask_height: usize,
        native_width: usize,
        native_height: usize,
    ) -> Option<Ellipse> {
        if mask_width == 0 || mask_height == 0 || mask.len() != mask_width * mask_height {
            return None;
        }
        let mut count = 0.0;
        let mut mean_x = 0.0;
        let mut mean_y = 0.0;
        for (index, &value) in mask.iter().enumerate() {
            if value == 0 {
                continue;
            }
            count += 1.0;
            mean_x += (index % mask_width) as f64 + 0.5;
            mean_y += (index / mask_width) as f64 + 0.5;
        }
        if count < 16.0 {
            return None;
        }
        mean_x /= count;
        mean_y /= count;
        let mut xx = 0.0;
        let mut xy = 0.0;
        let mut yy = 0.0;
        for (index, &value) in mask.iter().enumerate() {
            if value == 0 {
                continue;
            }
            let dx = (index % mask_width) as f64 + 0.5 - mean_x;
            let dy = (index / mask_width) as f64 + 0.5 - mean_y;
            xx += dx * dx;
            xy += dx * dy;
            yy += dy * dy;
        }
        xx /= count;
        xy /= count;
        yy /= count;
        let trace = xx + yy;
        let split = ((0.5 * (xx - yy)).powi(2) + xy * xy).sqrt();
        let major_variance = (0.5 * trace + split).max(1.0e-6);
        let minor_variance = (0.5 * trace - split).max(1.0e-6);
        let variance_ratio = (major_variance / minor_variance).sqrt().clamp(1.0, 8.0);
        let native_area =
            count * native_width as f64 * native_height as f64 / (mask_width * mask_height) as f64;
        let equivalent_radius = (native_area / std::f64::consts::PI).sqrt();
        Some(Ellipse {
            center: (
                mean_x * native_width as f64 / mask_width as f64,
                mean_y * native_height as f64 / mask_height as f64,
            ),
            major_radius: equivalent_radius * variance_ratio.sqrt(),
            minor_radius: equivalent_radius / variance_ratio.sqrt(),
            angle: 0.5 * (2.0 * xy).atan2(xx - yy),
        })
    }

    fn load_rejection_review_input(argument: &OsStr) -> Result<LoadedRaw, String> {
        load_rejection_review_history(argument)?
            .pop()
            .ok_or_else(|| "rejected-review input did not yield a frame".to_string())
    }

    fn load_rejection_review_history(argument: &OsStr) -> Result<Vec<LoadedRaw>, String> {
        let path = Path::new(argument);
        if path.extension() == Some(OsStr::new("raw10")) {
            return load_raw(path).map(|frame| vec![frame]);
        }
        let text = argument
            .to_str()
            .ok_or("rejected-review input is not UTF-8")?;
        let (capture_and_sequence, eye_label) = text.rsplit_once('@').ok_or_else(|| {
            format!("rejected-review input {text:?} must be RAW10 or CAPTURE@SEQUENCE@EYE_LABEL")
        })?;
        let (capture, sequence) = capture_and_sequence.rsplit_once('@').ok_or_else(|| {
            format!("rejected-review input {text:?} must be RAW10 or CAPTURE@SEQUENCE@EYE_LABEL")
        })?;
        let sequence = sequence
            .parse::<u64>()
            .map_err(|error| format!("invalid rejected-review sequence {sequence:?}: {error}"))?;
        load_capture_history(Path::new(capture), sequence, eye_label)
    }

    fn run_rejected_sam_mask(
        model: &Path,
        outer_prompt_bundle: &Path,
        prompt_bundle: &Path,
        argument: &OsStr,
    ) -> Result<RejectedSamMaskReview, String> {
        let mut loaded = load_rejection_review_history(argument)?;
        if loaded.len() > sam31_outer::HISTORY_FRAMES {
            loaded = loaded.split_off(loaded.len() - sam31_outer::HISTORY_FRAMES);
        }
        while loaded.len() < sam31_outer::HISTORY_FRAMES {
            let first = loaded
                .first()
                .cloned()
                .ok_or("rejected SAM review has an empty RAW10 history")?;
            loaded.insert(0, first);
        }
        let frames = loaded
            .iter()
            .map(|loaded| Arc::clone(&loaded.frame))
            .collect::<Vec<_>>();
        let suite = sam31_outer::run_offline_semantic_suite(
            model,
            outer_prompt_bundle,
            prompt_bundle,
            sam31_text::ARC_TRIAL_PROMPTS.len(),
            &frames,
            &[OUTER],
        )?;
        let pass = suite
            .passes
            .iter()
            .find(|pass| pass.prompt_index == OUTER)
            .ok_or("SAM31 omitted its initial outer-disk answer")?;
        let selected_query = pass
            .selected_query
            .or_else(|| {
                pass.masks
                    .iter()
                    .filter(|mask| mask.score.is_finite())
                    .max_by(|left, right| left.score.total_cmp(&right.score))
                    .map(|mask| mask.query)
            })
            .ok_or("SAM31 initial outer-disk answer contained no selectable mask")?;
        let mask = pass
            .masks
            .iter()
            .find(|mask| mask.query == selected_query)
            .cloned()
            .ok_or_else(|| format!("SAM31 selected missing outer-mask query {selected_query}"))?;
        let blank_color = vec![[0u8; 3]; suite.source_raw.len()];
        let preview = clean_raw_display_preview(
            &suite.source_raw,
            &blank_color,
            suite.source_width,
            suite.source_height,
        );
        let sequence = frames
            .last()
            .map(|frame| frame.sequence)
            .ok_or("SAM31 rejected review lost its target frame")?;
        let area_fraction = mask.pixels.iter().filter(|&&pixel| pixel != 0).count() as f64
            / mask.pixels.len().max(1) as f64;
        eprintln!(
            "outer-mask sequence={sequence} query={} score={:.4} area={:.3} geometry_accepted={} elapsed_ms={}",
            mask.query,
            mask.score,
            area_fraction,
            suite.outer_fit.is_some(),
            suite.elapsed_ms,
        );
        Ok(RejectedSamMaskReview {
            sequence,
            preview,
            width: suite.source_width,
            height: suite.source_height,
            mask_width: pass.width,
            mask_height: pass.height,
            mask,
            geometry_accepted: suite.outer_fit.is_some(),
            elapsed_ms: suite.elapsed_ms,
        })
    }

    fn noncomment_lines(path: &Path, kind: &str) -> Result<Vec<String>, String> {
        let source = fs::read_to_string(path)
            .map_err(|error| format!("read {kind} {}: {error}", path.display()))?;
        let lines = source
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(str::to_string)
            .collect::<Vec<_>>();
        if lines.is_empty() {
            return Err(format!("{kind} {} has no usable lines", path.display()));
        }
        Ok(lines)
    }

    fn native_sensor_mask(
        row: &PromptLabRow,
        candidate: &PromptLabCandidate,
    ) -> HashSet<(u32, u32)> {
        let mut pixels = HashSet::new();
        if candidate.mask_width == 0 || candidate.mask_height == 0 {
            return pixels;
        }
        for y in 0..row.height {
            let mask_y = ((y * candidate.mask_height) / row.height)
                .min(candidate.mask_height.saturating_sub(1));
            for x in 0..row.width {
                let mask_x = ((x * candidate.mask_width) / row.width)
                    .min(candidate.mask_width.saturating_sub(1));
                if candidate.mask.pixels[mask_y * candidate.mask_width + mask_x] != 0 {
                    pixels.insert((
                        row.sensor_origin.0.saturating_add(x as u32),
                        row.sensor_origin.1.saturating_add(y as u32),
                    ));
                }
            }
        }
        pixels
    }

    fn mask_centroid(mask: &HashSet<(u32, u32)>) -> Option<(f64, f64)> {
        (!mask.is_empty()).then(|| {
            let (sum_x, sum_y) = mask.iter().fold((0u64, 0u64), |sum, &(x, y)| {
                (
                    sum.0.saturating_add(x as u64),
                    sum.1.saturating_add(y as u64),
                )
            });
            (
                sum_x as f64 / mask.len() as f64,
                sum_y as f64 / mask.len() as f64,
            )
        })
    }

    fn mask_pair_metrics(
        first: &PromptLabRow,
        first_candidate: &PromptLabCandidate,
        second: &PromptLabRow,
        second_candidate: &PromptLabCandidate,
    ) -> Value {
        let first_mask = native_sensor_mask(first, first_candidate);
        let second_mask = native_sensor_mask(second, second_candidate);
        let intersection = first_mask.intersection(&second_mask).count();
        let union = first_mask.len() + second_mask.len() - intersection;
        let iou = (union != 0).then(|| intersection as f64 / union as f64);
        let centroid_shift_sensor_px = mask_centroid(&first_mask)
            .zip(mask_centroid(&second_mask))
            .map(|(first, second)| (second.0 - first.0).hypot(second.1 - first.1));
        let abs_log_area_ratio = (!first_mask.is_empty() && !second_mask.is_empty()).then(|| {
            (second_mask.len() as f64 / first_mask.len() as f64)
                .ln()
                .abs()
        });
        json!({
            "from_query": first_candidate.query,
            "to_query": second_candidate.query,
            "from_native_area": first_mask.len(),
            "to_native_area": second_mask.len(),
            "sensor_aligned_iou": iou,
            "centroid_shift_sensor_px": centroid_shift_sensor_px,
            "abs_log_area_ratio": abs_log_area_ratio,
        })
    }

    fn temporal_candidate_path(rows: &[PromptLabRow], prompt_index: usize) -> Option<Value> {
        let candidate_rows = rows
            .iter()
            .map(|row| row.steps.get(prompt_index).map(|step| &step.candidates))
            .collect::<Option<Vec<_>>>()?;
        if candidate_rows.is_empty() || candidate_rows.iter().any(|row| row.is_empty()) {
            return None;
        }
        let masks = rows
            .iter()
            .zip(candidate_rows.iter())
            .map(|(row, candidates)| {
                candidates
                    .iter()
                    .map(|candidate| native_sensor_mask(row, candidate))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut scores = candidate_rows[0]
            .iter()
            .map(|candidate| 0.10 * candidate.crust_score)
            .collect::<Vec<_>>();
        let mut parents = Vec::<Vec<usize>>::with_capacity(rows.len().saturating_sub(1));
        for row_index in 1..rows.len() {
            let mut next_scores = vec![f64::NEG_INFINITY; candidate_rows[row_index].len()];
            let mut next_parents = vec![0usize; candidate_rows[row_index].len()];
            for (next_index, next_candidate) in candidate_rows[row_index].iter().enumerate() {
                for (prior_index, prior_score) in scores.iter().copied().enumerate() {
                    let prior_mask = &masks[row_index - 1][prior_index];
                    let next_mask = &masks[row_index][next_index];
                    let intersection = prior_mask.intersection(next_mask).count();
                    let union = prior_mask.len() + next_mask.len() - intersection;
                    let iou = if union == 0 {
                        0.0
                    } else {
                        intersection as f64 / union as f64
                    };
                    let area_change = if prior_mask.is_empty() || next_mask.is_empty() {
                        8.0
                    } else {
                        (next_mask.len() as f64 / prior_mask.len() as f64)
                            .ln()
                            .abs()
                    };
                    let centroid_shift = mask_centroid(prior_mask)
                        .zip(mask_centroid(next_mask))
                        .map(|(first, second)| (second.0 - first.0).hypot(second.1 - first.1))
                        .unwrap_or(384.0);
                    let transition = 4.0 * iou - 0.80 * area_change - 0.012 * centroid_shift;
                    let objective = prior_score + transition + 0.10 * next_candidate.crust_score;
                    if objective > next_scores[next_index] {
                        next_scores[next_index] = objective;
                        next_parents[next_index] = prior_index;
                    }
                }
            }
            scores = next_scores;
            parents.push(next_parents);
        }
        let mut selected = vec![0usize; rows.len()];
        selected[rows.len() - 1] = scores
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))?
            .0;
        for row_index in (1..rows.len()).rev() {
            selected[row_index - 1] = parents[row_index - 1][selected[row_index]];
        }
        let selections = rows
            .iter()
            .zip(candidate_rows.iter())
            .zip(selected.iter().copied())
            .map(|((row, candidates), candidate_index)| {
                let candidate = &candidates[candidate_index];
                json!({
                    "sequence": row.sequence,
                    "rank_within_top_four": candidate_index + 1,
                    "query": candidate.query,
                    "crust_score": candidate.crust_score,
                    "area_fraction": candidate.area_fraction,
                })
            })
            .collect::<Vec<_>>();
        let pair_metrics = rows
            .windows(2)
            .enumerate()
            .map(|(index, pair)| {
                mask_pair_metrics(
                    &pair[0],
                    &candidate_rows[index][selected[index]],
                    &pair[1],
                    &candidate_rows[index + 1][selected[index + 1]],
                )
            })
            .collect::<Vec<_>>();
        Some(json!({
            "objective": scores[selected[rows.len() - 1]],
            "transition_weights": {
                "sensor_aligned_iou": 4.0,
                "abs_log_area_ratio": -0.80,
                "centroid_shift_sensor_px": -0.012,
                "crust_score_emission": 0.10,
            },
            "selections": selections,
            "pair_metrics": pair_metrics,
        }))
    }

    fn temporal_prompt_comparison(prompts: &[String], rows: &[PromptLabRow]) -> Value {
        let mut prompt_reports = Vec::with_capacity(prompts.len());
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            let mut pairs = Vec::with_capacity(rows.len().saturating_sub(1));
            for window in rows.windows(2) {
                let first = &window[0];
                let second = &window[1];
                let first_candidates = first
                    .steps
                    .get(prompt_index)
                    .map(|step| step.candidates.as_slice())
                    .unwrap_or_default();
                let second_candidates = second
                    .steps
                    .get(prompt_index)
                    .map(|step| step.candidates.as_slice())
                    .unwrap_or_default();
                let selected = first_candidates.first().zip(second_candidates.first()).map(
                    |(first_candidate, second_candidate)| {
                        mask_pair_metrics(first, first_candidate, second, second_candidate)
                    },
                );
                let mut best_available = None::<Value>;
                let mut best_iou = -1.0f64;
                for first_candidate in first_candidates {
                    for second_candidate in second_candidates {
                        let metrics =
                            mask_pair_metrics(first, first_candidate, second, second_candidate);
                        let iou = metrics["sensor_aligned_iou"].as_f64().unwrap_or(-1.0);
                        if iou > best_iou {
                            best_iou = iou;
                            best_available = Some(metrics);
                        }
                    }
                }
                pairs.push(json!({
                    "from_sequence": first.sequence,
                    "to_sequence": second.sequence,
                    "sequence_gap": second.sequence.saturating_sub(first.sequence),
                    "selected_top_candidate": selected,
                    "best_pair_within_top_four": best_available,
                }));
            }
            prompt_reports.push(json!({
                "prompt_step": prompt_index + 1,
                "prompt": prompt,
                "frame_pairs": pairs,
                "temporally_coherent_top_four_path": temporal_candidate_path(rows, prompt_index),
            }));
        }
        json!({
            "coordinate_system": "native full-sensor pixels after accounting for each RAW ROI origin",
            "selection_contract": "top-to-top measures the configured ranker; best top-four pair estimates proposal continuity independently of rank-one selection",
            "prompts": prompt_reports,
        })
    }

    fn prompt_label_benchmark(prompts: &[String], rows: &[PromptLabRow]) -> Value {
        let labeled_rows = rows
            .iter()
            .filter(|row| row.visible_label_points != 0)
            .count();
        let prompts = prompts
            .iter()
            .enumerate()
            .map(|(prompt_index, prompt)| {
                let mut rank_one = Vec::<f64>::new();
                let mut top_four_oracle = Vec::<f64>::new();
                let mut eligible_oracle = Vec::<f64>::new();
                let mut per_frame = Vec::new();
                for row in rows.iter().filter(|row| row.visible_label_points != 0) {
                    let Some(step) = row.steps.get(prompt_index) else { continue };
                    let rank_one_candidate = step.candidates.first();
                    let rank_one_error = rank_one_candidate.and_then(|candidate| candidate.label_fit_mean_px);
                    if let Some(error) = rank_one_error { rank_one.push(error); }
                    let best_top_four = step
                        .candidates
                        .iter()
                        .filter_map(|candidate| candidate.label_fit_mean_px.map(|error| (error, candidate.query)))
                        .min_by(|left, right| left.0.total_cmp(&right.0));
                    if let Some((error, _)) = best_top_four { top_four_oracle.push(error); }
                    let best_eligible = step
                        .all_candidates
                        .iter()
                        .filter(|candidate| candidate.eligible)
                        .filter_map(|candidate| candidate.label_fit_mean_px.map(|error| (error, candidate.query)))
                        .min_by(|left, right| left.0.total_cmp(&right.0));
                    if let Some((error, _)) = best_eligible { eligible_oracle.push(error); }
                    per_frame.push(json!({
                        "sequence": row.sequence,
                        "rank_one_query": rank_one_candidate.map(|candidate| candidate.query),
                        "rank_one_fit_mean_error_px": rank_one_error,
                        "best_top_four_query": best_top_four.map(|candidate| candidate.1),
                        "best_top_four_fit_mean_error_px": best_top_four.map(|candidate| candidate.0),
                        "best_eligible_query": best_eligible.map(|candidate| candidate.1),
                        "best_eligible_fit_mean_error_px": best_eligible.map(|candidate| candidate.0),
                    }));
                }
                let mean = |values: &[f64]| (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64);
                json!({
                    "prompt_step": prompt_index + 1,
                    "prompt": prompt,
                    "labeled_rows": labeled_rows,
                    "rank_one_valid_fits": rank_one.len(),
                    "rank_one_mean_fit_error_px": mean(&rank_one),
                    "top_four_oracle_valid_fits": top_four_oracle.len(),
                    "top_four_oracle_mean_fit_error_px": mean(&top_four_oracle),
                    "all_eligible_oracle_valid_fits": eligible_oracle.len(),
                    "all_eligible_oracle_mean_fit_error_px": mean(&eligible_oracle),
                    "per_frame": per_frame,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "contract": "Human labels are read only after model inference and candidate ranking. Rank-one measures the deployable current ranker; top-four and all-eligible oracle figures measure prompt headroom and never select a production candidate.",
            "labeled_rows": labeled_rows,
            "prompts": prompts,
        })
    }

    fn run_prompt_lab(
        output: &Path,
        prompt_file: &Path,
        manifest: &Path,
        prompt_bundle: &Path,
    ) -> Result<(), String> {
        let prompts = noncomment_lines(prompt_file, "prompt file")?;
        let preprocess = std::env::var("BUTTERCUP_SAM31_PREPROCESS")
            .ok()
            .map(|value| {
                sam31_outer::PreprocessRegime::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown SAM31 preprocessing regime {value:?}; expected one of {}",
                        sam31_outer::PreprocessRegime::ALL
                            .iter()
                            .map(|regime| regime.label())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
            })
            .transpose()?
            .unwrap_or_default();
        if prompts.len() > 32 {
            return Err(format!(
                "prompt lab supports at most 32 runtime prompt entries; {} contains {}",
                prompt_file.display(),
                prompts.len()
            ));
        }
        let inputs = noncomment_lines(manifest, "RAW10 manifest")?;
        if inputs.len() > 32 {
            return Err(format!(
                "prompt lab is bounded to 32 RAW10 targets per run; {} contains {}",
                manifest.display(),
                inputs.len()
            ));
        }
        let logical_panels = inputs
            .len()
            .saturating_mul(1 + prompts.len() * PROMPT_LAB_DISPLAY_CANDIDATES);
        if logical_panels > 1024 {
            return Err(format!(
                "runtime contact-sheet layout is too large: {} rows x {} columns = {} logical panels (maximum 1024)",
                inputs.len(),
                1 + prompts.len() * PROMPT_LAB_DISPLAY_CANDIDATES,
                logical_panels
            ));
        }
        let model = environment_path(
            "BUTTERCUP_SAM31_MODEL",
            "data/models/sam31_semantic_dynamic_u8.pt",
        );
        let outer_prompt_bundle = environment_path(
            "BUTTERCUP_SAM31_PROMPT_BUNDLE",
            "data/models/sam31_semantic_prompts_cuda_bf16.pt",
        );
        for required in [&model, &outer_prompt_bundle, prompt_bundle] {
            if !required.is_file() {
                return Err(format!(
                    "SAM31 prompt-lab input is unavailable: {}",
                    required.display()
                ));
            }
        }
        fs::create_dir_all(output)
            .map_err(|error| format!("create {}: {error}", output.display()))?;
        let prompt_indices = (1..=prompts.len()).collect::<Vec<_>>();
        let mut rows = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            eprintln!(
                "prompt-lab target {}/{}: {}",
                index + 1,
                inputs.len(),
                input
            );
            rows.push(run_prompt_lab_row(
                &model,
                &outer_prompt_bundle,
                prompt_bundle,
                prompts.len() + 1,
                &prompt_indices,
                OsStr::new(input),
                preprocess,
            )?);
        }
        let contact_sheet = output.join("top4-source-rows-contact-sheet.png");
        let affine_review = output.join("affine-review.mkv");
        render_prompt_lab_contact_sheet(&contact_sheet, &prompts, &rows)?;
        render_prompt_lab_affine_review(&affine_review, &prompts, &rows)?;
        fs::write(
            output.join("prompt-chain.txt"),
            fs::read(prompt_file).map_err(|error| {
                format!(
                    "read prompt file snapshot {}: {error}",
                    prompt_file.display()
                )
            })?,
        )
        .map_err(|error| format!("write prompt-chain snapshot: {error}"))?;
        fs::write(
            output.join("input-manifest.txt"),
            fs::read(manifest).map_err(|error| {
                format!("read manifest snapshot {}: {error}", manifest.display())
            })?,
        )
        .map_err(|error| format!("write manifest snapshot: {error}"))?;
        let report = json!({
            "schema": "buttercup-sam31-prompt-lab-v1",
            "raw_contract": "native lossless RAW10 input; masks are selected without human labels; no Python, JPEG, or desktop capture",
            "preprocessing_regime": preprocess.label(),
            "video_feature_shapes": rows.first().and_then(|row| row.video_feature_shapes.as_ref()).map(|shapes| json!({
                "pyramid": shapes.pyramid,
                "decoder_queries": shapes.decoder_queries,
            })),
            "sam31_execution_contract": {
                "backend": "detector-only TorchScript filmstrip",
                "native_video_memory_used": false,
                "native_object_pointers_used": false,
                "native_detector_tracker_association_used": false,
                "native_occlusion_state_used": false,
                "native_reconditioning_used": false,
                "warning": "This preprocessing experiment must not be represented as SAM3.1 native video tracking. The available TorchScript graph exports only image-detector forward(image, language_features, language_mask, img_ids, text_ids)."
            },
            "temporal_comparison": temporal_prompt_comparison(&prompts, &rows),
            "human_label_benchmark": prompt_label_benchmark(&prompts, &rows),
            "prompt_semantics": "Each line is an isolated SAM semantic question on exactly three unique sequential RAW10 frames: the target and its two logically preceding same-eye frames. The traced model's fixed five-tile input is satisfied by left-padding the earliest of those three; no additional historical frame is introduced. Prompt groups preserve order for visual chain review. In reflection-mask-first mode, steps one and two are mask-only evidence and only step three may fit and display an ellipse. In ordinary prompt-lab mode, the first prompt of each triplet retains the 5-to-50-percent disk rule. Each source image occupies one contact-sheet row.",
            "candidate_area_fraction_minimum": 0.05,
            "candidate_area_fraction_maximum": 0.50,
            "lower_stage_evidence_area_fraction_minimum": 0.0,
            "lower_stage_evidence_contract": "Evidence-only steps visibly retain nonempty masks up to 50% area but cannot independently fit or publish an ellipse. Reflection-mask-first reserves all explicit fit points and ellipse geometry for the final stage.",
            "prompts": prompts,
            "rows": rows.iter().map(|row| json!({
                "source": row.source,
                "sequence": row.sequence,
                "sensor_origin": row.sensor_origin,
                "logical_history_sequences": row.logical_history_sequences,
                "elapsed_ms": row.elapsed_ms,
                "label_path": row.label_path,
                "visible_label_points": row.visible_label_points,
                "steps": row.steps.iter().map(|step| json!({
                    "prompt_step": step.prompt_index,
                    "selected_query": step.selected_query,
                    "returned_candidates": step.returned_candidates,
                    "displayed_candidates": step.candidates.iter().map(|candidate| json!({
                        "query": candidate.query,
                        "model_score": candidate.score,
                        "area_fraction": candidate.area_fraction,
                        "segment_roundness": candidate.segment_roundness,
                        "crust_score": candidate.crust_score,
                        "evidence_only": candidate.evidence_only,
                        "display_eligible": candidate.display_eligible,
                        "fit": candidate.fitted.map(ellipse_json),
                        "human_label_boundary_mean_error_px": candidate.label_boundary_mean_px,
                        "human_label_boundary_rms_error_px": candidate.label_boundary_rms_px,
                        "human_label_boundary_max_error_px": candidate.label_boundary_max_px,
                        "human_label_fit_mean_error_px": candidate.label_fit_mean_px,
                        "human_label_fit_rms_error_px": candidate.label_fit_rms_px,
                        "human_label_fit_max_error_px": candidate.label_fit_max_px,
                    })).collect::<Vec<_>>(),
                    "all_candidates_ranked_high_to_low": step.all_candidates.iter().map(|candidate| json!({
                        "query": candidate.query,
                        "model_score": candidate.score,
                        "area_fraction": candidate.area_fraction,
                        "segment_roundness": candidate.segment_roundness,
                        "crust_score": candidate.crust_score,
                        "eligible_under_area_rule": candidate.eligible,
                        "evidence_only": candidate.evidence_only,
                        "display_eligible": candidate.display_eligible,
                        "fit": candidate.fitted.map(ellipse_json),
                        "human_label_boundary_mean_error_px": candidate.label_boundary_mean_px,
                        "human_label_boundary_rms_error_px": candidate.label_boundary_rms_px,
                        "human_label_boundary_max_error_px": candidate.label_boundary_max_px,
                        "human_label_fit_mean_error_px": candidate.label_fit_mean_px,
                        "human_label_fit_rms_error_px": candidate.label_fit_rms_px,
                        "human_label_fit_max_error_px": candidate.label_fit_max_px,
                    })).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "contact_sheet": contact_sheet,
            "affine_review": affine_review,
        });
        let report_path = output.join("report.json");
        fs::write(
            &report_path,
            serde_json::to_vec_pretty(&report)
                .map_err(|error| format!("encode prompt-lab report: {error}"))?,
        )
        .map_err(|error| format!("write {}: {error}", report_path.display()))?;
        println!("contact_sheet={}", contact_sheet.display());
        println!("affine_review={}", affine_review.display());
        println!("report={}", report_path.display());
        Ok(())
    }

    fn run_prompt_lab_row(
        model: &Path,
        outer_prompt_bundle: &Path,
        prompt_bundle: &Path,
        prompt_count: usize,
        prompt_indices: &[usize],
        input: &OsStr,
        preprocess: sam31_outer::PreprocessRegime,
    ) -> Result<PromptLabRow, String> {
        let mut loaded = load_rejection_review_history(input)?;
        const LOGICAL_PROMPT_FRAMES: usize = 3;
        if loaded.len() > LOGICAL_PROMPT_FRAMES {
            loaded = loaded.split_off(loaded.len() - LOGICAL_PROMPT_FRAMES);
        }
        if loaded.len() != LOGICAL_PROMPT_FRAMES {
            return Err(format!(
                "prompt-lab target {input:?} has only {} same-eye frames; exactly two logical predecessors are required",
                loaded.len()
            ));
        }
        let logical_history_sequences = loaded
            .iter()
            .map(|loaded| loaded.frame.sequence)
            .collect::<Vec<_>>();
        while loaded.len() < sam31_outer::HISTORY_FRAMES {
            let first = loaded
                .first()
                .cloned()
                .ok_or("prompt-lab RAW10 history is empty")?;
            loaded.insert(0, first);
        }
        let frames = loaded
            .iter()
            .map(|loaded| Arc::clone(&loaded.frame))
            .collect::<Vec<_>>();
        let target = loaded.last().ok_or("prompt-lab target is missing")?;
        let suite = sam31_outer::run_offline_semantic_suite_with_regime(
            model,
            outer_prompt_bundle,
            prompt_bundle,
            prompt_count,
            &frames,
            prompt_indices,
            preprocess,
        )?;
        let blank_color = vec![[0u8; 3]; suite.source_raw.len()];
        let preview = clean_raw_display_preview(
            &suite.source_raw,
            &blank_color,
            suite.source_width,
            suite.source_height,
        );
        let label_path = locate_labels(&target.source);
        let labels = label_path
            .as_ref()
            .map(|path| read_visible_labels(path))
            .transpose()?
            .unwrap_or_default();
        let reflection_mask_first = std::env::var_os("BUTTERCUP_SAM31_REFLECTION_MASK_FIRST")
            .is_some()
            || std::env::var_os("BUTTERCUP_SAM31_CENTER_VOID_MASK_FIRST").is_some()
            || std::env::var_os("BUTTERCUP_SAM31_PUPIL_ISOLATE_FIRST").is_some()
            || std::env::var_os("BUTTERCUP_SAM31_PUPIL_INNER_OCCLUDE_FIRST").is_some()
            || std::env::var_os("BUTTERCUP_SAM31_PUPIL_RESEGMENT_CONTROL").is_some();
        let mut steps = Vec::with_capacity(prompt_indices.len());
        let step_two_sweep = std::env::var_os("BUTTERCUP_SAM31_PUPIL_STEP2_SWEEP").is_some();
        let limbus_regime_sweep = std::env::var_os("BUTTERCUP_SAM31_LIMBUS_REGIME_SWEEP").is_some();
        let pupil_three_step = std::env::var_os("BUTTERCUP_SAM31_PUPIL_THREE_STEP").is_some();
        for (step_position, &prompt_index) in prompt_indices.iter().enumerate() {
            // Prompt-lab chains are coarse region -> local material -> final
            // boundary. In a reflection-first chain the first two stages are
            // masks only; only the final refined boundary may yield and show
            // a fitted disk ellipse.
            let evidence_only = if pupil_three_step {
                step_position != 1
            } else if limbus_regime_sweep {
                false
            } else if step_two_sweep {
                true
            } else if reflection_mask_first {
                step_position != 2
            } else {
                step_position % 3 != 0
            };
            let pass = suite
                .passes
                .iter()
                .find(|pass| pass.prompt_index == prompt_index)
                .ok_or_else(|| format!("SAM31 omitted prompt-lab step {prompt_index}"))?;
            // Detector query zero is not a privileged semantic answer: every
            // query is merely an object candidate for the same text. The raw
            // maximum model score is frequently attached to an empty mask,
            // which made a useful prompt appear to have returned nothing.
            // Rank useful object proposals by visible "pizza crust", before
            // any affine correction. This is intentionally a LOCAL,
            // affine-tolerant boundary-segment measurement rather than a
            // circle fit or whole-mask circularity: an oblique or occluded
            // iris can have an excellent surviving elliptical arc while its
            // complete mask is quite non-round.
            let mut all_candidates = pass
                .masks
                .iter()
                .map(|mask| {
                    let occupied = mask.pixels.iter().filter(|&&pixel| pixel != 0).count();
                    let area_fraction = occupied as f64 / mask.pixels.len().max(1) as f64;
                    let segment_roundness =
                        best_boundary_segment_arc_score(mask, pass.width, pass.height, occupied);
                    let crust_score = area_fraction.sqrt() * segment_roundness;
                    let disk_area_eligible =
                        occupied * 20 >= mask.pixels.len() && occupied * 2 <= mask.pixels.len();
                    let fit_review = (!evidence_only && disk_area_eligible)
                        .then(|| {
                            sam31_outer::diagnostic_fit_single_frame_mask(
                                &mask.pixels,
                                pass.width,
                                pass.height,
                            )
                        })
                        .flatten();
                    let fitted = fit_review.as_ref().map(|review| review.ellipse);
                    let fit_points = fit_review
                        .as_ref()
                        .map(|review| Arc::clone(&review.retained_points))
                        .unwrap_or_default();
                    let boundary_metrics = (!labels.is_empty())
                        .then(|| {
                            mask_label_boundary_metrics(
                                &mask.pixels,
                                pass.width,
                                pass.height,
                                suite.source_width,
                                suite.source_height,
                                &labels,
                            )
                        })
                        .flatten();
                    let fit_metrics = fitted
                        .filter(|_| !labels.is_empty())
                        .map(|ellipse| ellipse_label_metrics(ellipse, &labels));
                    PromptLabCandidate {
                        query: mask.query,
                        score: mask.score,
                        area_fraction,
                        segment_roundness,
                        crust_score,
                        eligible: !evidence_only && disk_area_eligible,
                        display_eligible: occupied != 0
                            && occupied * 2 <= mask.pixels.len()
                            && (evidence_only || disk_area_eligible),
                        evidence_only,
                        mask_width: pass.width,
                        mask_height: pass.height,
                        mask: mask.clone(),
                        fitted,
                        fit_points,
                        label_boundary_mean_px: boundary_metrics.map(|metrics| metrics.0),
                        label_boundary_rms_px: boundary_metrics.map(|metrics| metrics.1),
                        label_boundary_max_px: boundary_metrics.map(|metrics| metrics.2),
                        label_fit_mean_px: fit_metrics.map(|metrics| metrics.0),
                        label_fit_rms_px: fit_metrics.map(|metrics| metrics.1),
                        label_fit_max_px: fit_metrics.map(|metrics| metrics.2),
                    }
                })
                .collect::<Vec<_>>();
            all_candidates.sort_by(|left, right| {
                right
                    .crust_score
                    .total_cmp(&left.crust_score)
                    .then_with(|| right.segment_roundness.total_cmp(&left.segment_roundness))
                    .then_with(|| right.area_fraction.total_cmp(&left.area_fraction))
                    .then_with(|| right.score.total_cmp(&left.score))
                    .then_with(|| left.query.cmp(&right.query))
            });
            let mut candidates = all_candidates
                .iter()
                .filter(|candidate| candidate.display_eligible)
                .take(PROMPT_LAB_DISPLAY_CANDIDATES)
                .cloned()
                .collect::<Vec<_>>();
            if evidence_only {
                // Always expose the strongest genuinely sub-5% segment even
                // when larger evidence masks occupy the normal top four.
                if let Some(thin) = all_candidates
                    .iter()
                    .filter(|candidate| {
                        candidate.display_eligible && candidate.area_fraction < 0.05
                    })
                    .max_by(|left, right| {
                        left.segment_roundness
                            .total_cmp(&right.segment_roundness)
                            .then_with(|| left.score.total_cmp(&right.score))
                    })
                    .cloned()
                {
                    if !candidates
                        .iter()
                        .any(|candidate| candidate.query == thin.query)
                    {
                        if candidates.len() == PROMPT_LAB_DISPLAY_CANDIDATES {
                            candidates.pop();
                        }
                        candidates.push(thin);
                    }
                }
            }
            if reflection_mask_first {
                if let Some(selected) = pass.selected_query.and_then(|query| {
                    all_candidates
                        .iter()
                        .find(|candidate| candidate.query == query)
                        .cloned()
                }) {
                    candidates.retain(|candidate| candidate.query != selected.query);
                    candidates.insert(0, selected);
                    candidates.truncate(PROMPT_LAB_DISPLAY_CANDIDATES);
                }
            }
            steps.push(PromptLabStep {
                prompt_index,
                selected_query: pass.selected_query,
                returned_candidates: pass.masks.len(),
                candidates,
                all_candidates,
            });
        }
        let selected_first = steps.first().and_then(|step| {
            step.selected_query.and_then(|query| {
                step.candidates
                    .iter()
                    .find(|candidate| candidate.query == query)
            })
        });
        let conditioning_preview = if std::env::var_os("BUTTERCUP_SAM31_PUPIL_ISOLATE_FIRST")
            .is_some()
        {
            selected_first.map(|candidate| {
                isolated_pupil_preview(&preview, suite.source_width, suite.source_height, candidate)
            })
        } else if std::env::var_os("BUTTERCUP_SAM31_PUPIL_INNER_OCCLUDE_FIRST").is_some() {
            selected_first.map(|candidate| {
                inner_occluded_pupil_preview(
                    &preview,
                    suite.source_width,
                    suite.source_height,
                    candidate,
                    5,
                )
            })
        } else {
            None
        };
        Ok(PromptLabRow {
            source: target.source.display().to_string(),
            sequence: target.frame.sequence,
            sensor_origin: (target.frame.sensor_x, target.frame.sensor_y),
            logical_history_sequences,
            preview,
            conditioning_preview,
            width: suite.source_width,
            height: suite.source_height,
            steps,
            video_feature_shapes: suite.video_feature_shapes,
            elapsed_ms: suite.elapsed_ms,
            label_path: label_path.map(|path| path.display().to_string()),
            visible_label_points: labels.len(),
        })
    }

    /// Scores the strongest smooth convex boundary run without assuming that
    /// the pre-affine projection is circular. A projected limbus may be a very
    /// eccentric ellipse; what survives affine projection is ordered,
    /// one-direction turning whose curvature changes gradually. Straight
    /// chords, isolated corners, and oscillating/jagged contours score poorly.
    fn best_boundary_segment_arc_score(
        mask: &ProposalMask,
        width: usize,
        height: usize,
        occupied: usize,
    ) -> f64 {
        if width < 3 || height < 3 || occupied < 8 || mask.boundary_pixels.len() < 12 {
            return 0.0;
        }

        let mut centroid = (0.0, 0.0);
        for (index, &pixel) in mask.pixels.iter().enumerate() {
            if pixel != 0 {
                centroid.0 += (index % width) as f64 + 0.5;
                centroid.1 += (index / width) as f64 + 0.5;
            }
        }
        centroid.0 /= occupied as f64;
        centroid.1 /= occupied as f64;

        // Angular ordering is used only to form local contiguous runs. It does
        // not enter the score as a global circularity assumption.
        let mut boundary = mask
            .boundary_pixels
            .iter()
            .filter_map(|&linear| {
                let linear = linear as usize;
                (linear < width * height).then(|| {
                    let point = ((linear % width) as f64 + 0.5, (linear / width) as f64 + 0.5);
                    let angle = (point.1 - centroid.1).atan2(point.0 - centroid.0);
                    (angle, point)
                })
            })
            .collect::<Vec<_>>();
        boundary.sort_by(|left, right| left.0.total_cmp(&right.0));
        if boundary.len() < 12 {
            return 0.0;
        }

        let count = boundary.len();
        let starts = count.min(48);
        let mut best = 0.0f64;
        // Examine local runs from roughly 30 through 120 degrees of a
        // star-shaped contour. The metric itself remains independent of that
        // angle and accepts the changing curvature of an affine ellipse.
        for denominator in [12usize, 8, 6, 4, 3] {
            let window = (count / denominator).max(12).min(count / 2);
            if window < 12 {
                continue;
            }
            for start_slot in 0..starts {
                let start = start_slot * count / starts;
                let sample_count = 17usize.min(window);
                let mut points = Vec::with_capacity(sample_count);
                for sample in 0..sample_count {
                    let offset = sample * (window - 1) / (sample_count - 1);
                    points.push(boundary[(start + offset) % count].1);
                }
                best = best.max(boundary_arc_coherence(&points));
            }
        }
        best
    }

    fn boundary_arc_coherence(points: &[(f64, f64)]) -> f64 {
        if points.len() < 7 {
            return 0.0;
        }
        let mut directions = Vec::with_capacity(points.len() - 1);
        let mut path_length = 0.0;
        for pair in points.windows(2) {
            let delta = (pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
            let length = delta.0.hypot(delta.1);
            if length <= 1.0e-6 {
                continue;
            }
            path_length += length;
            directions.push(delta.1.atan2(delta.0));
        }
        if directions.len() < 6 {
            return 0.0;
        }

        let mut turns = Vec::with_capacity(directions.len() - 1);
        for pair in directions.windows(2) {
            let mut turn = pair[1] - pair[0];
            while turn > std::f64::consts::PI {
                turn -= std::f64::consts::TAU;
            }
            while turn < -std::f64::consts::PI {
                turn += std::f64::consts::TAU;
            }
            turns.push(turn);
        }
        let total_abs = turns.iter().map(|turn| turn.abs()).sum::<f64>();
        if total_abs < 0.16 {
            return 0.0;
        }
        let signed = turns.iter().sum::<f64>();
        let directional_coherence = (signed.abs() / total_abs).clamp(0.0, 1.0);
        if directional_coherence < 0.58 {
            return 0.0;
        }

        // A single sharp corner must not masquerade as a curved segment.
        let maximum_turn = turns.iter().map(|turn| turn.abs()).fold(0.0f64, f64::max);
        let concentration = (1.0 - maximum_turn / total_abs).clamp(0.0, 1.0);
        let distributed_turning = (concentration / 0.72).clamp(0.0, 1.0);

        // Curvature magnitude may change greatly along an affine ellipse, but
        // it changes smoothly. Penalize high-frequency curvature jerk rather
        // than deviation from constant (circular) curvature.
        let mean_turn = total_abs / turns.len() as f64;
        let curvature_jerk = turns
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .sum::<f64>()
            / turns.len().saturating_sub(1).max(1) as f64;
        let smooth_curvature = 1.0 / (1.0 + 1.6 * curvature_jerk / (mean_turn + 0.025));

        let chord = points
            .first()
            .zip(points.last())
            .map(|(first, last)| (last.0 - first.0).hypot(last.1 - first.1))
            .unwrap_or_default();
        let nondegenerate_path = if path_length > 0.0 {
            ((path_length / chord.max(1.0) - 1.0) / 0.025).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let visible_turn = (total_abs / 0.55).clamp(0.0, 1.0);
        visible_turn
            * directional_coherence.powi(2)
            * distributed_turning
            * smooth_curvature
            * nondegenerate_path
    }

    fn environment_path(name: &str, default: &str) -> PathBuf {
        std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(default))
    }

    fn load_raw(path: &Path) -> Result<LoadedRaw, String> {
        if path.extension() != Some(OsStr::new("raw10")) {
            return Err(format!("expected a .raw10 input, got {}", path.display()));
        }
        let metadata_path = path.with_extension("json");
        let metadata: Value = if metadata_path.is_file() {
            serde_json::from_slice(
                &fs::read(&metadata_path)
                    .map_err(|error| format!("read {}: {error}", metadata_path.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", metadata_path.display()))?
        } else {
            json!({
                "width": sam31_outer::FRAME_WIDTH,
                "height": sam31_outer::FRAME_HEIGHT,
                "stride": sam31_outer::FRAME_WIDTH / 4 * 5,
                "sequence": 0,
                "timestamp_ns": 0,
                "sensor_x": 0,
                "sensor_y": 0,
                "eye_id": 1,
            })
        };
        let number = |key: &str, fallback: u64| {
            metadata
                .get(key)
                .and_then(Value::as_u64)
                .unwrap_or(fallback)
        };
        let width = number("width", sam31_outer::FRAME_WIDTH as u64) as usize;
        let height = number("height", sam31_outer::FRAME_HEIGHT as u64) as usize;
        let stride = number("stride", (width / 4 * 5) as u64) as usize;
        if width != sam31_outer::FRAME_WIDTH || height != sam31_outer::FRAME_HEIGHT {
            return Err(format!(
                "{} is {width}x{height}; the current SAM3.1 graph requires {}x{}",
                path.display(),
                sam31_outer::FRAME_WIDTH,
                sam31_outer::FRAME_HEIGHT,
            ));
        }
        let packed = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let pixels = raw10::try_unpack_raw10(&packed, width, height, stride)?;
        let eye_id = number("eye_id", 1).max(1) as usize;
        Ok(LoadedRaw {
            source: path.to_path_buf(),
            frame: Arc::new(RawFrame {
                eye_index: eye_id.saturating_sub(1).min(1),
                sequence: number("sequence", 0),
                timestamp_ns: number("timestamp_ns", 0),
                sensor_x: number("sensor_x", 0) as u32,
                sensor_y: number("sensor_y", 0) as u32,
                width,
                height,
                registration_anchor: None,
                pupil_component_seed: None,
                pixels: Arc::new(pixels),
            }),
        })
    }

    fn load_capture_history(
        capture: &Path,
        target_sequence: u64,
        eye_label: &str,
    ) -> Result<Vec<LoadedRaw>, String> {
        let index_path = capture.join("frames.jsonl");
        let index = fs::read_to_string(&index_path)
            .map_err(|error| format!("read {}: {error}", index_path.display()))?;
        let mut records = index
            .lines()
            .enumerate()
            .map(|(line_index, line)| {
                serde_json::from_str::<Value>(line).map_err(|error| {
                    format!(
                        "parse {} line {}: {error}",
                        index_path.display(),
                        line_index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        records.retain(|record| {
            record.get("label").and_then(Value::as_str) == Some(eye_label)
                && record
                    .get("sequence")
                    .and_then(Value::as_u64)
                    .is_some_and(|sequence| sequence <= target_sequence)
        });
        records.sort_by_key(|record| record.get("sequence").and_then(Value::as_u64));
        let found_target = records
            .last()
            .and_then(|record| record.get("sequence"))
            .and_then(Value::as_u64)
            == Some(target_sequence);
        if !found_target {
            return Err(format!(
                "capture {} has no {eye_label} sequence {target_sequence}",
                capture.display()
            ));
        }
        let maximum_frames = std::env::var("BUTTERCUP_SAM31_VIDEO_MAX_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(VIDEO_FEATURE_DEFAULT_MAX_FRAMES)
            .max(2);
        if records.len() > maximum_frames {
            records = records.split_off(records.len() - maximum_frames);
        }
        let mut loaded = Vec::with_capacity(records.len());
        for record in records {
            let number = |key: &str, fallback: u64| {
                record.get(key).and_then(Value::as_u64).unwrap_or(fallback)
            };
            let width = number("width", sam31_outer::FRAME_WIDTH as u64) as usize;
            let height = number("height", sam31_outer::FRAME_HEIGHT as u64) as usize;
            let stride = number("stride", (width / 4 * 5) as u64) as usize;
            if width != sam31_outer::FRAME_WIDTH || height != sam31_outer::FRAME_HEIGHT {
                return Err(format!(
                    "capture frame {} is {width}x{height}; expected {}x{}",
                    number("sequence", 0),
                    sam31_outer::FRAME_WIDTH,
                    sam31_outer::FRAME_HEIGHT,
                ));
            }
            let stream_name = record
                .get("stream")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("capture frame {} has no stream", number("sequence", 0)))?;
            let stream_path = capture.join(stream_name);
            let offset = number("offset", u64::MAX);
            let length = number("length", 0) as usize;
            if offset == u64::MAX || length == 0 {
                return Err(format!(
                    "capture frame {} has invalid offset/length",
                    number("sequence", 0)
                ));
            }
            let mut stream = fs::File::open(&stream_path)
                .map_err(|error| format!("open {}: {error}", stream_path.display()))?;
            stream
                .seek(SeekFrom::Start(offset))
                .map_err(|error| format!("seek {} to {offset}: {error}", stream_path.display()))?;
            let mut packed = vec![0u8; length];
            stream.read_exact(&mut packed).map_err(|error| {
                format!(
                    "read {} bytes at {offset} from {}: {error}",
                    packed.len(),
                    stream_path.display()
                )
            })?;
            let pixels = raw10::try_unpack_raw10(&packed, width, height, stride)?;
            let sequence = number("sequence", 0);
            loaded.push(LoadedRaw {
                // A stable virtual per-frame name lets the ordinary label
                // discovery path find canonical labels without extracting or
                // copying the lossless capture stream.
                source: capture.join(format!("{eye_label}-seq-{sequence}.raw10")),
                frame: Arc::new(RawFrame {
                    eye_index: number("eye_id", 1).max(1) as usize - 1,
                    sequence,
                    timestamp_ns: number("timestamp_ns", 0),
                    sensor_x: number("sensor_x", 0) as u32,
                    sensor_y: number("sensor_y", 0) as u32,
                    width,
                    height,
                    registration_anchor: None,
                    pupil_component_seed: None,
                    pixels: Arc::new(pixels),
                }),
            });
        }
        Ok(loaded)
    }

    fn ellipse_point_normal(ellipse: Ellipse, phase: f64) -> ((f64, f64), (f64, f64)) {
        let (sin_phase, cos_phase) = phase.sin_cos();
        let (sin_angle, cos_angle) = ellipse.angle.sin_cos();
        let local_x = ellipse.major_radius * cos_phase;
        let local_y = ellipse.minor_radius * sin_phase;
        let point = (
            ellipse.center.0 + cos_angle * local_x - sin_angle * local_y,
            ellipse.center.1 + sin_angle * local_x + cos_angle * local_y,
        );
        let normal_local_x = cos_phase / ellipse.major_radius.max(1.0);
        let normal_local_y = sin_phase / ellipse.minor_radius.max(1.0);
        let normal_x = cos_angle * normal_local_x - sin_angle * normal_local_y;
        let normal_y = sin_angle * normal_local_x + cos_angle * normal_local_y;
        let length = normal_x.hypot(normal_y).max(1.0e-9);
        (point, (normal_x / length, normal_y / length))
    }

    fn ellipse_phase(ellipse: Ellipse, point: (f64, f64)) -> f64 {
        let (sin_angle, cos_angle) = ellipse.angle.sin_cos();
        let dx = point.0 - ellipse.center.0;
        let dy = point.1 - ellipse.center.1;
        let local_x = cos_angle * dx + sin_angle * dy;
        let local_y = -sin_angle * dx + cos_angle * dy;
        (local_y / ellipse.minor_radius.max(1.0))
            .atan2(local_x / ellipse.major_radius.max(1.0))
            .rem_euclid(std::f64::consts::TAU)
    }

    fn raw_sample(raw: &[u16], width: usize, height: usize, x: f64, y: f64) -> f64 {
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x >= width.saturating_sub(1) as f64
            || y >= height.saturating_sub(1) as f64
        {
            return 0.0;
        }
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let at = |xx: usize, yy: usize| raw[yy * width + xx] as f64;
        let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
        let bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    fn sample_proposal(
        mask: &ProposalMask,
        mask_width: usize,
        mask_height: usize,
        x: f64,
        y: f64,
        source_width: usize,
        source_height: usize,
    ) -> f64 {
        if mask.pixels.len() != mask_width * mask_height
            || x < 0.0
            || y < 0.0
            || x >= source_width as f64
            || y >= source_height as f64
        {
            return 0.0;
        }
        let low_x = ((x + 0.5) * mask_width as f64 / source_width as f64 - 0.5)
            .round()
            .clamp(0.0, mask_width.saturating_sub(1) as f64) as usize;
        let low_y = ((y + 0.5) * mask_height as f64 / source_height as f64 - 0.5)
            .round()
            .clamp(0.0, mask_height.saturating_sub(1) as f64) as usize;
        f64::from(mask.pixels[low_y * mask_width + low_x] != 0)
    }

    fn proposal_geometry_score(
        pass: &SemanticProposalMasks,
        mask: &ProposalMask,
        prompt_index: usize,
        ellipse: Ellipse,
        source_width: usize,
        source_height: usize,
    ) -> (f64, f64) {
        let sample = |x: f64, y: f64| {
            sample_proposal(
                mask,
                pass.width,
                pass.height,
                x,
                y,
                source_width,
                source_height,
            )
        };
        let mut inner = 0.0;
        let mut outer = 0.0;
        let mut boundary = 0.0;
        let mut far_inner = 0.0;
        let mut far_outer = 0.0;
        let phases = 96;
        for index in 0..phases {
            let phase = std::f64::consts::TAU * index as f64 / phases as f64;
            let (point, normal) = ellipse_point_normal(ellipse, phase);
            let inward = [3.0, 7.0, 12.0]
                .iter()
                .map(|distance| {
                    sample(point.0 - normal.0 * distance, point.1 - normal.1 * distance)
                })
                .sum::<f64>()
                / 3.0;
            let outward = [3.0, 7.0, 12.0]
                .iter()
                .map(|distance| {
                    sample(point.0 + normal.0 * distance, point.1 + normal.1 * distance)
                })
                .sum::<f64>()
                / 3.0;
            inner += inward;
            outer += outward;
            boundary += [
                sample(point.0, point.1),
                sample(point.0 - normal.0 * 2.0, point.1 - normal.1 * 2.0),
                sample(point.0 + normal.0 * 2.0, point.1 + normal.1 * 2.0),
            ]
            .into_iter()
            .fold(0.0, f64::max);
            far_inner += sample(
                ellipse.center.0 + (point.0 - ellipse.center.0) * 0.45,
                ellipse.center.1 + (point.1 - ellipse.center.1) * 0.45,
            );
            far_outer += sample(
                ellipse.center.0 + (point.0 - ellipse.center.0) * 1.35,
                ellipse.center.1 + (point.1 - ellipse.center.1) * 1.35,
            );
        }
        let denominator = phases as f64;
        inner /= denominator;
        outer /= denominator;
        boundary /= denominator;
        far_inner /= denominator;
        far_outer /= denominator;
        // Recompute the side terms without the running-boundary shortcut.
        let side_support = |want_left: bool| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for index in 0..phases {
                let phase = std::f64::consts::TAU * index as f64 / phases as f64;
                if (want_left && phase.cos() < -0.40) || (!want_left && phase.cos() > 0.40) {
                    let (point, normal) = ellipse_point_normal(ellipse, phase);
                    sum += [
                        sample(point.0, point.1),
                        sample(point.0 - normal.0 * 4.0, point.1 - normal.1 * 4.0),
                        sample(point.0 + normal.0 * 4.0, point.1 + normal.1 * 4.0),
                    ]
                    .into_iter()
                    .fold(0.0, f64::max);
                    count += 1;
                }
            }
            sum / count.max(1) as f64
        };
        let left = side_support(true);
        let right = side_support(false);
        let area_fraction = mask.pixels.iter().filter(|&&pixel| pixel != 0).count() as f64
            / mask.pixels.len().max(1) as f64;
        let model_term = (mask.score as f64).clamp(-1.0, 1.0) * 0.06;
        let objective = match prompt_index {
            OUTER => boundary + 0.35 * inner - 0.35 * outer + model_term,
            VISIBLE_IRIS => {
                1.10 * inner - 0.70 * outer + 0.25 * boundary + 0.10 * far_inner - 0.20 * far_outer
                    + model_term
            }
            ADJACENT_SCLERA => {
                1.15 * outer - 0.72 * inner + 0.20 * boundary + 0.12 * far_outer - 0.20 * far_inner
                    + model_term
            }
            VISIBLE_ARCS => {
                1.25 * boundary - 0.35 * far_inner - 0.35 * far_outer - 0.30 * area_fraction
                    + model_term
            }
            OCCLUDERS | UPPER_OCCLUSION | LOWER_OCCLUSION => {
                0.90 * boundary + 0.25 * outer - 0.35 * far_inner + model_term
            }
            LEFT_SLICE => 1.10 * left - 0.45 * right + 0.20 * boundary + model_term,
            RIGHT_SLICE => 1.10 * right - 0.45 * left + 0.20 * boundary + model_term,
            _ => boundary + model_term,
        };
        (objective, area_fraction)
    }

    fn build_mask_field(
        pass: &SemanticProposalMasks,
        prompt_index: usize,
        ellipse: Ellipse,
        source_width: usize,
        source_height: usize,
    ) -> MaskField {
        if pass.width == 0 || pass.height == 0 || pass.masks.is_empty() {
            return MaskField::empty();
        }
        let mut ranked = pass
            .masks
            .iter()
            .filter(|mask| mask.pixels.iter().any(|&pixel| pixel != 0))
            .map(|mask| {
                let (objective, area_fraction) = proposal_geometry_score(
                    pass,
                    mask,
                    prompt_index,
                    ellipse,
                    source_width,
                    source_height,
                );
                RankedMask {
                    query: mask.query,
                    objective,
                    model_score: mask.score,
                    area_fraction,
                }
            })
            .filter(|ranked| ranked.objective.is_finite())
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .objective
                .total_cmp(&left.objective)
                .then_with(|| right.model_score.total_cmp(&left.model_score))
        });
        let maximum = match prompt_index {
            ADJACENT_SCLERA | OCCLUDERS | UPPER_OCCLUSION | LOWER_OCCLUSION => 4,
            LEFT_SLICE | RIGHT_SLICE => 2,
            _ => 1,
        };
        let margin = match prompt_index {
            ADJACENT_SCLERA | OCCLUDERS | UPPER_OCCLUSION | LOWER_OCCLUSION => 0.24,
            _ => 0.14,
        };
        let best = ranked.first().map(|ranked| ranked.objective);
        let selected = ranked
            .into_iter()
            .filter(|ranked| best.is_some_and(|best| ranked.objective >= best - margin))
            .take(maximum)
            .collect::<Vec<_>>();
        let selected_queries = selected
            .iter()
            .map(|ranked| ranked.query)
            .collect::<BTreeSet<_>>();
        let mut pixels = vec![0u8; pass.width * pass.height];
        // Side-slice questions are used as edge-localizers. The direct visible
        // arc question is instead a visibility region: its filled mask was
        // empirically the useful one-prompt gate, while reducing it to the
        // mask perimeter discarded the labeled limbus arc.
        let boundary_answer = matches!(prompt_index, LEFT_SLICE | RIGHT_SLICE);
        for mask in pass
            .masks
            .iter()
            .filter(|mask| selected_queries.contains(&mask.query))
        {
            if boundary_answer {
                // SAM returns filled object masks even when the language asks
                // about an edge. Preserve the answer's actual boundary and a
                // one-cell uncertainty band instead of treating its entire
                // interior as a trustworthy limbus arc.
                for &index in mask.boundary_pixels.iter() {
                    let index = index as usize;
                    let x = index % pass.width;
                    let y = index / pass.width;
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            let xx = x as isize + dx;
                            let yy = y as isize + dy;
                            if xx >= 0
                                && yy >= 0
                                && xx < pass.width as isize
                                && yy < pass.height as isize
                            {
                                pixels[yy as usize * pass.width + xx as usize] = 1;
                            }
                        }
                    }
                }
            } else {
                for (destination, &source) in pixels.iter_mut().zip(mask.pixels.iter()) {
                    *destination |= u8::from(source != 0);
                }
            }
        }
        MaskField {
            width: pass.width,
            height: pass.height,
            pixels,
            selected,
        }
    }

    fn mean_field_along(
        field: &MaskField,
        point: (f64, f64),
        normal: (f64, f64),
        distances: &[f64],
        width: usize,
        height: usize,
    ) -> f64 {
        distances
            .iter()
            .map(|distance| {
                field.sample(
                    point.0 + normal.0 * distance,
                    point.1 + normal.1 * distance,
                    width,
                    height,
                )
            })
            .sum::<f64>()
            / distances.len().max(1) as f64
    }

    fn arc_evidence(
        point: (f64, f64),
        ellipse: Ellipse,
        fields: &[MaskField],
        raw: &[u16],
        width: usize,
        height: usize,
    ) -> ArcEvidence {
        let phase = ellipse_phase(ellipse, point);
        let (_, normal) = ellipse_point_normal(ellipse, phase);
        let inward = [-3.0, -7.0, -12.0];
        let outward = [3.0, 7.0, 12.0];
        let visible_iris =
            mean_field_along(&fields[VISIBLE_IRIS], point, normal, &inward, width, height);
        let adjacent_sclera = mean_field_along(
            &fields[ADJACENT_SCLERA],
            point,
            normal,
            &outward,
            width,
            height,
        );
        let direct_arc = mean_field_along(
            &fields[VISIBLE_ARCS],
            point,
            normal,
            &[-2.0, 0.0, 2.0],
            width,
            height,
        );
        let occluder = mean_field_along(
            &fields[OCCLUDERS],
            point,
            normal,
            &[-3.0, 0.0, 3.0],
            width,
            height,
        );
        let pizza_slice = fields[LEFT_SLICE]
            .sample(point.0, point.1, width, height)
            .max(fields[RIGHT_SLICE].sample(point.0, point.1, width, height));
        let upper_lower_occlusion = fields[UPPER_OCCLUSION]
            .sample(point.0, point.1, width, height)
            .max(fields[LOWER_OCCLUSION].sample(point.0, point.1, width, height));
        let inner_raw = inward
            .iter()
            .map(|distance| {
                raw_sample(
                    raw,
                    width,
                    height,
                    point.0 + normal.0 * distance,
                    point.1 + normal.1 * distance,
                )
            })
            .sum::<f64>()
            / inward.len() as f64;
        let outer_raw = outward
            .iter()
            .map(|distance| {
                raw_sample(
                    raw,
                    width,
                    height,
                    point.0 + normal.0 * distance,
                    point.1 + normal.1 * distance,
                )
            })
            .sum::<f64>()
            / outward.len() as f64;
        let raw_order = ((outer_raw - inner_raw) / (outer_raw + inner_raw + 32.0) * 4.0)
            .clamp(-1.0, 1.0)
            .mul_add(0.5, 0.5);
        ArcEvidence {
            point,
            phase,
            raw_order,
            visible_iris,
            adjacent_sclera,
            direct_arc,
            occluder,
            pizza_slice,
            upper_lower_occlusion,
        }
    }

    fn strategy_score(name: &str, evidence: &ArcEvidence) -> f64 {
        match name {
            "outer-only" => 0.65 + 0.35 * evidence.raw_order,
            "direct-arc-1" => 0.72 * evidence.direct_arc + 0.28 * evidence.raw_order,
            "material-pair-2" => {
                0.72 * (evidence.visible_iris * evidence.adjacent_sclera).sqrt()
                    + 0.28 * evidence.raw_order
            }
            "occlusion-guard-3" => {
                (0.74 * (evidence.visible_iris * evidence.adjacent_sclera).sqrt()
                    + 0.26 * evidence.raw_order)
                    * (1.0 - 0.82 * evidence.occluder).clamp(0.0, 1.0)
            }
            "pizza-sides-2" => {
                0.62 * evidence.pizza_slice
                    + 0.23 * (evidence.visible_iris * evidence.adjacent_sclera).sqrt()
                    + 0.15 * evidence.raw_order
            }
            "lid-material-3" => {
                (0.72 * evidence.visible_iris + 0.28 * evidence.raw_order)
                    * (1.0 - 0.82 * evidence.upper_lower_occlusion).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }

    fn circular_distance(first: f64, second: f64) -> f64 {
        let delta = (first - second).abs().rem_euclid(std::f64::consts::TAU);
        delta.min(std::f64::consts::TAU - delta)
    }

    fn select_trusted(evidence: &[ArcEvidence], scores: &[f64], outer_only: bool) -> Vec<bool> {
        if outer_only {
            return vec![true; evidence.len()];
        }
        let mut population = scores
            .iter()
            .copied()
            .filter(|score| score.is_finite())
            .collect::<Vec<_>>();
        population.sort_by(f64::total_cmp);
        let quantile = population
            .get(population.len().saturating_mul(42) / 100)
            .copied()
            .unwrap_or(1.0);
        let threshold = quantile.max(0.22);
        let mut selected = scores
            .iter()
            .map(|score| score.is_finite() && *score >= threshold)
            .collect::<Vec<_>>();
        // Suppress isolated low-resolution mask pixels while retaining arc
        // endpoints. Neighbors are defined by conic phase, not vector index,
        // so a curvature-censored gap cannot accidentally bridge a chord.
        let original = selected.clone();
        for index in 0..selected.len() {
            if !original[index] {
                continue;
            }
            let neighbors = evidence
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index && original[*other])
                .filter(|(_, other)| circular_distance(evidence[index].phase, other.phase) < 0.13)
                .count();
            if neighbors < 2 {
                selected[index] = false;
            }
        }
        if selected.iter().filter(|&&value| value).count() < 10 {
            let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
            ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
            selected.fill(false);
            for (index, _) in ranked.into_iter().take(evidence.len().min(16)) {
                selected[index] = true;
            }
        }
        selected
    }

    fn bounded_refit(reference: Ellipse, points: &[(f64, f64)]) -> Option<Ellipse> {
        let candidate = sam31_outer::fit_trusted_arc_points(points)?;
        let scale = (reference.major_radius * reference.minor_radius)
            .sqrt()
            .max(1.0);
        let center_error = (candidate.center.0 - reference.center.0)
            .hypot(candidate.center.1 - reference.center.1)
            / scale;
        let major_ratio = candidate.major_radius / reference.major_radius.max(1.0);
        let minor_ratio = candidate.minor_radius / reference.minor_radius.max(1.0);
        let area_ratio = major_ratio * minor_ratio;
        (center_error <= 0.28
            && (0.68..=1.35).contains(&major_ratio)
            && (0.68..=1.35).contains(&minor_ratio)
            && (0.58..=1.52).contains(&area_ratio))
        .then_some(candidate)
    }

    fn ellipse_residual_px(ellipse: Ellipse, point: (f64, f64)) -> f64 {
        let (sin_angle, cos_angle) = ellipse.angle.sin_cos();
        let dx = point.0 - ellipse.center.0;
        let dy = point.1 - ellipse.center.1;
        let local_x = cos_angle * dx + sin_angle * dy;
        let local_y = -sin_angle * dx + cos_angle * dy;
        let radius = ((local_x / ellipse.major_radius.max(1.0)).powi(2)
            + (local_y / ellipse.minor_radius.max(1.0)).powi(2))
        .sqrt();
        (radius - 1.0).abs() * (ellipse.major_radius * ellipse.minor_radius).sqrt()
    }

    fn build_strategies(
        evidence: &[ArcEvidence],
        reference: Ellipse,
        labels: &[(f64, f64)],
    ) -> Vec<Strategy> {
        const NONE: &[usize] = &[];
        const DIRECT: &[usize] = &[VISIBLE_ARCS];
        const MATERIAL: &[usize] = &[VISIBLE_IRIS, ADJACENT_SCLERA];
        const GUARDED: &[usize] = &[VISIBLE_IRIS, ADJACENT_SCLERA, OCCLUDERS];
        const PIZZA: &[usize] = &[LEFT_SLICE, RIGHT_SLICE];
        const LID_MATERIAL: &[usize] = &[VISIBLE_IRIS, UPPER_OCCLUSION, LOWER_OCCLUSION];
        let definitions = [
            ("outer-only", NONE),
            ("direct-arc-1", DIRECT),
            ("material-pair-2", MATERIAL),
            ("occlusion-guard-3", GUARDED),
            ("pizza-sides-2", PIZZA),
            ("lid-material-3", LID_MATERIAL),
        ];
        definitions
            .into_iter()
            .map(|(name, prompts)| {
                let scores = evidence
                    .iter()
                    .map(|sample| strategy_score(name, sample))
                    .collect::<Vec<_>>();
                let trusted = select_trusted(evidence, &scores, name == "outer-only");
                let points = evidence
                    .iter()
                    .zip(trusted.iter())
                    .filter_map(|(sample, &trusted)| trusted.then_some(sample.point))
                    .collect::<Vec<_>>();
                let fitted = bounded_refit(reference, &points).or(Some(reference));
                let coverage = points.len() as f64 / evidence.len().max(1) as f64;
                let sectors = evidence
                    .iter()
                    .zip(trusted.iter())
                    .filter_map(|(sample, &trusted)| {
                        trusted.then_some(
                            ((sample.phase / std::f64::consts::TAU * 16.0).floor() as usize)
                                .min(15),
                        )
                    })
                    .collect::<BTreeSet<_>>()
                    .len();
                let sector_coverage = sectors as f64 / 16.0;
                let mean_score = scores
                    .iter()
                    .zip(trusted.iter())
                    .filter_map(|(&score, &trusted)| trusted.then_some(score))
                    .sum::<f64>()
                    / points.len().max(1) as f64;
                let fit_residual_px = fitted.map(|ellipse| {
                    points
                        .iter()
                        .map(|&point| ellipse_residual_px(ellipse, point))
                        .sum::<f64>()
                        / points.len().max(1) as f64
                });
                let scale = (reference.major_radius * reference.minor_radius)
                    .sqrt()
                    .max(1.0);
                let fit_term = fit_residual_px
                    .map(|residual| (-residual / (0.07 * scale).max(1.0)).exp())
                    .unwrap_or(0.0);
                let coverage_term = (1.0 - ((coverage - 0.58) / 0.58).abs()).clamp(0.0, 1.0);
                let internal_quality = 0.36 * mean_score.clamp(0.0, 1.0)
                    + 0.28 * sector_coverage
                    + 0.18 * coverage_term
                    + 0.18 * fit_term;
                let (label_mean_px, label_rms_px, label_max_px) = fitted
                    .filter(|_| !labels.is_empty())
                    .map(|ellipse| {
                        let errors = labels
                            .iter()
                            .map(|&point| ellipse_residual_px(ellipse, point))
                            .collect::<Vec<_>>();
                        (
                            Some(errors.iter().sum::<f64>() / errors.len() as f64),
                            Some(
                                (errors.iter().map(|error| error * error).sum::<f64>()
                                    / errors.len() as f64)
                                    .sqrt(),
                            ),
                            errors.into_iter().max_by(f64::total_cmp),
                        )
                    })
                    .unwrap_or((None, None, None));
                Strategy {
                    name,
                    follow_on_prompts: prompts,
                    scores,
                    trusted,
                    fitted,
                    coverage,
                    sector_coverage,
                    mean_score,
                    fit_residual_px,
                    internal_quality,
                    label_mean_px,
                    label_rms_px,
                    label_max_px,
                }
            })
            .collect()
    }

    fn locate_labels(raw: &Path) -> Option<PathBuf> {
        let sibling = raw.with_extension("labels.json");
        if sibling.is_file() {
            return Some(sibling);
        }
        let wanted = format!(
            "{}.labels.json",
            raw.file_stem().and_then(OsStr::to_str).unwrap_or_default()
        );
        // Capture-backed virtual RAW names live below `extracted/<capture>`;
        // the canonical labeler stores evidence beside `extracted`, under the
        // capture set's `annotator/labels` directory.
        let mut ancestor = raw.parent();
        for _ in 0..5 {
            let Some(directory) = ancestor else { break };
            let candidate = directory.join("annotator/labels").join(&wanted);
            if candidate.is_file() {
                return Some(candidate);
            }
            ancestor = directory.parent();
        }
        let root = Path::new("/mnt/bulk_data/buttercup-eye-tracking/labeled-corpus");
        let mut stack = vec![root.to_path_buf()];
        while let Some(directory) = stack.pop() {
            let Ok(entries) = fs::read_dir(directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name() == Some(OsStr::new(&wanted)) {
                    return Some(path);
                }
            }
        }
        None
    }

    fn mask_label_boundary_metrics(
        mask: &[u8],
        mask_width: usize,
        mask_height: usize,
        native_width: usize,
        native_height: usize,
        labels: &[(f64, f64)],
    ) -> Option<(f64, f64, f64)> {
        if labels.is_empty()
            || mask_width < 3
            || mask_height < 3
            || mask.len() != mask_width * mask_height
        {
            return None;
        }
        let mut boundary = Vec::new();
        for y in 1..mask_height - 1 {
            for x in 1..mask_width - 1 {
                let index = y * mask_width + x;
                if mask[index] == 0 {
                    continue;
                }
                if mask[index - 1] == 0
                    || mask[index + 1] == 0
                    || mask[index - mask_width] == 0
                    || mask[index + mask_width] == 0
                {
                    boundary.push((
                        (x as f64 + 0.5) * native_width as f64 / mask_width as f64,
                        (y as f64 + 0.5) * native_height as f64 / mask_height as f64,
                    ));
                }
            }
        }
        if boundary.is_empty() {
            return None;
        }
        let errors = labels
            .iter()
            .map(|label| {
                boundary
                    .iter()
                    .map(|point| (point.0 - label.0).hypot(point.1 - label.1))
                    .min_by(f64::total_cmp)
                    .unwrap_or(f64::INFINITY)
            })
            .collect::<Vec<_>>();
        Some((
            errors.iter().sum::<f64>() / errors.len() as f64,
            (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt(),
            errors.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0),
        ))
    }

    fn ellipse_label_metrics(ellipse: Ellipse, labels: &[(f64, f64)]) -> (f64, f64, f64) {
        let errors = labels
            .iter()
            .map(|&point| ellipse_residual_px(ellipse, point))
            .collect::<Vec<_>>();
        (
            errors.iter().sum::<f64>() / errors.len().max(1) as f64,
            (errors.iter().map(|error| error * error).sum::<f64>() / errors.len().max(1) as f64)
                .sqrt(),
            errors.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0),
        )
    }

    fn read_visible_labels(path: &Path) -> Result<Vec<(f64, f64)>, String> {
        let document: Value = serde_json::from_slice(
            &fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?,
        )
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
        Ok(document
            .get("annotation_points")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|point| point.get("visibility").and_then(Value::as_str) == Some("visible"))
            .filter(|point| {
                matches!(
                    point.get("kind").and_then(Value::as_str),
                    Some("iris_edge" | "limbus_edge")
                )
            })
            .filter_map(|point| Some((point.get("x")?.as_f64()?, point.get("y")?.as_f64()?)))
            .collect())
    }

    fn ellipse_json(ellipse: Ellipse) -> Value {
        json!({
            "center": [ellipse.center.0, ellipse.center.1],
            "major_radius": ellipse.major_radius,
            "minor_radius": ellipse.minor_radius,
            "angle_radians": ellipse.angle,
            "post_affine_area_radius_px": (ellipse.major_radius * ellipse.minor_radius).sqrt(),
        })
    }

    fn strategy_json(strategy: &Strategy) -> Value {
        json!({
            "name": strategy.name,
            "follow_on_prompt_count": strategy.follow_on_prompts.len(),
            "follow_on_prompt_indices": strategy.follow_on_prompts,
            "trusted_points": strategy.trusted.iter().filter(|&&trusted| trusted).count(),
            "candidate_points": strategy.trusted.len(),
            "coverage": strategy.coverage,
            "sector_coverage": strategy.sector_coverage,
            "mean_evidence_score": strategy.mean_score,
            "fit_residual_px": strategy.fit_residual_px,
            "internal_quality": strategy.internal_quality,
            "ellipse": strategy.fitted.map(ellipse_json),
            "human_label_mean_error_px": strategy.label_mean_px,
            "human_label_rms_error_px": strategy.label_rms_px,
            "human_label_max_error_px": strategy.label_max_px,
        })
    }

    fn latest_quad_preview(
        filmstrip: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<[u8; 3]>, String> {
        let film_width = width * sam31_outer::HISTORY_FRAMES;
        let plane = film_width * height;
        if filmstrip.len() != plane * 3 {
            return Err(format!(
                "unexpected quantized Quad-RGB filmstrip size {}",
                filmstrip.len()
            ));
        }
        let start_x = film_width - width;
        let mut preview = vec![[0u8; 3]; width * height];
        for y in 0..height {
            for x in 0..width {
                let source = y * film_width + start_x + x;
                preview[y * width + x] = [
                    filmstrip[source],
                    filmstrip[plane + source],
                    filmstrip[2 * plane + source],
                ];
            }
        }
        Ok(preview)
    }

    fn smooth_preview_chroma(source: &[[u8; 3]], width: usize, height: usize) -> Vec<[u8; 3]> {
        if source.len() != width * height || width == 0 || height == 0 {
            return source.to_vec();
        }
        let luma = source
            .iter()
            .map(|pixel| 0.25 * pixel[0] as f64 + 0.50 * pixel[1] as f64 + 0.25 * pixel[2] as f64)
            .collect::<Vec<_>>();
        let red_difference = source
            .iter()
            .zip(luma.iter())
            .map(|(pixel, &luma)| pixel[0] as f64 - luma)
            .collect::<Vec<_>>();
        let blue_difference = source
            .iter()
            .zip(luma.iter())
            .map(|(pixel, &luma)| pixel[2] as f64 - luma)
            .collect::<Vec<_>>();
        let blurred = |plane: &[f64], x: usize, y: usize| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for dy in -2isize..=2 {
                let yy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                for dx in -2isize..=2 {
                    let xx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                    sum += plane[yy * width + xx];
                    count += 1;
                }
            }
            sum / count as f64
        };
        let mut result = Vec::with_capacity(source.len());
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let red = luma[index] + blurred(&red_difference, x, y);
                let blue = luma[index] + blurred(&blue_difference, x, y);
                let green = 2.0 * luma[index] - 0.5 * red - 0.5 * blue;
                result.push([
                    red.round().clamp(0.0, 255.0) as u8,
                    green.round().clamp(0.0, 255.0) as u8,
                    blue.round().clamp(0.0, 255.0) as u8,
                ]);
            }
        }
        result
    }

    fn clean_raw_display_preview(
        raw: &[u16],
        color: &[[u8; 3]],
        width: usize,
        height: usize,
    ) -> Vec<[u8; 3]> {
        if raw.len() != width * height || color.len() != raw.len() || width < 4 || height < 4 {
            return color.to_vec();
        }
        // Display only: use the mean of each local physical Quad-Bayer cell
        // as luminance. Inference and all arc evidence still consume untouched
        // native RAW10. A neutral review is preferable here to reintroducing
        // a distracting CFA lattice merely for display chroma.
        let integral_stride = width + 1;
        let mut integral = vec![0u32; integral_stride * (height + 1)];
        for y in 0..height {
            let mut row_sum = 0u32;
            for x in 0..width {
                row_sum += raw[y * width + x] as u32;
                integral[(y + 1) * integral_stride + x + 1] =
                    integral[y * integral_stride + x + 1] + row_sum;
            }
        }
        let mut neutral = vec![0.0f64; raw.len()];
        for y in 0..height {
            for x in 0..width {
                let x0 = x.saturating_sub(1).min(width - 4);
                let y0 = y.saturating_sub(1).min(height - 4);
                let x1 = x0 + 4;
                let y1 = y0 + 4;
                let sum = integral[y1 * integral_stride + x1] + integral[y0 * integral_stride + x0]
                    - integral[y0 * integral_stride + x1]
                    - integral[y1 * integral_stride + x0];
                neutral[y * width + x] = sum as f64 / 16.0;
            }
        }
        let mut population = neutral.clone();
        population.sort_by(f64::total_cmp);
        let low = population[population.len() * 5 / 1000];
        let high = population[population.len() * 995 / 1000].max(low + 1.0);
        neutral
            .iter()
            .map(|&luma| {
                let mapped = ((luma - low) / (high - low)).clamp(0.0, 1.0).powf(0.82) * 255.0;
                let gray = mapped.round().clamp(0.0, 255.0) as u8;
                [gray, gray, gray]
            })
            .collect()
    }

    fn put_pixel(frame: &mut [[u8; 3]], x: isize, y: isize, color: [u8; 3]) {
        if x >= 0 && y >= 0 && x < OUTPUT_WIDTH as isize && y < OUTPUT_HEIGHT as isize {
            frame[y as usize * OUTPUT_WIDTH + x as usize] = color;
        }
    }

    fn blend_pixel(frame: &mut [[u8; 3]], x: isize, y: isize, color: [u8; 3], alpha: f64) {
        if x < 0 || y < 0 || x >= OUTPUT_WIDTH as isize || y >= OUTPUT_HEIGHT as isize {
            return;
        }
        let pixel = &mut frame[y as usize * OUTPUT_WIDTH + x as usize];
        for channel in 0..3 {
            pixel[channel] = (pixel[channel] as f64 * (1.0 - alpha) + color[channel] as f64 * alpha)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }

    fn draw_disc(frame: &mut [[u8; 3]], x: f64, y: f64, radius: isize, color: [u8; 3]) {
        let center_x = x.round() as isize;
        let center_y = y.round() as isize;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    put_pixel(frame, center_x + dx, center_y + dy, color);
                }
            }
        }
    }

    fn draw_line(
        frame: &mut [[u8; 3]],
        first: (f64, f64),
        second: (f64, f64),
        color: [u8; 3],
        thickness: isize,
    ) {
        let steps = ((second.0 - first.0).abs().max((second.1 - first.1).abs()) * 1.5)
            .ceil()
            .max(1.0) as usize;
        for step in 0..=steps {
            let amount = step as f64 / steps as f64;
            draw_disc(
                frame,
                first.0 + (second.0 - first.0) * amount,
                first.1 + (second.1 - first.1) * amount,
                thickness,
                color,
            );
        }
    }

    fn glyph(character: char) -> [u8; 7] {
        match character {
            'A' => [14, 17, 17, 31, 17, 17, 17],
            'B' => [30, 17, 17, 30, 17, 17, 30],
            'C' => [14, 17, 16, 16, 16, 17, 14],
            'D' => [30, 17, 17, 17, 17, 17, 30],
            'E' => [31, 16, 16, 30, 16, 16, 31],
            'F' => [31, 16, 16, 30, 16, 16, 16],
            'G' => [14, 17, 16, 23, 17, 17, 14],
            'H' => [17, 17, 17, 31, 17, 17, 17],
            'I' => [31, 4, 4, 4, 4, 4, 31],
            'J' => [7, 2, 2, 2, 18, 18, 12],
            'K' => [17, 18, 20, 24, 20, 18, 17],
            'L' => [16, 16, 16, 16, 16, 16, 31],
            'M' => [17, 27, 21, 21, 17, 17, 17],
            'N' => [17, 25, 21, 19, 17, 17, 17],
            'O' => [14, 17, 17, 17, 17, 17, 14],
            'P' => [30, 17, 17, 30, 16, 16, 16],
            'Q' => [14, 17, 17, 17, 21, 18, 13],
            'R' => [30, 17, 17, 30, 20, 18, 17],
            'S' => [15, 16, 16, 14, 1, 1, 30],
            'T' => [31, 4, 4, 4, 4, 4, 4],
            'U' => [17, 17, 17, 17, 17, 17, 14],
            'V' => [17, 17, 17, 17, 17, 10, 4],
            'W' => [17, 17, 17, 21, 21, 21, 10],
            'X' => [17, 17, 10, 4, 10, 17, 17],
            'Y' => [17, 17, 10, 4, 4, 4, 4],
            'Z' => [31, 1, 2, 4, 8, 16, 31],
            '0' => [14, 17, 19, 21, 25, 17, 14],
            '1' => [4, 12, 4, 4, 4, 4, 14],
            '2' => [14, 17, 1, 2, 4, 8, 31],
            '3' => [30, 1, 1, 14, 1, 1, 30],
            '4' => [2, 6, 10, 18, 31, 2, 2],
            '5' => [31, 16, 16, 30, 1, 1, 30],
            '6' => [14, 16, 16, 30, 17, 17, 14],
            '7' => [31, 1, 2, 4, 8, 8, 8],
            '8' => [14, 17, 17, 14, 17, 17, 14],
            '9' => [14, 17, 17, 15, 1, 1, 14],
            '-' => [0, 0, 0, 31, 0, 0, 0],
            ':' => [0, 4, 4, 0, 4, 4, 0],
            '.' => [0, 0, 0, 0, 0, 12, 12],
            '/' => [1, 2, 2, 4, 8, 8, 16],
            _ => [0; 7],
        }
    }

    fn draw_text(
        frame: &mut [[u8; 3]],
        x: isize,
        y: isize,
        text: &str,
        scale: isize,
        color: [u8; 3],
    ) {
        let mut cursor = x;
        for character in text.chars() {
            let rows = glyph(character.to_ascii_uppercase());
            for (row, bits) in rows.into_iter().enumerate() {
                for column in 0..5 {
                    if bits & (1 << (4 - column)) != 0 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                put_pixel(
                                    frame,
                                    cursor + column * scale + dx,
                                    y + row as isize * scale + dy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            cursor += 6 * scale;
        }
    }

    fn affine_matrix(ellipse: Ellipse, amount: f64) -> [[f64; 2]; 2] {
        let ratio = ellipse.minor_radius / ellipse.major_radius.max(1.0);
        let (sine, cosine) = (ellipse.angle * amount).sin_cos();
        let vertical = 1.0 + (ratio - 1.0) * amount;
        [[cosine, -sine * vertical], [sine, cosine * vertical]]
    }

    fn invert_matrix(matrix: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        [
            [matrix[1][1] / determinant, -matrix[0][1] / determinant],
            [-matrix[1][0] / determinant, matrix[0][0] / determinant],
        ]
    }

    fn transformed_point(
        point: (f64, f64),
        center: (f64, f64),
        inverse: [[f64; 2]; 2],
    ) -> (f64, f64) {
        let dx = point.0 - center.0;
        let dy = point.1 - center.1;
        (
            center.0 + inverse[0][0] * dx + inverse[0][1] * dy,
            center.1 + inverse[1][0] * dx + inverse[1][1] * dy,
        )
    }

    fn preview_sample(preview: &[[u8; 3]], width: usize, height: usize, x: f64, y: f64) -> [u8; 3] {
        if x < 0.0 || y < 0.0 || x >= width as f64 || y >= height as f64 {
            return [8, 9, 12];
        }
        preview[y.floor() as usize * width + x.floor() as usize]
    }

    fn render_rejection_review_video(path: &Path, inputs: &[LoadedRaw]) -> Result<(), String> {
        if inputs.is_empty() {
            return Err("rejected review requires at least one RAW10 frame".to_string());
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let mut reviews = Vec::with_capacity(inputs.len());
        for input in inputs {
            let width = input.frame.width;
            let height = input.frame.height;
            if width * PANEL_SCALE * 2 != OUTPUT_WIDTH
                || PANEL_TOP + height * PANEL_SCALE > OUTPUT_HEIGHT
            {
                return Err(format!(
                    "rejected review expects 384x256 RAW10, got {width}x{height} for sequence {}",
                    input.frame.sequence
                ));
            }
            let blank_color = vec![[0u8; 3]; input.frame.pixels.len()];
            reviews.push((
                input.frame.sequence,
                clean_raw_display_preview(&input.frame.pixels, &blank_color, width, height),
                width,
                height,
            ));
        }

        let geometry = format!("{}x{}", OUTPUT_WIDTH, OUTPUT_HEIGHT);
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &geometry,
                "-framerate",
                &OUTPUT_FPS.to_string(),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "rgb24",
                path.to_str().ok_or("rejected-review path is not UTF-8")?,
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start ffmpeg: {error}"))?;
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;

        for (review_index, (sequence, preview, width, height)) in reviews.iter().enumerate() {
            let center = (*width as f64 * 0.5, *height as f64 * 0.5);
            for local_frame in 0..FRAMES_PER_REJECTION {
                let local = local_frame as f64 / FRAMES_PER_REJECTION as f64;
                let phase = std::f64::consts::TAU * local;
                // Orthographic projection of a gently yawing and pitching ROI
                // plane. The affine inverse below samples only the native RAW
                // preview; it does not manufacture or resize inference input.
                let yaw = 0.68 * phase.sin();
                let pitch = 0.22 * (phase + 0.7).sin();
                let matrix = [
                    [yaw.cos(), 0.14 * phase.cos()],
                    [0.07 * phase.sin(), pitch.cos()],
                ];
                let inverse = invert_matrix(matrix);
                let mut frame = vec![[7u8, 9u8, 13u8]; OUTPUT_WIDTH * OUTPUT_HEIGHT];
                draw_text(
                    &mut frame,
                    18,
                    16,
                    &format!("RAW10 REJECTED  SEQUENCE {sequence}"),
                    3,
                    [240, 240, 245],
                );
                draw_text(
                    &mut frame,
                    790,
                    16,
                    "ANIMATED AFFINE INSPECTION",
                    3,
                    [60, 235, 255],
                );

                for output_y in 0..*height * PANEL_SCALE {
                    for output_x in 0..*width * PANEL_SCALE {
                        let native_x = output_x as f64 / PANEL_SCALE as f64;
                        let native_y = output_y as f64 / PANEL_SCALE as f64;
                        let screen_y = PANEL_TOP + output_y;
                        frame[screen_y * OUTPUT_WIDTH + output_x] =
                            preview_sample(preview, *width, *height, native_x, native_y);

                        let source = transformed_point((native_x, native_y), center, inverse);
                        let right_x = *width * PANEL_SCALE + output_x;
                        frame[screen_y * OUTPUT_WIDTH + right_x] =
                            preview_sample(preview, *width, *height, source.0, source.1);
                    }
                }

                let display_point = |point: (f64, f64)| {
                    let point = transformed_point(point, center, matrix);
                    (
                        (*width as f64 + point.0) * PANEL_SCALE as f64,
                        PANEL_TOP as f64 + point.1 * PANEL_SCALE as f64,
                    )
                };
                let corners = [
                    (0.0, 0.0),
                    (*width as f64 - 1.0, 0.0),
                    (*width as f64 - 1.0, *height as f64 - 1.0),
                    (0.0, *height as f64 - 1.0),
                    (0.0, 0.0),
                ];
                for pair in corners.windows(2) {
                    draw_line(
                        &mut frame,
                        display_point(pair[0]),
                        display_point(pair[1]),
                        [0, 225, 255],
                        1,
                    );
                }
                let sweep_x = (*width as f64 - 1.0) * (0.5 + 0.46 * phase.sin());
                draw_line(
                    &mut frame,
                    display_point((sweep_x, 0.0)),
                    display_point((sweep_x, *height as f64 - 1.0)),
                    [0, 235, 255],
                    1,
                );
                draw_text(
                    &mut frame,
                    18,
                    594,
                    "NO ELLIPSE DRAWN  OUTER SAM ANCHOR FAILED ITS GEOMETRY GATE",
                    2,
                    [235, 120, 170],
                );
                draw_text(
                    &mut frame,
                    1280,
                    594,
                    &format!("{}/{}", review_index + 1, reviews.len()),
                    2,
                    [210, 220, 230],
                );
                let bytes = unsafe {
                    std::slice::from_raw_parts(frame.as_ptr() as *const u8, frame.len() * 3)
                };
                stdin
                    .write_all(bytes)
                    .map_err(|error| format!("write rejected-review frame: {error}"))?;
            }
        }
        drop(stdin);
        let status = child
            .wait()
            .map_err(|error| format!("wait for ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("ffmpeg exited with {status}"));
        }
        Ok(())
    }

    fn render_rejected_sam_review_video(
        path: &Path,
        reviews: &[RejectedSamMaskReview],
    ) -> Result<(), String> {
        if reviews.is_empty() {
            return Err("rejected SAM review requires at least one result".to_string());
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let geometry = format!("{}x{}", OUTPUT_WIDTH, OUTPUT_HEIGHT);
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &geometry,
                "-framerate",
                &OUTPUT_FPS.to_string(),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "rgb24",
                path.to_str()
                    .ok_or("rejected SAM review path is not UTF-8")?,
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start ffmpeg: {error}"))?;
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;

        for (review_index, review) in reviews.iter().enumerate() {
            if review.width * PANEL_SCALE * 2 != OUTPUT_WIDTH
                || PANEL_TOP + review.height * PANEL_SCALE > OUTPUT_HEIGHT
            {
                return Err(format!(
                    "rejected SAM review expects 384x256 RAW10, got {}x{} for sequence {}",
                    review.width, review.height, review.sequence
                ));
            }
            let mut boundary = vec![0u8; review.mask_width * review.mask_height];
            for &index in review.mask.boundary_pixels.iter() {
                if let Some(pixel) = boundary.get_mut(index as usize) {
                    *pixel = 1;
                }
            }
            let grid_index = |x: f64, y: f64| {
                if !x.is_finite()
                    || !y.is_finite()
                    || x < 0.0
                    || y < 0.0
                    || x >= review.width as f64
                    || y >= review.height as f64
                    || review.mask_width == 0
                    || review.mask_height == 0
                {
                    return None;
                }
                let low_x = ((x + 0.5) * review.mask_width as f64 / review.width as f64 - 0.5)
                    .round()
                    .clamp(0.0, review.mask_width.saturating_sub(1) as f64)
                    as usize;
                let low_y = ((y + 0.5) * review.mask_height as f64 / review.height as f64 - 0.5)
                    .round()
                    .clamp(0.0, review.mask_height.saturating_sub(1) as f64)
                    as usize;
                Some(low_y * review.mask_width + low_x)
            };
            let center = (review.width as f64 * 0.5, review.height as f64 * 0.5);
            for local_frame in 0..FRAMES_PER_REJECTION {
                let local = local_frame as f64 / FRAMES_PER_REJECTION as f64;
                let phase = std::f64::consts::TAU * local;
                let yaw = 0.68 * phase.sin();
                let pitch = 0.22 * (phase + 0.7).sin();
                let matrix = [
                    [yaw.cos(), 0.14 * phase.cos()],
                    [0.07 * phase.sin(), pitch.cos()],
                ];
                let inverse = invert_matrix(matrix);
                let mask_alpha = 0.23 + 0.09 * (0.5 + 0.5 * phase.sin());
                let paint_mask = |mut pixel: [u8; 3], x: f64, y: f64| {
                    let Some(index) = grid_index(x, y) else {
                        return pixel;
                    };
                    if boundary.get(index).copied().unwrap_or_default() != 0 {
                        return [255, 225, 30];
                    }
                    if review.mask.pixels.get(index).copied().unwrap_or_default() != 0 {
                        let color = [0u8, 220u8, 255u8];
                        for channel in 0..3 {
                            pixel[channel] = (pixel[channel] as f64 * (1.0 - mask_alpha)
                                + color[channel] as f64 * mask_alpha)
                                .round()
                                .clamp(0.0, 255.0)
                                as u8;
                        }
                    }
                    pixel
                };
                let mut frame = vec![[7u8, 9u8, 13u8]; OUTPUT_WIDTH * OUTPUT_HEIGHT];
                draw_text(
                    &mut frame,
                    18,
                    16,
                    &format!("RAW10 SEQUENCE {}  PRE-GATE MASK", review.sequence),
                    3,
                    [240, 240, 245],
                );
                draw_text(
                    &mut frame,
                    790,
                    16,
                    "SAM3 INITIAL OUTER IRIS MASK",
                    3,
                    [60, 235, 255],
                );

                for output_y in 0..review.height * PANEL_SCALE {
                    for output_x in 0..review.width * PANEL_SCALE {
                        let native_x = output_x as f64 / PANEL_SCALE as f64;
                        let native_y = output_y as f64 / PANEL_SCALE as f64;
                        let screen_y = PANEL_TOP + output_y;
                        let raw_pixel = preview_sample(
                            &review.preview,
                            review.width,
                            review.height,
                            native_x,
                            native_y,
                        );
                        frame[screen_y * OUTPUT_WIDTH + output_x] =
                            paint_mask(raw_pixel, native_x, native_y);

                        let source = transformed_point((native_x, native_y), center, inverse);
                        let raw_pixel = preview_sample(
                            &review.preview,
                            review.width,
                            review.height,
                            source.0,
                            source.1,
                        );
                        let right_x = review.width * PANEL_SCALE + output_x;
                        frame[screen_y * OUTPUT_WIDTH + right_x] =
                            paint_mask(raw_pixel, source.0, source.1);
                    }
                }

                let display_point = |point: (f64, f64)| {
                    let point = transformed_point(point, center, matrix);
                    (
                        (review.width as f64 + point.0) * PANEL_SCALE as f64,
                        PANEL_TOP as f64 + point.1 * PANEL_SCALE as f64,
                    )
                };
                let corners = [
                    (0.0, 0.0),
                    (review.width as f64 - 1.0, 0.0),
                    (review.width as f64 - 1.0, review.height as f64 - 1.0),
                    (0.0, review.height as f64 - 1.0),
                    (0.0, 0.0),
                ];
                for pair in corners.windows(2) {
                    draw_line(
                        &mut frame,
                        display_point(pair[0]),
                        display_point(pair[1]),
                        [0, 225, 255],
                        1,
                    );
                }
                let gate = if review.geometry_accepted {
                    "GEOMETRY GATE ACCEPTED ON RERUN"
                } else {
                    "GEOMETRY GATE REJECTED"
                };
                let gate_color = if review.geometry_accepted {
                    [80, 255, 130]
                } else {
                    [245, 100, 165]
                };
                draw_text(
                    &mut frame,
                    18,
                    594,
                    &format!(
                        "CYAN MASK  YELLOW BOUNDARY  {}  Q{}  SCORE {:.3}",
                        gate, review.mask.query, review.mask.score
                    ),
                    2,
                    gate_color,
                );
                draw_text(
                    &mut frame,
                    1240,
                    594,
                    &format!(
                        "{}/{}  {}MS",
                        review_index + 1,
                        reviews.len(),
                        review.elapsed_ms
                    ),
                    2,
                    [210, 220, 230],
                );
                let bytes = unsafe {
                    std::slice::from_raw_parts(frame.as_ptr() as *const u8, frame.len() * 3)
                };
                stdin
                    .write_all(bytes)
                    .map_err(|error| format!("write rejected SAM review frame: {error}"))?;
            }
        }
        drop(stdin);
        let status = child
            .wait()
            .map_err(|error| format!("wait for ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("ffmpeg exited with {status}"));
        }
        Ok(())
    }

    fn put_pixel_dynamic(
        frame: &mut [[u8; 3]],
        width: usize,
        height: usize,
        x: isize,
        y: isize,
        color: [u8; 3],
    ) {
        if x >= 0 && y >= 0 && x < width as isize && y < height as isize {
            frame[y as usize * width + x as usize] = color;
        }
    }

    fn draw_disc_dynamic(
        frame: &mut [[u8; 3]],
        width: usize,
        height: usize,
        x: f64,
        y: f64,
        radius: isize,
        color: [u8; 3],
    ) {
        let center_x = x.round() as isize;
        let center_y = y.round() as isize;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    put_pixel_dynamic(frame, width, height, center_x + dx, center_y + dy, color);
                }
            }
        }
    }

    fn draw_line_dynamic(
        frame: &mut [[u8; 3]],
        width: usize,
        height: usize,
        first: (f64, f64),
        second: (f64, f64),
        color: [u8; 3],
        thickness: isize,
    ) {
        let steps = ((second.0 - first.0).abs().max((second.1 - first.1).abs()) * 1.5)
            .ceil()
            .max(1.0) as usize;
        for step in 0..=steps {
            let amount = step as f64 / steps as f64;
            draw_disc_dynamic(
                frame,
                width,
                height,
                first.0 + (second.0 - first.0) * amount,
                first.1 + (second.1 - first.1) * amount,
                thickness,
                color,
            );
        }
    }

    fn fill_rect_dynamic(
        frame: &mut [[u8; 3]],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        rectangle_width: usize,
        rectangle_height: usize,
        color: [u8; 3],
    ) {
        let end_x = x.saturating_add(rectangle_width).min(width);
        let end_y = y.saturating_add(rectangle_height).min(height);
        for yy in y.min(height)..end_y {
            frame[yy * width + x.min(width)..yy * width + end_x].fill(color);
        }
    }

    fn draw_text_dynamic(
        frame: &mut [[u8; 3]],
        width: usize,
        height: usize,
        x: isize,
        y: isize,
        text: &str,
        scale: isize,
        color: [u8; 3],
    ) {
        let mut cursor = x;
        for character in text.chars() {
            let rows = glyph(character.to_ascii_uppercase());
            for (row, bits) in rows.into_iter().enumerate() {
                for column in 0..5 {
                    if bits & (1 << (4 - column)) != 0 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                put_pixel_dynamic(
                                    frame,
                                    width,
                                    height,
                                    cursor + column * scale + dx,
                                    y + row as isize * scale + dy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            cursor += 6 * scale;
        }
    }

    fn abbreviate(text: &str, maximum: usize) -> String {
        let mut result = text.chars().take(maximum).collect::<String>();
        if text.chars().count() > maximum {
            result.push_str("...");
        }
        result
    }

    fn prompt_lab_mask_pixel(
        candidate: &PromptLabCandidate,
        x: f64,
        y: f64,
        source_width: usize,
        source_height: usize,
    ) -> (bool, bool) {
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x >= source_width as f64
            || y >= source_height as f64
            || candidate.mask_width == 0
            || candidate.mask_height == 0
        {
            return (false, false);
        }
        let low_x = ((x + 0.5) * candidate.mask_width as f64 / source_width as f64 - 0.5)
            .round()
            .clamp(0.0, candidate.mask_width.saturating_sub(1) as f64) as usize;
        let low_y = ((y + 0.5) * candidate.mask_height as f64 / source_height as f64 - 0.5)
            .round()
            .clamp(0.0, candidate.mask_height.saturating_sub(1) as f64)
            as usize;
        let index = low_y * candidate.mask_width + low_x;
        if candidate
            .mask
            .pixels
            .get(index)
            .copied()
            .unwrap_or_default()
            == 0
        {
            return (false, false);
        }
        let boundary = low_x == 0
            || low_y == 0
            || low_x + 1 == candidate.mask_width
            || low_y + 1 == candidate.mask_height
            || candidate.mask.pixels[index - 1] == 0
            || candidate.mask.pixels[index + 1] == 0
            || candidate.mask.pixels[index - candidate.mask_width] == 0
            || candidate.mask.pixels[index + candidate.mask_width] == 0;
        (true, boundary)
    }

    fn prompt_lab_overlay(
        mut pixel: [u8; 3],
        candidate: Option<&PromptLabCandidate>,
        x: f64,
        y: f64,
        source_width: usize,
        source_height: usize,
        alpha: f64,
        show_boundary: bool,
    ) -> [u8; 3] {
        let Some(candidate) = candidate else {
            return pixel;
        };
        let (inside, boundary) =
            prompt_lab_mask_pixel(candidate, x, y, source_width, source_height);
        if boundary && show_boundary {
            return if candidate.evidence_only {
                [255, 55, 210]
            } else {
                [255, 225, 30]
            };
        }
        if inside {
            let color = if candidate.evidence_only {
                [190u8, 55u8, 255u8]
            } else {
                [0u8, 220u8, 255u8]
            };
            for channel in 0..3 {
                pixel[channel] = (pixel[channel] as f64 * (1.0 - alpha)
                    + color[channel] as f64 * alpha)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
        pixel
    }

    fn prompt_lab_conditioning_pixel(
        row: &PromptLabRow,
        step_index: usize,
        source_pixel: [u8; 3],
        x: f64,
        y: f64,
    ) -> [u8; 3] {
        if step_index == 0 {
            return source_pixel;
        }
        if let Some(preview) = row.conditioning_preview.as_ref() {
            return preview_sample(preview, row.width, row.height, x, y);
        }
        source_pixel
    }

    fn isolated_pupil_preview(
        preview: &[[u8; 3]],
        width: usize,
        height: usize,
        candidate: &PromptLabCandidate,
    ) -> Vec<[u8; 3]> {
        let mut result = vec![[255, 0, 255]; preview.len()];
        for y in 0..height {
            for x in 0..width {
                if prompt_lab_mask_pixel(candidate, x as f64, y as f64, width, height).0 {
                    result[y * width + x] = preview[y * width + x];
                }
            }
        }
        result
    }

    fn inner_occluded_pupil_preview(
        preview: &[[u8; 3]],
        width: usize,
        height: usize,
        candidate: &PromptLabCandidate,
        inset: usize,
    ) -> Vec<[u8; 3]> {
        let mut native = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                native[y * width + x] =
                    prompt_lab_mask_pixel(candidate, x as f64, y as f64, width, height).0 as u8;
            }
        }
        let mut smooth = vec![0u8; native.len()];
        for y in 0..height {
            for x in 0..width {
                let mut occupied = 0usize;
                let mut samples = 0usize;
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        let xx = x as isize + dx;
                        let yy = y as isize + dy;
                        if xx < 0 || yy < 0 || xx >= width as isize || yy >= height as isize {
                            continue;
                        }
                        samples += 1;
                        occupied += (native[yy as usize * width + xx as usize] != 0) as usize;
                    }
                }
                smooth[y * width + x] = (occupied * 2 >= samples) as u8;
            }
        }
        let inset = inset as isize;
        let mut result = preview.to_vec();
        for y in inset..height as isize - inset {
            for x in inset..width as isize - inset {
                let covered = (-inset..=inset).all(|dy| {
                    (-inset..=inset).all(|dx| {
                        dx * dx + dy * dy > inset * inset
                            || smooth[(y + dy) as usize * width + (x + dx) as usize] != 0
                    })
                });
                if covered {
                    result[y as usize * width + x as usize] = [255, 0, 255];
                }
            }
        }
        result
    }

    fn write_rgb_png(
        path: &Path,
        pixels: &[[u8; 3]],
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        let geometry = format!("{width}x{height}");
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &geometry,
                "-i",
                "-",
                "-frames:v",
                "1",
                "-compression_level",
                "4",
                path.to_str().ok_or("contact-sheet path is not UTF-8")?,
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start contact-sheet ffmpeg: {error}"))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or("contact-sheet ffmpeg has no stdin")?;
        let bytes =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 3) };
        stdin
            .write_all(bytes)
            .map_err(|error| format!("write contact sheet: {error}"))?;
        drop(stdin);
        let status = child
            .wait()
            .map_err(|error| format!("wait for contact-sheet ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("contact-sheet ffmpeg exited with {status}"));
        }
        Ok(())
    }

    fn render_prompt_lab_contact_sheet(
        path: &Path,
        prompts: &[String],
        rows: &[PromptLabRow],
    ) -> Result<(), String> {
        let first = rows.first().ok_or("prompt lab has no rows")?;
        let cell_width = first.width;
        let cell_height = first.height;
        if rows
            .iter()
            .any(|row| row.width != cell_width || row.height != cell_height)
        {
            return Err("prompt-lab contact sheet requires equal RAW10 dimensions".to_string());
        }
        let header_height = 52usize;
        let candidates_per_prompt = PROMPT_LAB_DISPLAY_CANDIDATES;
        let width = cell_width * (prompts.len() * candidates_per_prompt + 1);
        let height = header_height + cell_height * rows.len();
        let mut sheet = vec![[7u8, 9u8, 13u8]; width * height];
        draw_text_dynamic(
            &mut sheet,
            width,
            height,
            10,
            8,
            "RAW10 REFERENCE",
            2,
            [235, 240, 245],
        );
        for (step_index, prompt) in prompts.iter().enumerate() {
            for candidate_index in 0..candidates_per_prompt {
                let column = 1 + step_index * candidates_per_prompt + candidate_index;
                let x = column * cell_width + 10;
                draw_text_dynamic(
                    &mut sheet,
                    width,
                    height,
                    x as isize,
                    5,
                    &format!("STEP {}  CAND {}", step_index + 1, candidate_index + 1),
                    2,
                    [40, 235, 255],
                );
                draw_text_dynamic(
                    &mut sheet,
                    width,
                    height,
                    x as isize,
                    29,
                    &abbreviate(prompt, 58),
                    1,
                    [205, 215, 225],
                );
            }
        }
        for (row_index, row) in rows.iter().enumerate() {
            let top = header_height + row_index * cell_height;
            for y in 0..cell_height {
                for x in 0..cell_width {
                    let source = row.preview[y * cell_width + x];
                    sheet[(top + y) * width + x] = source;
                    for (step_index, step) in row.steps.iter().enumerate() {
                        for candidate_index in 0..candidates_per_prompt {
                            let column = 1 + step_index * candidates_per_prompt + candidate_index;
                            let conditioned = prompt_lab_conditioning_pixel(
                                row, step_index, source, x as f64, y as f64,
                            );
                            sheet[(top + y) * width + column * cell_width + x] = prompt_lab_overlay(
                                conditioned,
                                step.candidates.get(candidate_index),
                                x as f64,
                                y as f64,
                                cell_width,
                                cell_height,
                                0.31,
                                step_index + 1 == row.steps.len(),
                            );
                        }
                    }
                }
            }
            fill_rect_dynamic(&mut sheet, width, height, 0, top, 126, 22, [5, 7, 10]);
            draw_text_dynamic(
                &mut sheet,
                width,
                height,
                7,
                top as isize + 5,
                &format!("SEQ {}", row.sequence),
                2,
                [240, 240, 245],
            );
            for (step_index, step) in row.steps.iter().enumerate() {
                for candidate_index in 0..candidates_per_prompt {
                    let column = 1 + step_index * candidates_per_prompt + candidate_index;
                    let x = column * cell_width;
                    fill_rect_dynamic(
                        &mut sheet,
                        width,
                        height,
                        x,
                        top + cell_height - 19,
                        250,
                        19,
                        [5, 7, 10],
                    );
                    let statistics = step
                        .candidates
                        .get(candidate_index)
                        .map(|candidate| {
                            format!(
                                "Q{} {} AREA {:.4} SEG {:.3} CRUST {:.3}",
                                candidate.query,
                                if candidate.evidence_only {
                                    "EVID"
                                } else {
                                    "DISK"
                                },
                                candidate.area_fraction,
                                candidate.segment_roundness,
                                candidate.crust_score,
                            )
                        })
                        .unwrap_or_else(|| "NO NONEMPTY CANDIDATE".to_string());
                    draw_text_dynamic(
                        &mut sheet,
                        width,
                        height,
                        x as isize + 6,
                        (top + cell_height - 16) as isize,
                        &statistics,
                        1,
                        [215, 220, 230],
                    );
                }
            }
        }
        write_rgb_png(path, &sheet, width, height)
    }

    /// A source-major, prompt-minor matrix for comparing candidate rank across
    /// the entire corpus: columns are (source, prompt) and rows are descending
    /// eligible candidate rank. Bands are streamed directly to ffmpeg so the
    /// very wide native-resolution sheet never exists as duplicate giant
    /// frame buffers in host memory.
    fn render_prompt_lab_comparison_matrix(
        path: &Path,
        prompts: &[String],
        rows: &[PromptLabRow],
    ) -> Result<(), String> {
        let first = rows.first().ok_or("prompt lab has no rows")?;
        if prompts.is_empty() {
            return Err("prompt lab has no prompts".to_string());
        }
        if rows.iter().any(|row| {
            row.width != first.width
                || row.height != first.height
                || row.steps.len() != prompts.len()
        }) {
            return Err(
                "prompt-lab comparison matrix requires equal RAW10 dimensions and prompts"
                    .to_string(),
            );
        }
        let gutter = 192usize;
        let header_height = 72usize;
        let column_count = rows.len() * prompts.len();
        let rank_count = rows
            .iter()
            .flat_map(|row| row.steps.iter())
            .map(|step| {
                step.all_candidates
                    .iter()
                    .filter(|candidate| candidate.eligible)
                    .count()
            })
            .max()
            .unwrap_or_default();
        if rank_count == 0 {
            return Err("comparison matrix has no candidates in the 5%-50% band".to_string());
        }
        let width = gutter + column_count * first.width;
        let height = header_height + rank_count * first.height;
        let geometry = format!("{width}x{height}");
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &geometry,
                "-i",
                "-",
                "-frames:v",
                "1",
                "-compression_level",
                "4",
                path.to_str().ok_or("comparison matrix path is not UTF-8")?,
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start comparison-matrix ffmpeg: {error}"))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or("comparison-matrix ffmpeg has no stdin")?;

        let mut header = vec![[7u8, 9u8, 13u8]; width * header_height];
        draw_text_dynamic(
            &mut header,
            width,
            header_height,
            8,
            8,
            "RANK",
            2,
            [235, 240, 245],
        );
        draw_text_dynamic(
            &mut header,
            width,
            header_height,
            8,
            36,
            "HIGH TO LOW",
            1,
            [205, 215, 225],
        );
        for (row_index, row) in rows.iter().enumerate() {
            for (prompt_index, prompt) in prompts.iter().enumerate() {
                let column = row_index * prompts.len() + prompt_index;
                let x = gutter + column * first.width;
                let accent = if prompt_index == 0 {
                    [255, 180, 35]
                } else {
                    [40, 235, 255]
                };
                fill_rect_dynamic(
                    &mut header,
                    width,
                    header_height,
                    x,
                    0,
                    3,
                    header_height,
                    accent,
                );
                draw_text_dynamic(
                    &mut header,
                    width,
                    header_height,
                    x as isize + 8,
                    5,
                    &format!(
                        "IMG {:02}/{} SEQ {}  P{}",
                        row_index + 1,
                        rows.len(),
                        row.sequence,
                        prompt_index + 1
                    ),
                    1,
                    accent,
                );
                draw_text_dynamic(
                    &mut header,
                    width,
                    header_height,
                    x as isize + 8,
                    25,
                    &abbreviate(prompt, 55),
                    1,
                    [215, 220, 230],
                );
                draw_text_dynamic(
                    &mut header,
                    width,
                    header_height,
                    x as isize + 8,
                    45,
                    "AREA 5%-50%  CRUST SCORE DESC",
                    1,
                    [165, 175, 188],
                );
            }
        }
        let header_bytes =
            unsafe { std::slice::from_raw_parts(header.as_ptr() as *const u8, header.len() * 3) };
        stdin
            .write_all(header_bytes)
            .map_err(|error| format!("write comparison-matrix header: {error}"))?;

        for rank in 0..rank_count {
            let mut band = vec![[7u8, 9u8, 13u8]; width * first.height];
            draw_text_dynamic(
                &mut band,
                width,
                first.height,
                12,
                16,
                &format!("CANDIDATE RANK {:03}", rank + 1),
                2,
                [235, 240, 245],
            );
            draw_text_dynamic(
                &mut band,
                width,
                first.height,
                12,
                48,
                "YELLOW EDGE",
                1,
                [255, 225, 30],
            );
            draw_text_dynamic(
                &mut band,
                width,
                first.height,
                12,
                65,
                "CYAN MASK",
                1,
                [40, 235, 255],
            );
            for (row_index, row) in rows.iter().enumerate() {
                for (prompt_index, step) in row.steps.iter().enumerate() {
                    let column = row_index * prompts.len() + prompt_index;
                    let left = gutter + column * first.width;
                    let candidate = step
                        .all_candidates
                        .iter()
                        .filter(|candidate| candidate.eligible)
                        .nth(rank);
                    if let Some(candidate) = candidate {
                        for y in 0..first.height {
                            for x in 0..first.width {
                                band[y * width + left + x] = prompt_lab_overlay(
                                    row.preview[y * first.width + x],
                                    Some(candidate),
                                    x as f64,
                                    y as f64,
                                    first.width,
                                    first.height,
                                    0.31,
                                    true,
                                );
                            }
                        }
                        fill_rect_dynamic(
                            &mut band,
                            width,
                            first.height,
                            left,
                            first.height - 20,
                            first.width,
                            20,
                            [5, 7, 10],
                        );
                        draw_text_dynamic(
                            &mut band,
                            width,
                            first.height,
                            left as isize + 6,
                            (first.height - 17) as isize,
                            &format!(
                                "Q{} AREA {:.1}% SEG {:.3} CRUST {:.3}",
                                candidate.query,
                                candidate.area_fraction * 100.0,
                                candidate.segment_roundness,
                                candidate.crust_score
                            ),
                            1,
                            [220, 225, 235],
                        );
                    } else {
                        fill_rect_dynamic(
                            &mut band,
                            width,
                            first.height,
                            left,
                            0,
                            first.width,
                            first.height,
                            [12, 14, 18],
                        );
                        draw_text_dynamic(
                            &mut band,
                            width,
                            first.height,
                            left as isize + 10,
                            16,
                            "NO CANDIDATE AT THIS RANK",
                            1,
                            [110, 118, 128],
                        );
                    }
                    let accent = if prompt_index == 0 {
                        [255, 180, 35]
                    } else {
                        [40, 235, 255]
                    };
                    fill_rect_dynamic(
                        &mut band,
                        width,
                        first.height,
                        left,
                        0,
                        3,
                        first.height,
                        accent,
                    );
                }
            }
            let band_bytes =
                unsafe { std::slice::from_raw_parts(band.as_ptr() as *const u8, band.len() * 3) };
            stdin
                .write_all(band_bytes)
                .map_err(|error| format!("write comparison-matrix rank {}: {error}", rank + 1))?;
        }
        drop(stdin);
        let status = child
            .wait()
            .map_err(|error| format!("wait for comparison-matrix ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("comparison-matrix ffmpeg exited with {status}"));
        }
        Ok(())
    }

    fn render_prompt_lab_affine_review(
        path: &Path,
        prompts: &[String],
        rows: &[PromptLabRow],
    ) -> Result<(), String> {
        let first = rows.first().ok_or("prompt lab has no rows")?;
        let pupil_three_step = std::env::var_os("BUTTERCUP_SAM31_PUPIL_THREE_STEP").is_some();
        if pupil_three_step && prompts.len() != 2 {
            return Err(
                "three-step pupil review requires exactly two semantic prompts".to_string(),
            );
        }
        let header_height = 64usize;
        let width = first.width * (prompts.len() + 1);
        let height = header_height + first.height;
        let geometry = format!("{width}x{height}");
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &geometry,
                "-framerate",
                &OUTPUT_FPS.to_string(),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "rgb24",
                path.to_str().ok_or("prompt-lab review path is not UTF-8")?,
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start prompt-lab review ffmpeg: {error}"))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or("prompt-lab review ffmpeg has no stdin")?;
        for (row_index, row) in rows.iter().enumerate() {
            let center = (row.width as f64 * 0.5, row.height as f64 * 0.5);
            for local_frame in 0..FRAMES_PER_REJECTION {
                let candidate_index = (local_frame * PROMPT_LAB_DISPLAY_CANDIDATES
                    / FRAMES_PER_REJECTION)
                    .min(PROMPT_LAB_DISPLAY_CANDIDATES - 1);
                let candidate_frames = FRAMES_PER_REJECTION / PROMPT_LAB_DISPLAY_CANDIDATES;
                let local = (local_frame % candidate_frames) as f64 / candidate_frames as f64;
                let phase = std::f64::consts::TAU * local;
                let matrix = [
                    [(0.62 * phase.sin()).cos(), 0.11 * phase.cos()],
                    [0.055 * phase.sin(), (0.18 * (phase + 0.7).sin()).cos()],
                ];
                let inverse = invert_matrix(matrix);
                let mut frame = vec![[7u8, 9u8, 13u8]; width * height];
                if pupil_three_step {
                    let headings = [
                        "STEP 1  SEGMENT PUPIL",
                        "STEP 2  MASK MIDDLE",
                        "STEP 3  RE-SEGMENT PUPIL",
                    ];
                    let details = [
                        prompts[0].as_str(),
                        "SMOOTH + INSET 5 RAW PIXELS + SOLID HOT PINK",
                        prompts[1].as_str(),
                    ];
                    for index in 0..3 {
                        let x = index * row.width + 8;
                        draw_text_dynamic(
                            &mut frame,
                            width,
                            height,
                            x as isize,
                            5,
                            headings[index],
                            2,
                            [40, 235, 255],
                        );
                        draw_text_dynamic(
                            &mut frame,
                            width,
                            height,
                            x as isize,
                            29,
                            &abbreviate(details[index], 58),
                            1,
                            [205, 215, 225],
                        );
                    }
                } else {
                    draw_text_dynamic(
                        &mut frame,
                        width,
                        height,
                        8,
                        8,
                        &format!(
                            "RAW10  SEQ {}  {}/{}",
                            row.sequence,
                            row_index + 1,
                            rows.len()
                        ),
                        2,
                        [235, 240, 245],
                    );
                    for (index, prompt) in prompts.iter().enumerate() {
                        let x = (index + 1) * row.width + 8;
                        let candidate = row.steps[index].candidates.get(candidate_index);
                        draw_text_dynamic(
                            &mut frame,
                            width,
                            height,
                            x as isize,
                            5,
                            &format!(
                                "STEP {} CAND {} Q{}",
                                index + 1,
                                candidate_index + 1,
                                candidate
                                    .map(|candidate| candidate.query.to_string())
                                    .unwrap_or_else(|| "NONE".to_string())
                            ),
                            2,
                            [40, 235, 255],
                        );
                        draw_text_dynamic(
                            &mut frame,
                            width,
                            height,
                            x as isize,
                            29,
                            &abbreviate(prompt, 58),
                            1,
                            [205, 215, 225],
                        );
                    }
                }
                for y in 0..row.height {
                    for x in 0..row.width {
                        let source = transformed_point((x as f64, y as f64), center, inverse);
                        let raw =
                            preview_sample(&row.preview, row.width, row.height, source.0, source.1);
                        if pupil_three_step {
                            let first_candidate = row.steps[0].candidates.get(candidate_index);
                            frame[(header_height + y) * width + x] = prompt_lab_overlay(
                                raw,
                                first_candidate,
                                source.0,
                                source.1,
                                row.width,
                                row.height,
                                0.28,
                                false,
                            );
                            let conditioned =
                                prompt_lab_conditioning_pixel(row, 1, raw, source.0, source.1);
                            frame[(header_height + y) * width + row.width + x] = conditioned;
                            let final_candidate = row.steps[1].candidates.get(candidate_index);
                            frame[(header_height + y) * width + 2 * row.width + x] =
                                prompt_lab_overlay(
                                    conditioned,
                                    final_candidate,
                                    source.0,
                                    source.1,
                                    row.width,
                                    row.height,
                                    0.28,
                                    true,
                                );
                        } else {
                            frame[(header_height + y) * width + x] = row.preview[y * row.width + x];
                            for (step_index, step) in row.steps.iter().enumerate() {
                                let candidate = step.candidates.get(candidate_index);
                                let raw = prompt_lab_conditioning_pixel(
                                    row, step_index, raw, source.0, source.1,
                                );
                                frame[(header_height + y) * width
                                    + (step_index + 1) * row.width
                                    + x] = prompt_lab_overlay(
                                    raw,
                                    candidate,
                                    source.0,
                                    source.1,
                                    row.width,
                                    row.height,
                                    0.28,
                                    step_index + 1 == row.steps.len(),
                                );
                            }
                        }
                    }
                }
                // Only the final chain stage exposes geometric evidence.
                // Earlier panels deliberately remain mask-only so a rough
                // semantic region cannot be mistaken for the fitted pupil.
                if let Some(candidate) = row
                    .steps
                    .last()
                    .and_then(|step| step.candidates.get(candidate_index))
                {
                    let panel_left = row.steps.len() as f64 * row.width as f64;
                    let map_to_panel = |point: (f64, f64)| {
                        let transformed = transformed_point(point, center, matrix);
                        (
                            panel_left + transformed.0,
                            header_height as f64 + transformed.1,
                        )
                    };
                    if let Some(ellipse) = candidate.fitted {
                        let ellipse_points = ellipse.dense_points(241);
                        for pair in ellipse_points.windows(2) {
                            draw_line_dynamic(
                                &mut frame,
                                width,
                                height,
                                map_to_panel(pair[0]),
                                map_to_panel(pair[1]),
                                [35, 245, 255],
                                1,
                            );
                        }
                    }
                    let stride = (candidate.fit_points.len() / 48).max(1);
                    for &point in candidate.fit_points.iter().step_by(stride) {
                        let point = map_to_panel(point);
                        draw_disc_dynamic(
                            &mut frame,
                            width,
                            height,
                            point.0,
                            point.1,
                            2,
                            [255, 225, 30],
                        );
                    }
                }
                let bytes = unsafe {
                    std::slice::from_raw_parts(frame.as_ptr() as *const u8, frame.len() * 3)
                };
                stdin
                    .write_all(bytes)
                    .map_err(|error| format!("write prompt-lab review frame: {error}"))?;
            }
        }
        drop(stdin);
        let status = child
            .wait()
            .map_err(|error| format!("wait for prompt-lab review ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("prompt-lab review ffmpeg exited with {status}"));
        }
        Ok(())
    }

    /// Shows every candidate in the visible 5%-50% surface-area band in
    /// strict descending crust-score order. All 200 query slots remain in the
    /// JSON report even though out-of-band masks consume no review frames.
    fn render_prompt_lab_ranked_candidates_review(
        path: &Path,
        prompts: &[String],
        rows: &[PromptLabRow],
    ) -> Result<(), String> {
        const FRAMES_PER_CANDIDATE: usize = 5;
        let first = rows.first().ok_or("prompt lab has no rows")?;
        let header_height = 72usize;
        let width = first.width * (prompts.len() + 1);
        let height = header_height + first.height;
        let geometry = format!("{width}x{height}");
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &geometry,
                "-framerate",
                &OUTPUT_FPS.to_string(),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "rgb24",
                path.to_str()
                    .ok_or("all-candidates review path is not UTF-8")?,
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start all-candidates review ffmpeg: {error}"))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or("all-candidates review ffmpeg has no stdin")?;

        for (row_index, row) in rows.iter().enumerate() {
            let candidate_count = row
                .steps
                .iter()
                .map(|step| {
                    step.all_candidates
                        .iter()
                        .filter(|candidate| candidate.eligible)
                        .count()
                })
                .max()
                .unwrap_or_default();
            let center = (row.width as f64 * 0.5, row.height as f64 * 0.5);
            for rank in 0..candidate_count {
                for held_frame in 0..FRAMES_PER_CANDIDATE {
                    let sweep_frame = rank * FRAMES_PER_CANDIDATE + held_frame;
                    let phase = std::f64::consts::TAU * (sweep_frame % 120) as f64 / 120.0;
                    let matrix = [
                        [(0.62 * phase.sin()).cos(), 0.11 * phase.cos()],
                        [0.055 * phase.sin(), (0.18 * (phase + 0.7).sin()).cos()],
                    ];
                    let inverse = invert_matrix(matrix);
                    let mut frame = vec![[7u8, 9u8, 13u8]; width * height];
                    draw_text_dynamic(
                        &mut frame,
                        width,
                        height,
                        8,
                        8,
                        &format!(
                            "RAW10 SEQ {}  IMAGE {}/{}  AREA 5%-50% HIGH TO LOW",
                            row.sequence,
                            row_index + 1,
                            rows.len()
                        ),
                        2,
                        [235, 240, 245],
                    );
                    draw_text_dynamic(
                        &mut frame,
                        width,
                        height,
                        8,
                        36,
                        "YELLOW BOUNDARY  CYAN MASK  ALL 200 QUERY SLOTS RETAINED IN JSON",
                        1,
                        [205, 215, 225],
                    );
                    for (step_index, prompt) in prompts.iter().enumerate() {
                        let visible_count = row.steps[step_index]
                            .all_candidates
                            .iter()
                            .filter(|candidate| candidate.eligible)
                            .count();
                        let candidate = row.steps[step_index]
                            .all_candidates
                            .iter()
                            .filter(|candidate| candidate.eligible)
                            .nth(rank);
                        let panel_x = (step_index + 1) * row.width + 7;
                        let title_color = [40, 235, 255];
                        draw_text_dynamic(
                            &mut frame,
                            width,
                            height,
                            panel_x as isize,
                            4,
                            &format!(
                                "P{} RANK {:03}/{} Q{}",
                                step_index + 1,
                                rank + 1,
                                visible_count,
                                candidate
                                    .map(|candidate| candidate.query.to_string())
                                    .unwrap_or_else(|| "NONE".to_string())
                            ),
                            1,
                            title_color,
                        );
                        if let Some(candidate) = candidate {
                            draw_text_dynamic(
                                &mut frame,
                                width,
                                height,
                                panel_x as isize,
                                22,
                                &format!(
                                    "AREA {:.1}% SEG {:.3} CRUST {:.3}",
                                    candidate.area_fraction * 100.0,
                                    candidate.segment_roundness,
                                    candidate.crust_score,
                                ),
                                1,
                                title_color,
                            );
                        }
                        draw_text_dynamic(
                            &mut frame,
                            width,
                            height,
                            panel_x as isize,
                            40,
                            &abbreviate(prompt, 55),
                            1,
                            [205, 215, 225],
                        );
                    }
                    for y in 0..row.height {
                        for x in 0..row.width {
                            frame[(header_height + y) * width + x] = row.preview[y * row.width + x];
                            for (step_index, step) in row.steps.iter().enumerate() {
                                let candidate = step
                                    .all_candidates
                                    .iter()
                                    .filter(|candidate| candidate.eligible)
                                    .nth(rank);
                                let source =
                                    transformed_point((x as f64, y as f64), center, inverse);
                                let raw = preview_sample(
                                    &row.preview,
                                    row.width,
                                    row.height,
                                    source.0,
                                    source.1,
                                );
                                frame[(header_height + y) * width
                                    + (step_index + 1) * row.width
                                    + x] = prompt_lab_overlay(
                                    raw, candidate, source.0, source.1, row.width, row.height,
                                    0.28, true,
                                );
                            }
                        }
                    }
                    let bytes = unsafe {
                        std::slice::from_raw_parts(frame.as_ptr() as *const u8, frame.len() * 3)
                    };
                    stdin
                        .write_all(bytes)
                        .map_err(|error| format!("write all-candidates review frame: {error}"))?;
                }
            }
        }
        drop(stdin);
        let status = child
            .wait()
            .map_err(|error| format!("wait for all-candidates review ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("all-candidates review ffmpeg exited with {status}"));
        }
        Ok(())
    }

    fn render_review_video(
        path: &Path,
        preview: &[[u8; 3]],
        width: usize,
        height: usize,
        outer_fit: &sam31_outer::OuterMaskFitReview,
        fields: &[MaskField],
        evidence: &[ArcEvidence],
        strategies: &[Strategy],
    ) -> Result<(), String> {
        let geometry = format!("{}x{}", OUTPUT_WIDTH, OUTPUT_HEIGHT);
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &geometry,
                "-framerate",
                &OUTPUT_FPS.to_string(),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "rgb24",
                path.to_str().ok_or("review path is not UTF-8")?,
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start ffmpeg: {error}"))?;
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;
        let total_frames = strategies.len() * FRAMES_PER_STRATEGY;
        for frame_index in 0..total_frames {
            let strategy_index = (frame_index / FRAMES_PER_STRATEGY).min(strategies.len() - 1);
            let strategy = &strategies[strategy_index];
            let local = (frame_index % FRAMES_PER_STRATEGY) as f64 / FRAMES_PER_STRATEGY as f64;
            let affine_amount = 0.5 - 0.5 * (std::f64::consts::TAU * local).cos();
            let matrix = affine_matrix(outer_fit.ellipse, affine_amount);
            let inverse = invert_matrix(matrix);
            let mut frame = vec![[7u8, 9u8, 13u8]; OUTPUT_WIDTH * OUTPUT_HEIGHT];

            draw_text(
                &mut frame,
                18,
                16,
                "RAW10 SAM31 ARC TRIAL",
                3,
                [235, 240, 245],
            );
            draw_text(
                &mut frame,
                790,
                16,
                &format!(
                    "{}  {} PROMPTS",
                    strategy.name,
                    strategy.follow_on_prompts.len()
                ),
                3,
                [80, 235, 255],
            );
            for output_y in 0..height * PANEL_SCALE {
                for output_x in 0..width * PANEL_SCALE {
                    let source_x = output_x as f64 / PANEL_SCALE as f64;
                    let source_y = output_y as f64 / PANEL_SCALE as f64;
                    let pixel = preview_sample(preview, width, height, source_x, source_y);
                    let left_x = output_x;
                    let screen_y = PANEL_TOP + output_y;
                    frame[screen_y * OUTPUT_WIDTH + left_x] = pixel;

                    let qx = source_x - outer_fit.ellipse.center.0;
                    let qy = source_y - outer_fit.ellipse.center.1;
                    let warped_source = (
                        outer_fit.ellipse.center.0 + matrix[0][0] * qx + matrix[0][1] * qy,
                        outer_fit.ellipse.center.1 + matrix[1][0] * qx + matrix[1][1] * qy,
                    );
                    let right_pixel =
                        preview_sample(preview, width, height, warped_source.0, warped_source.1);
                    let right_x = width * PANEL_SCALE + output_x;
                    frame[screen_y * OUTPUT_WIDTH + right_x] = right_pixel;
                }
            }

            let field_colors = match strategy.name {
                // The direct question's filled region is only a gate whose
                // intersection with the outer contour creates the cyan arcs.
                // Painting its broad interior would obscure the RAW evidence
                // and falsely imply that it is itself the final segmentation.
                "direct-arc-1" => Vec::new(),
                "material-pair-2" => vec![
                    (VISIBLE_IRIS, [255, 170, 20], 0.20),
                    (ADJACENT_SCLERA, [30, 120, 255], 0.20),
                ],
                "occlusion-guard-3" => vec![
                    (VISIBLE_IRIS, [255, 170, 20], 0.18),
                    (ADJACENT_SCLERA, [30, 120, 255], 0.18),
                    (OCCLUDERS, [255, 30, 170], 0.28),
                ],
                "pizza-sides-2" => vec![
                    (LEFT_SLICE, [60, 255, 120], 0.26),
                    (RIGHT_SLICE, [60, 220, 255], 0.26),
                ],
                "lid-material-3" => vec![
                    (VISIBLE_IRIS, [255, 170, 20], 0.18),
                    (UPPER_OCCLUSION, [255, 60, 180], 0.24),
                    (LOWER_OCCLUSION, [255, 60, 180], 0.24),
                ],
                _ => Vec::new(),
            };
            for y in 0..height {
                for x in 0..width {
                    for &(field_index, color, alpha) in &field_colors {
                        if fields[field_index].sample(x as f64, y as f64, width, height) > 0.5 {
                            for display_y in 0..PANEL_SCALE {
                                for display_x in 0..PANEL_SCALE {
                                    blend_pixel(
                                        &mut frame,
                                        (x * PANEL_SCALE + display_x) as isize,
                                        (PANEL_TOP + y * PANEL_SCALE + display_y) as isize,
                                        color,
                                        alpha,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            for &point in outer_fit.flat_tire_points.iter() {
                draw_disc(
                    &mut frame,
                    point.0 * PANEL_SCALE as f64,
                    PANEL_TOP as f64 + point.1 * PANEL_SCALE as f64,
                    4,
                    [255, 40, 165],
                );
            }
            for (sample, &trusted) in evidence.iter().zip(strategy.trusted.iter()) {
                let color = if trusted {
                    [30, 255, 225]
                } else {
                    [255, 80, 160]
                };
                draw_disc(
                    &mut frame,
                    sample.point.0 * PANEL_SCALE as f64,
                    PANEL_TOP as f64 + sample.point.1 * PANEL_SCALE as f64,
                    if trusted { 4 } else { 3 },
                    color,
                );
                let transformed =
                    transformed_point(sample.point, outer_fit.ellipse.center, inverse);
                draw_disc(
                    &mut frame,
                    (width as f64 + transformed.0) * PANEL_SCALE as f64,
                    PANEL_TOP as f64 + transformed.1 * PANEL_SCALE as f64,
                    if trusted { 4 } else { 3 },
                    color,
                );
            }

            let ellipse = strategy.fitted.unwrap_or(outer_fit.ellipse);
            let points = ellipse.dense_points(240);
            for pair in points.windows(2) {
                draw_line(
                    &mut frame,
                    (
                        pair[0].0 * PANEL_SCALE as f64,
                        PANEL_TOP as f64 + pair[0].1 * PANEL_SCALE as f64,
                    ),
                    (
                        pair[1].0 * PANEL_SCALE as f64,
                        PANEL_TOP as f64 + pair[1].1 * PANEL_SCALE as f64,
                    ),
                    [255, 225, 35],
                    1,
                );
                let first = transformed_point(pair[0], outer_fit.ellipse.center, inverse);
                let second = transformed_point(pair[1], outer_fit.ellipse.center, inverse);
                draw_line(
                    &mut frame,
                    (
                        (width as f64 + first.0) * PANEL_SCALE as f64,
                        PANEL_TOP as f64 + first.1 * PANEL_SCALE as f64,
                    ),
                    (
                        (width as f64 + second.0) * PANEL_SCALE as f64,
                        PANEL_TOP as f64 + second.1 * PANEL_SCALE as f64,
                    ),
                    [255, 225, 35],
                    1,
                );
            }

            // A rotating cyan meridian makes the ellipse-to-fronto-parallel
            // circle transition unmistakably animated rather than a static
            // before/after panel.
            let sweep_phase = std::f64::consts::TAU * local;
            let (sweep_point, _) = ellipse_point_normal(ellipse, sweep_phase);
            let sweep_point = transformed_point(sweep_point, outer_fit.ellipse.center, inverse);
            let center = (
                (width as f64 + outer_fit.ellipse.center.0) * PANEL_SCALE as f64,
                PANEL_TOP as f64 + outer_fit.ellipse.center.1 * PANEL_SCALE as f64,
            );
            draw_line(
                &mut frame,
                center,
                (
                    (width as f64 + sweep_point.0) * PANEL_SCALE as f64,
                    PANEL_TOP as f64 + sweep_point.1 * PANEL_SCALE as f64,
                ),
                [0, 235, 255],
                2,
            );
            draw_text(
                &mut frame,
                18,
                594,
                "CYAN TRUSTED   PINK CENSORED",
                2,
                [210, 220, 230],
            );
            draw_text(
                &mut frame,
                790,
                594,
                &format!(
                    "FIT {:.0} PCT  SECTORS {:.0} PCT",
                    strategy.coverage * 100.0,
                    strategy.sector_coverage * 100.0
                ),
                2,
                [210, 220, 230],
            );
            let bytes =
                unsafe { std::slice::from_raw_parts(frame.as_ptr() as *const u8, frame.len() * 3) };
            stdin
                .write_all(bytes)
                .map_err(|error| format!("write review frame to ffmpeg: {error}"))?;
        }
        drop(stdin);
        let status = child
            .wait()
            .map_err(|error| format!("wait for ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("ffmpeg exited with {status}"));
        }
        Ok(())
    }
}

#[cfg(feature = "sam31")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    enabled::main()
}

#[cfg(not(feature = "sam31"))]
fn main() {
    eprintln!("rebuild buttercup-sam31-arc-trial with --features sam31");
    std::process::exit(2);
}
