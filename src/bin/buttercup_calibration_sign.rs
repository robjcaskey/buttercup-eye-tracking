//! Cold, target-supervised two-RAW-frame sign experiment. No learned ancestors.
#![allow(dead_code)]
#[path = "../bootstrapability.rs"]
mod bootstrapability;
#[path = "buttercup_calibration_sign/branch_labels.rs"]
mod branch_labels;
#[path = "buttercup_calibration_sign/candidate_movie.rs"]
mod candidate_movie;
#[path = "buttercup_calibration_sign/data.rs"]
mod data;
#[path = "buttercup_calibration_sign/movie.rs"]
mod movie;
#[path = "buttercup_calibration_sign/native.rs"]
mod native;
#[path = "buttercup_calibration_sign/review.rs"]
mod review;
#[cfg(feature = "sam31")]
#[path = "buttercup_calibration_sign/roi_anatomy.rs"]
mod roi_anatomy;
#[cfg(feature = "sam31")]
#[path = "buttercup_calibration_sign/roi_resegment.rs"]
mod roi_resegment;
#[path = "buttercup_calibration_sign/sam_export.rs"]
mod sam_export;
#[cfg(feature = "sam31")]
#[path = "buttercup_calibration_sign/sam_native.rs"]
mod sam_native;
#[path = "buttercup_calibration_sign/train.rs"]
mod train;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() {
    if let Err(error) = run() {
        eprintln!("calibration sign: {error}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        #[cfg(feature = "sam31")]
        Some("sclera-splat-inputs") => roi_anatomy::sclera_splat_inputs::run(&args),
        #[cfg(feature = "sam31")]
        Some("roi-contact-sheet") => roi_anatomy::contact_sheet::run(&args),
        #[cfg(feature = "sam31")]
        Some("roi-resegment") => roi_resegment::run(&args),
        #[cfg(feature = "sam31")]
        Some("roi-anatomy") => roi_anatomy::run(&args),
        #[cfg(feature = "sam31")]
        Some("roi-anatomy-review") => roi_anatomy::review(&args),
        #[cfg(feature = "sam31")]
        Some("roi-anatomy-compare") => roi_anatomy::compare(&args),
        #[cfg(feature = "sam31")]
        Some("pupil-prompt-select") => roi_anatomy::pupil_prompts::select(&args),
        #[cfg(feature = "sam31")]
        Some("pupil-prompt-run" | "roi-prompt-run") => roi_anatomy::pupil_prompts::run(&args),
        #[cfg(feature = "sam31")]
        Some("sclera-arcs") => roi_anatomy::pupil_prompts::sclera_arcs::run(&args),
        Some("inventory") if args.len() == 3 => data::inventory(&args[1], &args[2]),
        Some("cold") if args.len() == 3 => train::cold(&args[1], &args[2]),
        Some("cold-branches") if args.len() == 3 => train::cold_mode(&args[1], &args[2], true, false),
        Some("branch-labels") if args.len() == 3 => train::cold_mode(&args[1], &args[2], true, true),
        Some("infer") => review::infer(&args),
        Some("sam-export" | "sam-export-anatomy" | "sam-export-anatomy-cpu") => sam_export::run(&args),
        #[cfg(feature = "sam31")]
        Some("sam-native" | "sam-native-single") => sam_native::run(&args),
        #[cfg(feature = "sam31")]
        Some("cold-sam-branches" | "sam-branch-labels") => train::cold_sam(&args),
        Some("movie") => movie::run(&args),
        Some("candidate-movie") => candidate_movie::run(&args),
        Some("native") => native::run(&args),
        Some("native-conics") => native::run(&args),
        Some("review") if args.len() == 2 => review::review(&args[1]),
        _ => Err(
            "usage: buttercup_calibration_sign roi-contact-sheet REQUEST_JSON NEW_OUT; inventory|cold|cold-branches|branch-labels CORPUS NEW_OUT; native|native-conics CORPUS NEW_OUT [ARCHIVE]; sam-export|sam-export-anatomy CHECKPOINT NEW_OUT; sam-export-anatomy-cpu CHECKPOINT NEW_OUT [SIX_PROMPTS_JSON]; roi-anatomy AREA_FIRST_DIR CPU_EXPORT NEW_OUT [PILOT_LIMIT] [quad_rgb|gray|gray-bilateral-v1]; roi-anatomy-review RUN NEW_OUT; pupil-prompt-select AREA_FIRST NEW_OUT natural|matched|matched-random [SEED]; pupil-prompt-run SELECTION CPU_EXPORT NEW_OUT; roi-prompt-run SELECTION CPU_EXPORT NEW_OUT [pupil|black-pupil|skin|eye-corners|vein-regions|lower-eyelid-arc|sclera]; sclera-arcs COMPLETED_SCLERA_RUN NEW_OUT; roi-anatomy-compare RUN_A RUN_B RUN_C NEW_OUT PROMPT_INDEX; sam-native|sam-native-single CORPUS EXPORT NEW_OUT [ARCHIVE]; cold-sam-branches CORPUS CHECKPOINT NEW_OUT; sam-branch-labels CORPUS CHECKPOINT NEW_OUT [ARCHIVE]; roi-resegment BINARY NEURAL_EVAL SAM_EXPORT OBELISK NEW_OUT [LIMIT] (SAM modes require sam31); movie CORPUS RUN NEW_OUT [ARCHIVE]; candidate-movie RUN EXISTING_MOVIE_DIR NEW_OUT; infer MODEL PAIR OUTPUT [GEOMETRY]; review RUN"
                .into(),
        ),
    }
}
