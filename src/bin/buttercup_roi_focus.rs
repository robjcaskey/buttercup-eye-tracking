//! Temporary compact ROI ellipse archive and unsupervised focus-volume replay.
#[path = "buttercup_roi_focus/archive.rs"]
mod archive;
#[path = "buttercup_roi_focus/area_consistency.rs"]
mod area_consistency;
#[path = "buttercup_roi_focus/continuity.rs"]
mod continuity;
#[path = "buttercup_roi_focus/fresh_focus.rs"]
mod fresh_focus;
#[path = "buttercup_roi_focus/lighting.rs"]
mod lighting;
#[path = "buttercup_roi_focus/model_eval.rs"]
mod model_eval;
#[path = "buttercup_roi_focus/movie.rs"]
mod movie;
#[path = "buttercup_roi_focus/pack.rs"]
mod pack;
#[path = "buttercup_roi_focus/replay.rs"]
mod replay;
#[path = "buttercup_roi_focus/surface_sign.rs"]
mod surface_sign;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let result=match args.first().map(String::as_str) {
        Some("surface-report") if args.len()==2=>surface_sign::report::run(&args[1]),
        Some("surface-sclera-report") if args.len()==2=>surface_sign::report::sclera(&args[1]),
        Some("surface-rigid-report") if args.len()==3=>surface_sign::report::sclera_rigid(&args[1],&args[2]),
        Some("surface-temporal-report") if args.len()==4=>surface_sign::report::temporal(&args[1],&args[2],&args[3]),
        Some("surface-sclera-motion") if args.len()==4||args.len()==5=>surface_sign::sclera_motion_run(&args[1],&args[2],&args[3],args.get(4).map_or("sclera",String::as_str)),
        Some("surface-temporal-lighting") if args.len()==4||args.len()==5=>surface_sign::temporal_run(&args[1],&args[2],&args[3],args.get(4).map(String::as_str)),
        Some("surface-lid-oracle") if args.len()==4=>surface_sign::lid_circle::run(&args[1],&args[2],&args[3]),
        Some("surface-native-lids") if args.len()==4=>surface_sign::lid_circle::native::run(&args[1],&args[2],&args[3]),
        Some("surface-lid-visual") if args.len()==6=>surface_sign::lid_circle::visual::run(&args[1],&args[2],&args[3],&args[4],&args[5]),
        Some("surface-evidence-audit") if args.len()==4=>surface_sign::evidence_audit::run(&args[1],&args[2],&args[3]),
        Some("surface-native-lid-report") if args.len()==3=>surface_sign::lid_circle::native::report(&args[1],&args[2]),
        Some("surface-lid-review") if args.len()==4=>surface_sign::lid_review::prepare(&args[1],&args[2],&args[3]),
        Some("surface-lid-proposals") if args.len()==5=>surface_sign::lid_circle::proposals::run(&args[1],&args[2],&args[3],&args[4]),
        Some("surface-center-motion") if args.len()==4||args.len()==5=>args.get(4).map_or(Ok(4),|s|s.parse::<i32>().map_err(Into::into)).and_then(|radius|surface_sign::center_motion::run(&args[1],&args[2],&args[3],radius)),
        Some("surface-sam-anatomy") if args.len()==5=>surface_sign::anatomy_run(&args[1],&args[2],&args[3],&args[4]),
        Some("surface-lids") if args.len()==4||args.len()==5=>args.get(4).map_or(Ok(0),|s|s.parse::<usize>().map_err(Into::into)).and_then(|limit|surface_sign::lids_run(&args[1],&args[2],&args[3],limit)),
        Some("surface-lighting") if args.len()==4||args.len()==5=>args.get(4).map_or(Ok(0),|s|s.parse::<usize>().map_err(Into::into)).and_then(|limit|surface_sign::lighting_run(&args[1],&args[2],&args[3],limit)),
        Some("surface-sign") if args.len()==4=>surface_sign::run(&args[1],&args[2],&args[3]),
        Some("fresh-focus") if args.len()==4=>fresh_focus::run(&args[1],&args[2],&args[3]),
        Some("lighting") if args.len()==3=>lighting::run(&args[1],&args[2]),
        Some("pack") if args.len()>=4=>pack::run(&args[1],&args[2],&args[3..]),
        Some("replay") if args.len()==3=>replay::run(&args[1],&args[2]),
        Some("controls") if args.len()==2=>replay::controls(&args[1]),
        Some("compact") if args.len()==3=>archive::compact_file(&args[1],&args[2]),
        Some("compare") if args.len()==3=>replay::compare(&args[1],&args[2]),
        Some("movie") if args.len()==4||args.len()==5=>movie::run(&args[1],&args[2],&args[3],args.get(4).map(String::as_str)),
        Some("continuity") if args.len()==3=>continuity::run(&args[1],&args[2]),
        Some("model-eval") if args.len()==7=>model_eval::run(&args[1..]),
        _=>Err("usage: buttercup_roi_focus pack NEW_TMP_BINARY NATIVE_CONICS_JSONL ROOT... | compact OLD_BINARY NEW_BINARY | replay BINARY NEW_OUTPUT_DIR | compare BINARY NEW_OUTPUT_JSON | controls NEW_OUTPUT_JSON | movie BINARY REPLAY_DIR NEW_OUTPUT_DIR [RAW_RECOVERY_DIR] | continuity REPLAY_DIR NEW_OUTPUT_DIR | model-eval BINARY REPLAY_DIR MODEL_RUN CONTINUITY_DIR MOVIE_DIR NEW_OUT | lighting FRESH_SEGMENTATION_DIR NEW_OUT | fresh-focus FRESH_SEGMENTATION_DIR ORIGINAL_FOCUS_DIR NEW_OUT | surface-evidence-audit AREA_FIRST_DIR FRESH_DIR NEW_OUT | surface-lid-visual AREA_FIRST_DIR FRESH_DIR REVIEW_DIR ESTIMATES_JSON_OR_DASH NEW_OUT | surface-lid-review AREA_FIRST_DIR FRESH_DIR NEW_REVIEW_NAME | surface-native-lids AREA_FIRST_DIR FRESH_DIR NEW_OUT | surface-native-lid-report RUN_DIR AREA_FIRST_DIR | surface-lid-oracle AREA_FIRST_DIR FRESH_DIR NEW_OUT | surface-lid-proposals AREA_FIRST_DIR FRESH_DIR ANATOMY_RUN NEW_OUT | surface-center-motion AREA_FIRST_DIR FRESH_DIR NEW_OUT [PATCH_RADIUS=4|8] | surface-sign AREA_FIRST_DIR FRESH_DIR NEW_OUT | surface-lids AREA_FIRST_DIR FRESH_DIR NEW_OUT [PILOT_LIMIT] | surface-lighting AREA_FIRST_DIR FRESH_DIR NEW_OUT [PILOT_LIMIT] | surface-sam-anatomy AREA_FIRST_DIR FRESH_DIR ANATOMY_RUN NEW_OUT | surface-sclera-motion AREA_FIRST_DIR FRESH_DIR NEW_OUT [whole|whole-dense|sclera|rigid] | surface-temporal-lighting AREA_FIRST_DIR FRESH_DIR NEW_OUT [SAM_ANATOMY_RUN] | surface-temporal-report BASELINE_RUN CANDIDATE_RUN NEW_OUT | surface-rigid-report BASELINE_RUN RIGID_RUN | surface-sclera-report RUN_DIR | surface-report RUN_DIR".into()),
    };
    if let Err(e) = result {
        eprintln!("ROI focus: {e}");
        std::process::exit(1);
    }
}
