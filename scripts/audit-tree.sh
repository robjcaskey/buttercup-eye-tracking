#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

status=0
device_project_token='o''sbot'

mapfile -t unexpected_files < <(
  comm -23 \
    <(find -P . -path './.git' -prune -o -type f -print | sort) \
    <(printf '%s\n' \
      ./.cargo/config.toml \
      ./.gitignore \
      ./AGENTS.md \
      ./bootstrapability.md \
      ./Cargo.lock \
      ./Cargo.toml \
      ./crates/buttercup/Cargo.toml \
      ./build.rs \
      ./README.md \
      ./docs/viewer-overview.png \
      ./docs/geometry-architecture.md \
      ./docs/joint-conic-solver.md \
      ./docs/flat-tire-area-and-motion.md \
      ./docs/eye-anatomy-sign-evidence.md \
      ./docs/sign-acquisition-trials.md \
      ./docs/meridian-sign-continuity.md \
      ./docs/kinematic-sign-beam.md \
      ./docs/perspective-sign.md \
      ./docs/roi-reframe-continuity.md \
      ./docs/viewer-workspaces.md \
      ./docs/void-sightline-review.md \
      ./docs/sam-concurrency-and-gaze-latency.md \
      ./docs/eye-student.md \
      ./docs/raw-native-student.md \
      ./docs/limbus-refiner.md \
      ./docs/limbus-frame-pair-training.md \
      ./docs/calibration-sign-model.md \
      ./docs/roi-focus-stream.md \
      ./docs/roi-lighting.md \
      ./docs/sclera-shape-sign.md \
      ./docs/sam-roi-contact-sheets.md \
      ./docs/pink-waterfall-contact-sheets.md \
      ./docs/globe-meridian-contact-sheets.md \
      ./docs/sclera-measurement-requirements.md \
      ./docs/camera-mount-assumption.md \
      ./docs/camera-startup.md \
      ./docs/calibration-playback.md \
      ./docs/raw-recording-evidence.md \
      ./docs/presence-cooperation.md \
      ./docs/eyewear-reflection-labels.md \
      ./docs/glasses-parallax.md \
      ./docs/optical-clock-debugging.md \
      ./docs/physics-naming-audit.md \
      ./scripts/audit-tree.sh \
      ./scripts/run-viewer.sh \
      ./scripts/camera-focus.py \
      ./scripts/camera-exposure.py \
      ./scripts/eye-focus-osd.py \
      ./scripts/audit-readmission-corpus.py \
      ./scripts/inventory-stereo-corpus.py \
      ./scripts/prepare-stereo-replay.py \
      ./scripts/report-stereo-conics.py \
      ./scripts/report-stereo-motion.py \
      ./scripts/test-stereo-conics-report.py \
      ./scripts/score-stereo-labels.py \
      ./scripts/validate-recording.py \
      ./scripts/toggle-mouse-output.py \
      ./scripts/test-mouse-output.py \
      ./scripts/run-sam31-arc-trial.sh \
      ./scripts/run-sam31-prompt-lab.sh \
      ./scripts/prompt-lab-troublesome-3.txt \
      ./scripts/prompt-lab-troublesome-20.txt \
      ./src/checkerboard_calibration.rs \
      ./src/bootstrapability.rs \
      ./src/bootstrapability/tests.rs \
      ./src/bin/buttercup_bootstrap_check.rs \
      ./src/bin/buttercup_calibration_sign.rs \
      ./src/bin/buttercup_calibration_sign/data.rs \
      ./src/bin/buttercup_calibration_sign/train.rs \
      ./src/bin/buttercup_calibration_sign/review.rs \
      ./src/bin/buttercup_calibration_sign/movie.rs \
      ./src/bin/buttercup_calibration_sign/candidate_movie.rs \
      ./src/bin/buttercup_calibration_sign/canvas.rs \
      ./src/bin/buttercup_calibration_sign/native.rs \
      ./src/bin/buttercup_calibration_sign/branch_labels.rs \
      ./src/bin/buttercup_calibration_sign/raw_conic.rs \
      ./src/bin/buttercup_calibration_sign/sam_export.rs \
      ./src/bin/buttercup_calibration_sign/sam_native.rs \
      ./src/bin/buttercup_glint_evidence.rs \
      ./src/bin/buttercup_sam31_gaze_dot_trial.rs \
      ./src/bin/buttercup_limbus_sign_probe.rs \
      ./src/bin/buttercup_motion_sign_sheet.rs \
      ./src/bin/buttercup_motion_sign_sheet/clusters.rs \
      ./src/bin/buttercup_motion_sign_sheet/vectors.rs \
      ./src/bin/buttercup_motion_sign_sheet/eye_center.rs \
      ./src/bin/buttercup_motion_sign_sheet/sign_video.rs \
      ./src/bin/buttercup_motion_sign_sheet/eyelid_cue.rs \
      ./src/projected_eye_center.rs \
      ./src/bin/buttercup_projected_center_corpus.rs \
      ./src/bin/buttercup_iris_feature_motion.rs \
      ./src/bin/buttercup_cluster_replay.rs \
      ./src/bin/buttercup_cluster_replay/targets.rs \
      ./src/bin/buttercup_cluster_replay/cohorts.rs \
      ./src/bin/buttercup_cluster_replay/carrier.rs \
      ./src/bin/buttercup_cluster_replay/partition.rs \
      ./src/bin/buttercup_cluster_replay/demo.rs \
      ./src/bin/buttercup_cluster_replay/vessels.rs \
      ./src/bin/buttercup_cluster_replay/vessel_ridges.rs \
      ./src/bin/buttercup_iris_feature_motion/layers.rs \
      ./src/bin/buttercup_outer_rigid_motion.rs \
      ./src/bin/buttercup_outer_rigid_motion/stabilized.rs \
      ./src/bin/buttercup_outer_rigid_motion/factorization.rs \
      ./src/bin/buttercup_sclera_specular_sheet.rs \
      ./src/bin/buttercup_sclera_visibility.rs \
      ./src/bin/buttercup_sign_expert_trial.rs \
      ./src/bin/buttercup_sign_expert_trial/scene.rs \
      ./src/bin/buttercup_prepare_eye_student.rs \
      ./src/bin/buttercup_calibration_strata.rs \
      ./src/bin/buttercup_prepare_limbus_refiner.rs \
      ./src/bin/buttercup_report_eye_student.rs \
      ./src/bin/buttercup_report_limbus_refiner.rs \
      ./src/training_refiner_data.rs \
      ./src/student_comparison.rs \
      ./src/calibration_acquisition.rs \
      ./src/geometry.rs \
      ./src/focus_region.rs \
      ./src/bin/buttercup_roi_focus.rs \
      ./src/bin/buttercup_roi_focus/archive.rs \
      ./src/bin/buttercup_roi_focus/continuity.rs \
      ./src/bin/buttercup_roi_focus/pack.rs \
      ./src/bin/buttercup_roi_focus/replay.rs \
      ./src/bin/buttercup_roi_focus/movie.rs \
      ./src/bin/buttercup_roi_focus/model_eval.rs \
      ./src/bin/buttercup_roi_focus/lighting.rs \
      ./src/bin/buttercup_roi_focus/fresh_focus.rs \
      ./src/bin/buttercup_roi_focus/area_consistency.rs \
      ./src/bin/buttercup_roi_focus/surface_sign.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/geometry.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/lids.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/lid_circle.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/native_lids.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/lid_visual.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/evidence_audit.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/lid_review.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/lid_proposals.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/temporal.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/sclera_motion.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/rigid_motion.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/anatomy_masks.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/center_motion.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/photometry.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/paired.rs \
      ./src/bin/buttercup_roi_focus/surface_sign/report.rs \
      ./src/bin/buttercup_calibration_sign/roi_resegment.rs \
      ./src/bin/buttercup_calibration_sign/roi_anatomy.rs \
      ./src/bin/buttercup_calibration_sign/pupil_prompts.rs \
      ./src/bin/buttercup_calibration_sign/contact_sheet.rs \
      ./src/bin/buttercup_calibration_sign/sclera_arcs.rs \
      ./src/bin/buttercup_calibration_sign/sclera_splat_inputs.rs \
      ./src/bin/buttercup_cluster_replay/sclera_splats.rs \
      ./src/bin/buttercup_cluster_replay/colmap_probe.rs \
      ./src/bin/buttercup_cluster_replay/colmap_playback.rs \
      ./src/bin/buttercup_cluster_replay/colmap_groups.rs \
      ./src/bin/buttercup_cluster_replay/warp_probe.rs \
      ./src/bin/buttercup_cluster_replay/iris_pivot.rs \
      ./src/bin/buttercup_cluster_replay/iris_pivot_math.rs \
      ./src/bin/buttercup_cluster_replay/iris_pivot_prior.rs \
      ./src/bin/buttercup_cluster_replay/iris_pivot_binary.rs \
      ./src/bin/buttercup_cluster_replay/iris_pivot_color.rs \
      ./src/bin/buttercup_cluster_replay/iris_pivot_viewer.html \
      ./src/bin/buttercup_cluster_replay/z_motion_groups.rs \
      ./src/bin/buttercup_cluster_replay/z_motion_groups_math.rs \
      ./src/bin/buttercup_cluster_replay/z_motion_groups_pixels.rs \
      ./src/bin/buttercup_cluster_replay/z_motion_groups_history.rs \
      ./src/bin/buttercup_cluster_replay/z_motion_scene.rs \
      ./src/bin/buttercup_cluster_replay/z_motion_groups_history.html \
      ./src/bin/buttercup_cluster_replay/z_motion_groups_viewer.html \
      ./src/bin/buttercup_cluster_replay/z_motion3d.rs \
      ./src/bin/buttercup_cluster_replay/z_motion3d_math.rs \
      ./src/bin/buttercup_cluster_replay/z_motion3d_viewer.html \
      ./src/bin/buttercup_cluster_replay/z_discovery.rs \
      ./src/bin/buttercup_cluster_replay/z_discovery_math.rs \
      ./src/bin/buttercup_cluster_replay/z_discovery_bounds.rs \
      ./src/bin/buttercup_cluster_replay/z_discovery_viewer.html \
      ./src/bin/buttercup_cluster_replay/iris_layers.rs \
      ./src/bin/buttercup_cluster_replay/iris_layers_viewer.html \
      ./src/bin/buttercup_cluster_replay/warp_math.rs \
      ./src/bin/buttercup_cluster_replay/warp_viewer.html \
      ./src/bin/buttercup_cluster_replay/colmap_viewer.html \
      ./src/bin/buttercup_cluster_replay/raw_cadence.rs \
      ./src/bin/buttercup_cluster_replay/splat_geometry.rs \
      ./src/sclera_splat_input_recipe.rs \
      ./src/limbus_refiner.rs \
      ./src/limbus_refiner_train.rs \
      ./src/limbus_refiner_cpu.rs \
      ./src/bin/buttercup_limbus_cpu.rs \
      ./src/limbus_refiner_view.rs \
      ./src/limbus_refinement.rs \
      ./src/eye_evidence_stage.rs \
      ./src/refinement_target_report.rs \
      ./src/refinement_frame_pairs.rs \
      ./src/calibration_frame_pairs.rs \
      ./src/calibration_sign_model.rs \
      ./src/recorded_bundle.rs \
      ./src/display_pose_wireframe.rs \
      ./src/monitor_location.rs \
      ./src/mouse_output.rs \
      ./src/mouse_output/linux.rs \
      ./src/desktop_gaze.rs \
      ./src/wleyes.rs \
      ./src/wleyes/argb.rs \
      ./src/presence_protocol.rs \
      ./src/camera_cooperation.rs \
      ./src/camera_cooperation/diagnostics.rs \
      ./src/camera_cooperation/tests.rs \
      ./src/bin/buttercup_eyewear_review.rs \
      ./src/bin/buttercup_glasses_parallax.rs \
      ./src/glasses_parallax.rs \
      ./src/glasses_parallax_inventory.rs \
      ./src/recorded_stimulus.rs \
      ./src/gaze_focus.rs \
      ./src/gaze_focus/sway.rs \
      ./src/gaze_focus/tests.rs \
      ./src/gaze_accuracy.rs \
      ./src/conic_solver/parallax_sign.rs \
      ./src/conic_solver.rs \
      ./src/conic_solver/joint.rs \
      ./src/conic_solver/joint/mask_levels.rs \
      ./src/conic_solver/joint/tests.rs \
      ./src/conic_solver/joint/uncertainty.rs \
      ./src/conic_solver/joint/posterior.rs \
      ./src/conic_solver/joint/posterior/annealed.rs \
      ./src/conic_solver/joint/posterior/populations.rs \
      ./src/conic_solver/fidelity.rs \
      ./src/conic_solver/constraint_tests.rs \
      ./src/outline_conic_segments.rs \
      ./src/outline_conic_segments/recent_exclusion.rs \
      ./src/outline_conic_segments/sparse_evidence.rs \
      ./src/outline_conic_segments/sparse_evidence/uncertainty.rs \
      ./src/outline_conic_segments/sparse_evidence/outer_candidates.rs \
      ./src/outline_conic_segments/partial_outline.rs \
      ./src/bin/buttercup_stereo_conic_eval/source_order.rs \
      ./src/roi_evidence.rs \
      ./src/roi_evidence/timing.rs \
      ./src/roi_continuity.rs \
      ./src/roi_visibility.rs \
      ./src/eye_scene_model.rs \
      ./src/eye_scene_model/limbus_scale.rs \
      ./src/eye_scene_model/binocular_pose.rs \
      ./src/conic_solver/camera_mount.rs \
      ./src/conic_solver/continuous_gaze_sign.rs \
      ./src/eye_scene_model/pupil_center.rs \
      ./src/eye_scene_model/pupil_projection.rs \
      ./src/eye_scene_model/pupil_size.rs \
      ./src/eye_scene_model/sign_motion.rs \
      ./src/eye_scene_model/sign_continuity.rs \
      ./src/eye_scene_model/sign_kinematic_beam.rs \
      ./src/eye_scene_model/perspective_sign.rs \
      ./src/gaze_target_solver.rs \
      ./src/joint_gaze_live.rs \
      ./src/gaze_target_solver/joint_tracking.rs \
      ./src/binocular_coordinator.rs \
      ./src/binocular_coordinator/source_pairing.rs \
      ./src/bin/buttercup_screen_reflection_raw_decode.rs \
      ./src/bin/buttercup_sam31_arc_trial.rs \
      ./src/bin/buttercup_flat_tire_eval.rs \
      ./src/bin/buttercup_kinematic_sign_replay.rs \
      ./src/bin/buttercup_perspective_sign_replay.rs \
      ./src/bin/buttercup_stereo_conic_eval.rs \
      ./src/bin/buttercup_sam31_prompt_bundle.rs \
      ./src/bin/buttercup_sam31_video_graph.rs \
      ./src/bin/buttercup_sam31_tracker_bundle.rs \
      ./src/bin/buttercup_raw10_preview.rs \
      ./src/bin/buttercup_pink_waterfall_sheet.rs \
      ./src/bin/buttercup_globe_meridian_sheet.rs \
      ./src/bin/buttercup_waterfall_validate.rs \
      ./src/bin/buttercup_waterfall_validate/composite.rs \
      ./src/bin/buttercup_waterfall_validate/landmarks.rs \
      ./src/bin/buttercup_eye_student.rs \
      ./src/bin/buttercup_limbus_refiner.rs \
      ./src/coupled_eye_kinematics.rs \
      ./src/keyboard_peeper.rs \
      ./src/main.rs \
      ./src/lib.rs \
      ./src/viewer_ui.rs \
      ./src/viewer_ui/stereo.rs \
      ./src/viewer_ui/void_sightline.rs \
      ./src/conic_solver/nested_pupil.rs \
      ./src/native_mediapipe.rs \
      ./src/offline_segmentation_replay.rs \
      ./src/offline_segmentation_replay/showcase.rs \
      ./src/offline_segmentation_replay/contact_sign.rs \
      ./src/offline_segmentation_replay/sign_acquisition.rs \
      ./src/offline_segmentation_replay/stereo.rs \
      ./src/offline_segmentation_replay/calibration.rs \
      ./src/offline_segmentation_replay/calibration/playback.rs \
      ./src/offline_segmentation_replay/calibration/geometry.rs \
      ./src/pupil_clock_supervision.rs \
      ./src/pivot_region_scheduler.rs \
      ./src/raw10.rs \
      ./src/raw_preview.rs \
      ./src/portable.rs \
      ./src/raw_eye_model_protocol.rs \
      ./src/raw_eye_model_protocol/thumbnail.rs \
      ./src/recording_trace.rs \
      ./src/recording_trace/scene.rs \
      ./src/parallel_work.rs \
      ./src/gaze_cursor.rs \
      ./src/raw_iris_focus.rs \
      ./src/raw_motion_octrees.rs \
      ./src/raw_sclera_vein_graph.rs \
      ./src/raw_sclera_red_canny.rs \
      ./src/sam31_outer.rs \
      ./src/sam31_boundary_logits.rs \
      ./src/sam31_student.rs \
      ./src/sam31_student_raw.rs \
      ./src/sam31_pipeline.rs \
      ./src/source_queue_budget.rs \
      ./src/student_preview.rs \
      ./src/sam31_cuda_stream.cpp \
      ./src/sam31_photometric.rs \
      ./src/sam31_text.rs \
      ./src/screen_reflection_clock.rs \
      ./src/screen_reflection_code.rs \
      ./src/screen_reflection_live.rs \
      ./src/screen_reflection_raw.rs \
      ./src/screen_reflection_stimulus.rs \
      ./src/screen_reflection_temporal.rs \
      ./src/screen_reflection_border.rs \
      ./src/specular_map.rs \
      ./src/visible_lighthouse_control.rs | sort)
)
if ((${#unexpected_files[@]})); then
  printf 'files outside the reviewed allowlist:\n%s\n' "${unexpected_files[*]}" >&2
  status=1
fi

mapfile -t forbidden_files < <(
  find -P . -path './.git' -prune -o -type f \
    ! -path './docs/viewer-overview.png' \
    \( -iname '*.raw' -o -iname '*.raw10' -o -iname '*.gray16le' \
       -o -iname '*.nv12' -o -iname '*.yuv' -o -iname '*.dng' \
       -o -iname '*.tar' -o -iname '*.zip' -o -iname '*.7z' \
       -o -iname '*.mkv' -o -iname '*.mp4' -o -iname '*.ppm' \
       -o -iname '*.pgm' -o -iname '*.pnm' -o -iname '*.png' \
       -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \
       -o -iname '*.log' -o -iname '*.jsonl' -o -iname '*.pt' \
       -o -iname '*.pth' -o -iname '*.onnx' -o -iname '*.so' \
       -o -iname '*.ko' -o -iname '*.dll' -o -iname '*.exe' \
       -o -iname '*.bin' \) -print
)
if ((${#forbidden_files[@]})); then
  printf 'forbidden source-tree files:\n%s\n' "${forbidden_files[*]}" >&2
  status=1
fi

mapfile -t forbidden_names < <(
  find -P . -path './.git' -prune -o -iname "*${device_project_token}*" -print
)
if ((${#forbidden_names[@]})); then
  printf 'device-project names in source tree:\n%s\n' "${forbidden_names[*]}" >&2
  status=1
fi

if rg -n '/home/rob/|extracted_rootfs|/app/bin/camera|af_test|UVCIOC|PyUSB' \
  --glob '!Cargo.lock' --glob '!AGENTS.md' --glob '!scripts/audit-tree.sh' .; then
  printf 'forbidden source-project or device-control references found\n' >&2
  status=1
fi

for link in data outputs; do
  if [[ ! -L $link ]]; then
    printf 'required data link is missing: %s\n' "$link" >&2
    status=1
  elif [[ $(readlink -f "$link") != /mnt/bulk_data/buttercup-eye-tracking* ]]; then
    printf 'data link escapes the Buttercup bulk root: %s -> %s\n' \
      "$link" "$(readlink -f "$link")" >&2
    status=1
  fi
done

if ((status == 0)); then
  printf 'source-tree audit passed\n'
fi
exit "$status"
