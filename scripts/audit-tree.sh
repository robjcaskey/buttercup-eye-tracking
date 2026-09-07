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
      ./Cargo.lock \
      ./Cargo.toml \
      ./crates/buttercup/Cargo.toml \
      ./build.rs \
      ./README.md \
      ./docs/viewer-overview.png \
      ./docs/geometry-architecture.md \
      ./docs/joint-conic-solver.md \
      ./docs/flat-tire-area-and-motion.md \
      ./docs/sign-acquisition-trials.md \
      ./docs/meridian-sign-continuity.md \
      ./docs/roi-reframe-continuity.md \
      ./docs/viewer-workspaces.md \
      ./docs/sam-concurrency-and-gaze-latency.md \
      ./docs/raw-recording-evidence.md \
      ./docs/physics-naming-audit.md \
      ./scripts/audit-tree.sh \
      ./scripts/run-viewer.sh \
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
      ./src/calibration_acquisition.rs \
      ./src/geometry.rs \
      ./src/display_pose_wireframe.rs \
      ./src/monitor_location.rs \
      ./src/mouse_output.rs \
      ./src/mouse_output/linux.rs \
      ./src/desktop_gaze.rs \
      ./src/gaze_focus.rs \
      ./src/gaze_focus/sway.rs \
      ./src/gaze_focus/tests.rs \
      ./src/gaze_accuracy.rs \
      ./src/conic_solver.rs \
      ./src/conic_solver/joint.rs \
      ./src/conic_solver/joint/tests.rs \
      ./src/conic_solver/fidelity.rs \
      ./src/conic_solver/constraint_tests.rs \
      ./src/outline_conic_segments.rs \
      ./src/outline_conic_segments/recent_exclusion.rs \
      ./src/outline_conic_segments/sparse_evidence.rs \
      ./src/outline_conic_segments/partial_outline.rs \
      ./src/roi_evidence.rs \
      ./src/roi_evidence/timing.rs \
      ./src/roi_continuity.rs \
      ./src/roi_visibility.rs \
      ./src/eye_scene_model.rs \
      ./src/eye_scene_model/limbus_scale.rs \
      ./src/eye_scene_model/binocular_pose.rs \
      ./src/eye_scene_model/pupil_center.rs \
      ./src/eye_scene_model/pupil_projection.rs \
      ./src/eye_scene_model/pupil_size.rs \
      ./src/eye_scene_model/sign_motion.rs \
      ./src/eye_scene_model/sign_continuity.rs \
      ./src/gaze_target_solver.rs \
      ./src/joint_gaze_live.rs \
      ./src/gaze_target_solver/joint_tracking.rs \
      ./src/binocular_coordinator.rs \
      ./src/binocular_coordinator/source_pairing.rs \
      ./src/bin/buttercup_screen_reflection_raw_decode.rs \
      ./src/bin/buttercup_sam31_arc_trial.rs \
      ./src/bin/buttercup_flat_tire_eval.rs \
      ./src/bin/buttercup_stereo_conic_eval.rs \
      ./src/bin/buttercup_sam31_prompt_bundle.rs \
      ./src/bin/buttercup_sam31_video_graph.rs \
      ./src/bin/buttercup_sam31_tracker_bundle.rs \
      ./src/bin/buttercup_raw10_preview.rs \
      ./src/coupled_eye_kinematics.rs \
      ./src/keyboard_peeper.rs \
      ./src/main.rs \
      ./src/viewer_ui.rs \
      ./src/native_mediapipe.rs \
      ./src/offline_segmentation_replay.rs \
      ./src/offline_segmentation_replay/showcase.rs \
      ./src/offline_segmentation_replay/contact_sign.rs \
      ./src/offline_segmentation_replay/sign_acquisition.rs \
      ./src/offline_segmentation_replay/stereo.rs \
      ./src/pupil_clock_supervision.rs \
      ./src/pivot_region_scheduler.rs \
      ./src/raw10.rs \
      ./src/raw_preview.rs \
      ./src/portable.rs \
      ./src/raw_eye_model_protocol.rs \
      ./src/raw_eye_model_protocol/thumbnail.rs \
      ./src/recording_trace.rs \
      ./src/recording_trace/scene.rs \
      ./src/raw_iris_focus.rs \
      ./src/raw_motion_octrees.rs \
      ./src/raw_sclera_vein_graph.rs \
      ./src/raw_sclera_red_canny.rs \
      ./src/sam31_outer.rs \
      ./src/sam31_pipeline.rs \
      ./src/sam31_cuda_stream.cpp \
      ./src/sam31_photometric.rs \
      ./src/sam31_text.rs \
      ./src/screen_reflection_clock.rs \
      ./src/screen_reflection_code.rs \
      ./src/screen_reflection_live.rs \
      ./src/screen_reflection_raw.rs \
      ./src/screen_reflection_stimulus.rs \
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
