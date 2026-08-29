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
      ./README.md \
      ./docs/viewer-overview.png \
      ./scripts/audit-tree.sh \
      ./scripts/run-viewer.sh \
      ./src/checkerboard_calibration.rs \
      ./src/bin/buttercup_screen_reflection_raw_decode.rs \
      ./src/coupled_eye_kinematics.rs \
      ./src/main.rs \
      ./src/native_mediapipe.rs \
      ./src/offline_segmentation_replay.rs \
      ./src/pupil_clock_supervision.rs \
      ./src/raw10.rs \
      ./src/raw_eye_model_protocol.rs \
      ./src/raw_iris_focus.rs \
      ./src/raw_motion_octrees.rs \
      ./src/raw_sclera_red_canny.rs \
      ./src/sam31_outer.rs \
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
  --glob '!Cargo.lock' --glob '!scripts/audit-tree.sh' .; then
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
