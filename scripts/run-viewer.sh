#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

camera_address=${BUTTERCUP_CAMERA_ADDRESS:-192.168.88.10:5001}
vcm_address=${BUTTERCUP_VCM_ADDRESS:-192.168.88.10:5002}
origin=${BUTTERCUP_SENSOR_ORIGIN:-0,3250}
left=${BUTTERCUP_LEFT_EYE:-3448,3340}
right=${BUTTERCUP_RIGHT_EYE:-4492,3480}
eye_size=${BUTTERCUP_EYE_SIZE:-384x256}
window_size=${BUTTERCUP_SENSOR_WINDOW:-8000x576}
focus_eye=${BUTTERCUP_FOCUS_EYE:-auto}
segmentation=${BUTTERCUP_SEGMENTATION_MODE:-native}
rough_center=${BUTTERCUP_ROUGH_CENTER_MODE:-iris-guided}
control_socket=${BUTTERCUP_CONTROL_SOCKET:-/tmp/buttercup-eye-control.sock}

args=(
  --camera "$camera_address"
  --vcm "$vcm_address"
  --origin "$origin"
  --left "$left"
  --right "$right"
  --eye "$eye_size"
  --window "$window_size"
  --focus-eye "$focus_eye"
  --segmentation "$segmentation"
  --rough-center "$rough_center"
  --control "$control_socket"
)

if [[ -n ${BUTTERCUP_TRACKING_FRAME:-} ]]; then
  args+=(--tracking "$BUTTERCUP_TRACKING_FRAME")
fi
if [[ -n ${BUTTERCUP_SAM31_MODEL:-} ]]; then
  args+=(--sam31-model "$BUTTERCUP_SAM31_MODEL")
fi

cargo_args=(run --release)
if [[ -n ${BUTTERCUP_CARGO_FEATURES:-} ]]; then
  cargo_args+=(--features "$BUTTERCUP_CARGO_FEATURES")
fi

exec cargo "${cargo_args[@]}" -- "${args[@]}" "$@"
