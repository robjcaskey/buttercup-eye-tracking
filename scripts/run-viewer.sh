#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

cargo_features=${BUTTERCUP_CARGO_FEATURES:-}

# The UI always exposes SAM31 in its segmentation cycle, so the normal viewer
# must carry the native Rust/tch implementation too.  Keep the sizeable model
# and LibTorch/CUDA runtime out of the source tree under the checked `data`
# link.  Setting BUTTERCUP_ENABLE_SAM31=0 remains useful for lightweight
# development builds that never select SAM31.
if [[ ${BUTTERCUP_ENABLE_SAM31:-1} != 0 ]]; then
  case ",${cargo_features// /,}," in
    *,sam31,*) ;;
    *) cargo_features="${cargo_features:+$cargo_features,}sam31" ;;
  esac

  sam31_runtime=${BUTTERCUP_SAM31_RUNTIME:-$project_dir/data/runtime/libtorch-2.9.0-cu128}
  export LIBTORCH=${LIBTORCH:-$sam31_runtime/torch}
  export LIBTORCH_CXX11_ABI=${LIBTORCH_CXX11_ABI:-1}
  if [[ ! -f $LIBTORCH/lib/libtorch.so || ! -f $LIBTORCH/lib/libtorch_cuda.so ]]; then
    printf 'Buttercup SAM31 native LibTorch runtime is unavailable: %s\n' "$LIBTORCH" >&2
    exit 1
  fi

  torch_library_path=$LIBTORCH/lib
  nvidia_runtime=${BUTTERCUP_NVIDIA_RUNTIME:-$sam31_runtime/nvidia}
  for directory in "$nvidia_runtime"/*/lib; do
    [[ -d $directory ]] || continue
    torch_library_path="$torch_library_path:$directory"
  done
  export LD_LIBRARY_PATH="$torch_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  # The shared-feature graph preserves the original detector and lets the
  # pupil prompt reuse its image encoder. Older exported graphs still work;
  # the worker detects the optional method once during warmup.
  sam31_default_model=$project_dir/data/models/sam31_semantic_video_shared_features_u8.pt
  if [[ ! -f $sam31_default_model ]]; then
    sam31_default_model=$project_dir/data/models/sam31_semantic_video_features_u8.pt
  fi
  export BUTTERCUP_SAM31_MODEL=${BUTTERCUP_SAM31_MODEL:-$sam31_default_model}
  if [[ ! -f $BUTTERCUP_SAM31_MODEL ]]; then
    printf 'Buttercup SAM31 promptable graph is unavailable: %s\n' "$BUTTERCUP_SAM31_MODEL" >&2
    exit 1
  fi
  export BUTTERCUP_SAM31_PROMPT_BUNDLE=${BUTTERCUP_SAM31_PROMPT_BUNDLE:-$project_dir/data/models/sam31_semantic_prompts_cuda_bf16.pt}
  if [[ ! -f $BUTTERCUP_SAM31_PROMPT_BUNDLE ]]; then
    printf 'Buttercup SAM31 semantic prompt bundle is unavailable: %s\n' "$BUTTERCUP_SAM31_PROMPT_BUNDLE" >&2
    exit 1
  fi
  export BUTTERCUP_SAM31_TRACKER_BUNDLE=${BUTTERCUP_SAM31_TRACKER_BUNDLE:-$project_dir/data/models/sam31_tracker_weights.pt}
  if [[ ! -f $BUTTERCUP_SAM31_TRACKER_BUNDLE ]]; then
    printf 'Buttercup SAM31 native tracker bundle is unavailable: %s\n' "$BUTTERCUP_SAM31_TRACKER_BUNDLE" >&2
    exit 1
  fi
fi

cargo_profile=${BUTTERCUP_CARGO_PROFILE:-live}
cargo_args=(run --profile "$cargo_profile")
if [[ -n $cargo_features ]]; then
  cargo_args+=(--features "$cargo_features")
fi

# The in-window SAM PROMPT editor invokes this sibling directly to encode a
# replacement text row without Python or a nested Cargo process. Cached builds
# make this effectively free after source changes settle.
if [[ ${BUTTERCUP_ENABLE_SAM31:-1} != 0 ]]; then
  cargo build --profile "$cargo_profile" --features "$cargo_features" \
    --bin buttercup_sam31_prompt_bundle
fi

exec cargo "${cargo_args[@]}" -- "$@"
