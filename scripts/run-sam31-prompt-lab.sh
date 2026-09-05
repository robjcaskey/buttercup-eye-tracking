#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

prompt_file=${1:-/tmp/prompt}
manifest=${2:-$project_dir/scripts/prompt-lab-troublesome-3.txt}
if (($# >= 3)); then
  output_dir=$3
else
  output_dir=$project_dir/data/outputs/sam31-prompt-lab-$(date +%Y%m%d-%H%M%S)
fi

[[ -f $prompt_file ]] || {
  printf 'Prompt file unavailable: %s\nPut one prompt per line in it.\n' "$prompt_file" >&2
  exit 1
}
[[ -f $manifest ]] || { printf 'RAW10 manifest unavailable: %s\n' "$manifest" >&2; exit 1; }

sam31_runtime=${BUTTERCUP_SAM31_RUNTIME:-$project_dir/data/runtime/libtorch-2.9.0-cu128}
export LIBTORCH=${LIBTORCH:-$sam31_runtime/torch}
export LIBTORCH_CXX11_ABI=${LIBTORCH_CXX11_ABI:-1}
torch_library_path=$LIBTORCH/lib
nvidia_runtime=${BUTTERCUP_NVIDIA_RUNTIME:-$sam31_runtime/nvidia}
for directory in "$nvidia_runtime"/*/lib; do
  [[ -d $directory ]] || continue
  torch_library_path="$torch_library_path:$directory"
done
export LD_LIBRARY_PATH="$torch_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

checkpoint=${BUTTERCUP_SAM31_CHECKPOINT:-$project_dir/data/models/sam31_multiplex.pt}
bpe=${BUTTERCUP_SAM31_BPE:-$project_dir/data/models/sam31_bpe_simple_vocab_16e6.txt.gz}
export BUTTERCUP_SAM31_MODEL=${BUTTERCUP_SAM31_MODEL:-$project_dir/data/models/sam31_semantic_dynamic_u8.pt}
export BUTTERCUP_SAM31_PROMPT_BUNDLE=${BUTTERCUP_SAM31_PROMPT_BUNDLE:-$project_dir/data/models/sam31_semantic_prompts_cuda_bf16.pt}
[[ -f $checkpoint ]] || { printf 'SAM3.1 checkpoint unavailable: %s\n' "$checkpoint" >&2; exit 1; }
[[ -f $bpe ]] || { printf 'SAM3.1 BPE vocabulary unavailable: %s\n' "$bpe" >&2; exit 1; }

prompt_hash=$(
  sha256sum "$prompt_file" "$project_dir/src/sam31_text.rs" \
    "$project_dir/src/bin/buttercup_sam31_prompt_bundle.rs" | sha256sum | cut -c1-16
)
prompt_bundle=$project_dir/data/models/sam31_prompt_lab_${prompt_hash}.pt
if [[ ! -f $prompt_bundle ]]; then
  BUTTERCUP_SAM31_TEXT_CUDA=1 BUTTERCUP_SAM31_TEXT_BF16=1 \
    cargo run --profile live --features sam31 \
      --bin buttercup_sam31_prompt_bundle -- \
      "$checkpoint" "$bpe" "$prompt_bundle" --prompt-file "$prompt_file"
fi

exec cargo run --profile live --features sam31 \
  --bin buttercup-sam31-arc-trial -- \
  --prompt-lab "$output_dir" "$prompt_file" "$manifest" "$prompt_bundle"
