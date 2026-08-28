#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

cargo_args=(run --release)
if [[ -n ${BUTTERCUP_CARGO_FEATURES:-} ]]; then
  cargo_args+=(--features "$BUTTERCUP_CARGO_FEATURES")
fi

exec cargo "${cargo_args[@]}" -- "$@"
