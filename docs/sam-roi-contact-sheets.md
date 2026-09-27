# Reusable SAM3.1 RAW ROI contact sheets

For the **pink waterfall lines bending into the pupil**, use
[the pink waterfall guide](pink-waterfall-contact-sheets.md). This SAM guide
produces segmentation comparisons, not that illustrative overlay.

Ask an agent: **“Follow docs/sam-roi-contact-sheets.md; ask SAM for [region]
on these RAW image filenames: [list].”** The native Rust
command below performs selection, export, inference and contact-sheet rendering.
New wording goes in JSON; no Rust edits or new per-prompt modes are needed.

## Input comes first: use the supplied RAW filenames

The agent's first-choice input is an explicit list of native RAW image filenames
from the user. Preserve that list and its order. Do not silently substitute
previous comparison images, choose different recordings, discard difficult
images, or fill missing inputs with random examples. A previously supplied list
or explicitly requested frozen comparison still counts; do not ask for it again.

Resolve every filename against its native capture metadata before running.
Check the RAW hash, dimensions, stride, pixel format, recording identity, and
frame identity. A recording's `subject-right.raw10` or `subject-left.raw10` may
contain many exposures, and many recordings reuse those names. In that case a
filename alone is insufficient: retain the archive/recording, stream entry and
frame sequence or byte offset/length. Ask for the missing identity if it cannot
be recovered unambiguously; never guess the exposure or decode format. A PNG,
screen capture, or existing rendered overlay is not a substitute for native RAW.

**Current command boundary:** `roi-contact-sheet` accepts a metadata-backed
frozen `selection` directory, not a bare filename-array JSON field. Resolve a
supplied filename list to that selection and verify its ordered RAW identities
before inference. An existing frozen selection is reusable only if it matches
the requested files/exposures. If an arbitrary list has no prepared selection,
explicit-list preparation must be implemented in the shared native selector
before this command can handle it; do not claim unsupported direct-file input
works or silently switch to random mode. The current layout uses six originals
and six matched blurred copies; agree on a batch if the supplied list differs.

If **no input list or frozen comparison is provided**, offer a choice, for example:

> “Would you like me to choose six reproducible random RAW eye images from the
> SAM-segmented, area-consistent corpus, preferably from different recordings,
> and show each beside a blurred copy?”

Do not run random selection merely because filenames are absent. An explicit
request for a random sample, or acceptance of this offer, is authorization to
use the procedure below; no repeated confirmation is needed. Record the chosen
seed, sampling policy and exact source identities. The eligible set excludes
grossly inconsistent frontal-equivalent iris area before this prompt experiment;
it is not a set of human-verified segmentations or an unrestricted corpus draw.

## Quick start: an explicitly requested frozen comparison

Work from the Buttercup repository root. Put this request under `outputs`
(for example `outputs/my-roi-request.json`), replacing the title and six literal
prompts as appropriate:

```json
{
  "title": "areas around tiny eyeball veins",
  "selection": "outputs/roi-pupil-prompt-selection-v1",
  "prompts": [
    "area around tiny tiny veins on the eyeball",
    "sclera around tiny blood vessels",
    "white of the eye with tiny veins",
    "eye surface around fine red veins",
    "sclera containing fine blood vessels",
    "white tissue around tiny eye veins"
  ]
}
```

Prepare the existing native runtime and build when source has changed:

```bash
export LIBTORCH="$(pwd)/data/runtime/libtorch-2.9.0-cu128/torch"
export LIBTORCH_CXX11_ABI=1
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$HOME/.local/bin/agent-coord" list
"$HOME/.local/bin/agent-coord" run \
  --owner sam-roi-build --resource 'cpu=16-23;memory-bandwidth=*;block=*' \
  --mode shared --sensitivity low --priority 10 --ttl 600 \
  --note 'Build native SAM ROI contact-sheet command' -- \
  taskset -c 16-23 cargo build --profile live --no-default-features \
    --features sam31 --bin buttercup_calibration_sign -j4
```

Inspect current resource claims and choose suitable cores before running. The
following is one command for all remaining computation; use a **new** output
directory each time. Save its stdout/stderr as a sibling log if desired.

```bash
"$HOME/.local/bin/agent-coord" run \
  --owner sam-roi-sheet --resource 'cpu=16-23;memory-bandwidth=*;block=*' \
  --mode shared --sensitivity low --priority 10 --ttl 1800 \
  --note 'SAM3.1 CPU prompt comparison on matched RAW ROIs' -- \
  taskset -c 16-23 data/target/live/buttercup_calibration_sign \
    roi-contact-sheet outputs/my-roi-request.json outputs/my-roi-sheet-v1
```

The main image is `outputs/my-roi-sheet-v1/results/contact-sheet.png`.
Prior runs took roughly two minutes for a fresh prompt-specific CPU export and
two minutes for inference/rendering on twelve images. New corpus selection adds
RAW reading/scoring time. These are observations on this host, not guarantees.

## New random corpus sample, when requested or accepted

After the user requests or accepts a random draw, replace `selection` with these
fields; keep `title` and `prompts`:

```json
{
  "area_first": "outputs/roi-area-first-focus-v1",
  "seed": "my-new-sample-2026-09-18-v1",
  "sampling": "matched-random"
}
```

This fragment belongs inside the request, not in a separate invocation.
Exactly one of `selection` and `area_first` is required. New selections require
an explicit nonempty seed. Changing the seed draws another reproducible sample.
Reusing a completed selection is preferable when comparing wording: changing
both images and prompts confounds the comparison.

Sampling modes:

| Mode | Selection |
| --- | --- |
| `matched-random` (new-command default) | Six random unique RAW exposures across the entire admitted pool, each followed by its Gaussian-blurred copy. |
| `matched` | Six exposures from the sharpness proxy's top quintile, each paired with its blurred copy; reproduces the original experiment's policy. |
| `natural` | Six recorded bottom-quintile and six recorded top-quintile exposures, without added blur; these are **not matched pairs**. |

All modes first keep **SAM-only, area-admitted** rows from a completed
`buttercup-area-first-focus-v1` run and deduplicate by RAW hash. This deliberately
preserves the request to filter for consistent frontal-equivalent iris area
before this experiment. It is **not random sampling of the unrestricted corpus**.
The original pool was 1,617 exposures across 15 recordings from one person.
The eligible pool can differ in a later area-first run.

The seeded order is SHA-256 of `seed:group:RAW-hash`. Selection prefers distinct
recordings, with a second pass only if there are too few; thus it is not a
uniform draw over every exposure. Selection happens before new prompt inference.
The sharpness proxy is contrast-normalized gradient energy after mild smoothing,
not a measured optical focus label. `matched-random` does not compute or filter
on it: only the selected exposures are decoded, avoiding a full-corpus image pass.
Historical `sharp-*` filenames mean the originals in matched-random runs, not
that those images have been proven sharp. The code currently requires at least
30 admitted unique exposures and always produces 12 rows.

## Reuse a verified prompt export for speed

Add `"export": "outputs/roi-vein-region-export-v1"` to the first example to
reuse that exact prompt set. The prompt strings **and their order must match**.
An export embeds its text prompts; do not relabel old masks or edit its receipt
to pretend it evaluated new wording. For new wording, omit `export`.

Without `export`, the command regenerates the CPU graph from the pinned official
checkpoint, defaulting to `data/models/sam31_multiplex.pt`. An optional
`checkpoint` path selects another copy of those same pinned bytes; it is not
an arbitrary-model override. Do not provide both `export` and `checkpoint`.
The exporter runs the native bootstrap DAG preflight, strict checkpoint loading,
source checks and export/reload parity checks. Cached exports still undergo
checkpoint/revision, adapter, CPU/FP32, prompt and model-hash validation before
inference. Cache reuse is not a new cold-bootstrap proof or model training.

No camera connection, live viewer, external SAM server or GPU is needed.
SAM's pinned upstream Python API is used only by the existing export adapter;
selection, RAW decoding, inference orchestration and reporting are Rust.
Do not edit the checkout during an export: its source-stamp check will reject it.

## What is actually shown

- Native source RAW10 is reopened and hash-checked. The existing `quad_rgb`
  adapter produces 420×280 RGB, then the model receives antialiased 1008-square
  input normalized as `(RGB/255 - 0.5)/0.5`. This is not a new enhancement filter.
- Matched blur is Gaussian sigma **3 adapter pixels**, applied to each RGB
  channel with clamped edges. Originals are unchanged. Pair labels and hashes
  preserve that distinction.
- Each of six literal prompts produces 200 query scores. The highest-scoring
  query is shown, including empty and low-score results. No query is chosen
  because it looks more anatomically plausible.
- Its mask logits are resized to the adapter image, sigmoid-transformed and
  quantized; values at least 128/255 define the cyan boundary. Model scores
  are **not accuracy or calibrated probabilities**. The orange LOW label below
  0.5 is informational and does not hide the mask.
- Generic requests display regions, not invented point predictions. Centroids
  remain in the numeric records but are not drawn. The old pupil-center mode
  drew pink computed mask centroids; those were never native SAM point outputs.
- No gaze target, fitted iris ellipse, anatomical oracle, or human landmark
  selects the new response. No training happens. Sample admission remains
  model-derived, which must be disclosed separately.

## Outputs and review procedure

The command refuses to overwrite an output directory and confines runtime data
to the checked `data`/`outputs` links. It writes:

- `request.json`: normalized request with absolute existing input paths.
- `prompts.json` and `run.json`: literal prompts, input/runner hashes, timing,
  selected paths and completion state. A failed run stays incomplete.
- `selection/`: frozen sampling manifest and input preview, only for a new draw.
- `export/`: pinned export and provenance, only when not using a cache.
- `results/contact-sheet.png`: all twelve rows; matched originals immediately
  above their blurred copies. Six prompts always keep the same column order.
- `results/contact-sheet-page-1.png` and `...-2.png`: readable page splits.
- `results/pair-1.png` through `pair-6.png`: matched closeups.
- `results/sharp-contact-sheet.png`, `blurry-contact-sheet.png`: group views.
- `results/frames.jsonl`, `summary.json`, RGB bytes and masks: exact source
  identities/hashes, all query scores, selected masks and inference receipt.

Before reporting success, verify completion and 12×6 responses; for comparisons
verify RAW and RGB hashes against the previous run. Inspect **both pages**, then
closeups if needed. Record observations in `results/visual-review.md`: which
prompts find the intended tissue, failures, blur-induced changes, and missing
human labels. Distinguish mask consistency from anatomical correctness. Do not
claim localization accuracy, reliable vein detection or gaze-sign correctness
from these pictures alone. Do not hide failed masks or swap samples after seeing
their predictions.

To show an explicitly requested sheet, use the installed `queued-open` entry
point and await its result. Use `--placement new` for a deliberately separate
review workspace; never focus/approve/dispatch it yourself. Say “queued” until
the helper confirms “opened”. For optional results, use `attention-yield` per
the home AGENTS.md instead. Inspection via `view_image` alone does not open a
desktop window for Rob.

## Existing comparisons and source entry points

The shared original selection is `outputs/roi-pupil-prompt-selection-v1`.
Existing result directories are `roi-pupil-prompt-results-v1`,
`roi-lower-eyelid-arc-results-v1`, `roi-sclera-prompt-results-v1`,
`roi-black-pupil-results-v1`, `roi-skin-results-v1`,
`roi-eye-corner-results-v1` and `roi-vein-region-results-v1` under `outputs`.
`roi-sclera-arc-results-v2` is a separate boundary-fitting experiment, **not**
part of this generic mask-only command.

Implementation:

- `src/bin/buttercup_calibration_sign/contact_sheet.rs`: request validation and orchestration.
- `.../pupil_prompts.rs`: shared selection, inference records and rendering.
- `.../roi_anatomy.rs`: verified CPU loader and shared input/inference contract.
- `.../sam_export.rs`: pinned official SAM3.1 export recipe and checks.

Keep future improvements in these shared paths rather than writing another
one-off contact-sheet generator. Existing named prompt modes remain available.

## Validation of the reusable command

The development check in `outputs/roi-contact-sheet-automation-check-v2`
completed a new `matched-random` draw from 1,617 admitted exposures across six
selected recordings and generated all 72 responses with a verified cached
export in about 133 seconds. Both pages were visually reviewed. Shared resource
claims overlapped during this run, so the timing is not an isolated benchmark.
Same-seed RAW/RGB identities matched exactly, a different seed changed the draw,
and the legacy default seed still reproduced the original comparison inputs.
A mismatched cached prompt set was rejected before output creation. The native
build, source-tree audit and whitespace checks passed. This validates the
workflow; segmentation correctness remains a separate, visibly imperfect result.
