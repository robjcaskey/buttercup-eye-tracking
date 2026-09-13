# Butter Obelisk: RAW-native Eye Student experiment

## Inference device selection

`--obelisk-device auto|cpu|gpu` selects the inference device for the fixed-label
student/Obelisk worker; `--eye-student-device` is an alias. Default `auto` uses
CUDA device 0 when available, otherwise CPU. `BUTTERCUP_OBELISK_DEVICE` supplies
the same setting to native replay tools; an explicit viewer CLI option wins.
Forced GPU fails clearly if CUDA is unavailable. Forced CPU does not create a
CUDA stream, probe CUDA availability, or synchronize CUDA. Its LibTorch CPU
intra-op pool is bounded to two threads. Worker status identifies CPU/GPU.

This selects inference placement, not a different architecture or checkpoint.
SAM and offline CUDA training recipes are unchanged. The present `sam31` build
still links the installed CUDA-enabled LibTorch runtime: CPU inference on a
machine without a usable GPU is supported, not a dependency-free CPU-only build.
No fallback conceals a corrupt model or inference error.

Validation: device-selection unit test; two paired RAW exposures (four ROI
frames) through forced CPU with CUDA hidden, auto with CUDA hidden, forced GPU,
and auto with GPU available. All four paths retained identical admission/pupil
decisions; maximum CPU/GPU fitted-center difference was 0.0038 sensor pixels.
Forced unavailable GPU and misspelled CLI values failed as expected. This is a
smoke test, not a CPU throughput or whole-corpus equivalence claim.

`buttercup-eye-mask-unet-raw16-v1` predicts the same six masks as the RGB
student. It does not regress gaze or a completed ellipse. Existing RAW admission,
partial-outline/flat-tire fitting, pupil alternatives and subsequent geometry
remain authoritative. RGB weights remain usable; defaults are not replaced.

The human-facing name is **Butter Obelisk** (one b and one l in “obelisk”).
Stable internal architecture IDs and the existing `eye-student` acquisition-mode
identifier are retained for compatibility. The worker identifies RAW explicitly.
The G selector, ROI headings and analysis status use the validated asset's
human-facing name: Butter Obelisk for RAW, Eye Student RGB for the original
model. `VIEW STATUS` adds `global_gaze.detector_display` while keeping the stable
`detector` identifier. Model identity is cached once per process, not read from
disk on every repaint. Existing sessions need a restart to show a newly built UI;
do not discard an unsaved calibration merely to refresh a label.

## Sensor and coordinates

`sam31_student_raw.rs` defines `quad-rggb16-linear-code-f32-v1`: native
`RAW10_LE40_1X1`, with the 4x4 `RRGG/RRGG/GGBB/GGBB` pattern, not 2x2 Bayer.
Channel `4*(sensor_y mod 4)+(sensor_x mod 4)` is one photosite phase. Each phase
is interpolated on its own lattice into a registered 192x128 floating-point
field covering coordinates `(output+0.5)*source_size/output_size-0.5`. Crop
origin selects the lattice, including odd origins. Borders replicate within that
phase. No demosaic, white balance, tone curve or uint8 conversion precedes the
network. Packed input uses the checked RAW10 decoder; live workers reuse their
unpacked source buffer.

Normalization divides by 1023, the nominal digital maximum, **not a measured
black/white calibration**. This corpus does not supply measured black level or
exposure/gain calibration; those fields remain null. Saturation is unrecoverable.
Each phase samples every four native pixels; interpolation is not new detail.
The tensor is 393,216 floats (1.57 MB), versus 294,912 uint8 values for RGB.
Less preparation does not imply faster transfer or inference.

The RAW stem takes 16 channels at stride one, versus RGB's three at stride two.
Later feature-map sizes and six 384x256 outputs are unchanged. Parameter counts
are 214,566 RAW versus 213,162 RGB. Manifests require the exact contract and shape;
incompatible assets fail closed. Co-registered phase fields are affine-warped or
flipped without changing color identity. Labels receive the same normalized
affine at their own resolution; unavailable pixels receive zero loss. Shadow
and gain augmentation model appearance, not calibrated sensor noise.

## Native API and recipes

The library exports `raw_student_input::{Source, prepare, from_packed, contract}`
and `sam31_outer::{Client, RawFrame, ...}`. With `sam31`,
`sam31_outer::student::Model::infer_source` returns six logits. Select a variant
with `BUTTERCUP_EYE_STUDENT_MODEL=/absolute/path/raw.ot` before starting the Eye
Student client. The manifest chooses preprocessing, not the preview. Workers
are not hot-swapped; existing eye-mode/source-epoch invalidation still applies.
This is native CUDA/LibTorch, not CPU/Wasm or a personal refinement model.

Build in the installed LibTorch/CUDA environment, with a host `agent-coord` lease:

```sh
cargo build --offline --profile live --features sam31 --bin buttercup_eye_student --bin buttercup_report_eye_student
```

With `RUN` a new directory under `outputs`, `INDEX` a native hash-verified
selection, and pinned `BUTTERCUP_SAM31_MODEL` / `BUTTERCUP_SAM31_PROMPT_BUNDLE`:

```sh
data/target/live/buttercup_eye_student export "$INDEX" "$RUN/teacher"
BUTTERCUP_EYE_STUDENT_TRAIN_ARCHITECTURE=rgb BUTTERCUP_EYE_STUDENT_TRAIN_AUGMENTATION=shadow-crop-v1 data/target/live/buttercup_eye_student train "$RUN/teacher" "$RUN/rgb.ot" 120
BUTTERCUP_EYE_STUDENT_TRAIN_ARCHITECTURE=raw16-v1 BUTTERCUP_EYE_STUDENT_TRAIN_AUGMENTATION=shadow-crop-v1 data/target/live/buttercup_eye_student train "$RUN/teacher" "$RUN/raw.ot" 120
BUTTERCUP_EYE_STUDENT_MODEL="$RUN/rgb.ot" BUTTERCUP_EYE_STUDENT_TIMING=1 data/target/live/buttercup_eye_student replay-paired "$EVAL_INDEX" student "$RUN/rgb-replay.jsonl"
BUTTERCUP_EYE_STUDENT_MODEL="$RUN/raw.ot" BUTTERCUP_EYE_STUDENT_TIMING=1 data/target/live/buttercup_eye_student replay-paired "$EVAL_INDEX" student "$RUN/raw-replay.jsonl"
data/target/live/buttercup_report_eye_student --compare "$RUN/rgb-replay.jsonl" "$RUN/raw-replay.jsonl"
```

Also replay the current RGB weights and SAM on identical exposures/settings.
`--compare BASE CAND [LABEL_DATASET]` matches exact RAW hashes, ROI metadata and
source clocks. Canonical reviewed labels are optional, read **after inference**.
Whole source-clock sessions and RAW deduplication govern training splits. The
730-frame pilot uses reused development holdouts, not a pristine new-user test.
Separately replay the shadow/glasses/reframe clip and canonical label sources;
report training overlap instead of calling all reviewed labels held out.

Teacher-v2 stores both representations from the same exposure with fresh six-
prompt SAM masks; unavailable targets have zero weight. Ellipse rasters are not
targets. Native bootstrap DAG preflights precede export and training. Source
stamps bracket both; index, payload, SAM and weight hashes are recorded. This
cold path rejects initial custom weights and old unstamped caches for training:
historical warm-start recipes in the RGB notes now fail closed. Cached evaluation
also requires the same stamped source tree. Receipts pin supplied SAM assets,
but do not independently prove their original upstream acquisition/export chain.
A structural DAG certificate is not proof of successful cold execution.

## Interpretation

`BUTTERCUP_EYE_STUDENT_TIMING=1` synchronizes CUDA for separate preparation,
H2D, forward, mask materialization and remaining downstream timings. First use
per eye is warmup. Whole-worker time starts at submission, excludes archive IO/
unpacking, and is not camera-to-cursor latency or offered-load throughput.
Device synchronization can include other work: record lease token, honored
soft-exclusivity and contention. Interleave/repeat comparisons; separate stage
quantiles are not additive.

Report admissions, dropouts, pupil-with-outer coverage and independent reviewed
rim localization together, by eye and crop movement. Displacement is not error.
SN-FEIDA requires independently supplied scale: missing scale stays missing,
never replaced by candidate radius. Approximate paired-label midpoints are not
guaranteed surface apexes. Missing labels/timing/scale and reused sessions limit
every conclusion. Stable wrong ellipses and SAM agreement do not prove accuracy.
Promotion, abrupt-lighting robustness and cross-user readiness require evidence,
not assumptions from this single-user pilot.

## First matched run: 2026-09-12

Conclusion: keep Obelisk as a usable experimental option, **not a default
replacement yet**. It improves the troublesome shadowed eye and pupil coverage
on the sparse benchmark, but loses some pupil coverage on the other eye and
has a meaningful low-light localization regression. Overall reviewed-label
localization versus the current student is effectively unchanged.

Artifacts and exact shell recipes are in
`outputs/raw-student.DUYOAP/{run-experiment,replay-experiment,report-experiment,render-review}.sh`.
All training, inference, scoring and native RAW rendering use Rust; FFmpeg only
encodes the comparison movie and adds panel captions. Nothing was trained from
rendered previews or human-label evaluation results. No weights were promoted.

### Data, provenance and limitations

The actual input is `outputs/eye-student.FQGerU/pilot-frames.jsonl`: 730 exposures
from 64 source-clock sessions, split 574/84/72 training/validation/test across
51/7/6 sessions. Both new models started from seed 17091 with `shadow-crop-v1`
and 120 epochs, using the same fresh SAM teacher-v2 cache. RGB selected epoch
90, RAW epoch 75, by validation iris/pupil mask IoU, not test or human labels.
Training took 59.8 seconds RGB and 73.8 seconds RAW. Unknown heads/pixels had
zero supervision. Checkpoint parameters were 213,162 and 214,566 respectively.
The augmentation recipe and seed match, not necessarily every random transform:
different network initialization shapes consume different RNG draws. RAW gain/
offset units also differ from display RGB. This tests the complete RAW-input
variant, not bit depth alone under perfectly paired perturbations.

Cold-run source revision was `eda3aa0981b9890adc02de56dfa908d52dcd0dc7`, with
dirty-tree fingerprint
`ad92bb126f8585aa94320efed334ef1d3aa2d40198d15184826e16df3bb94ac4`.
Export and both training runs passed matching before/after source stamps and
native DAG preflights. The model SHA-256 values are:

- RGB control: `dbcba07a0b81fcc37f1a7d50e7b69aa2679fc9ea3d3f045b97a8e43196ef660e`.
- Obelisk RAW: `22dcd42ab3bdb7dc2a966c940351c2cc7cfc0e78954b85e740848531de334fc9`.

`assets.sha256` pins the current RGB checkpoint and all supplied SAM assets.
This work subsequently added diagnostic reporting, rendering, naming and
documentation; the completed training receipts retain
their actual source fingerprint, not a fabricated final-tree identity. New
current-tree training must regenerate its teacher cache. This experiment is a
successful cold training run with supplied SAM assets, not a full independent
SAM acquisition/export audit or cross-user bootstrap demonstration.

An initial export accidentally selected the expanded 7,741-frame inventory.
It was stopped and its incomplete `teacher/` output was **not used**. The
completed, verified cache is `teacher-pilot/`.

The shadow sequence is 129 simultaneous eye pairs (258 exposures, 11.385 seconds,
source sequences 4395–4523) from `both-eyes-1789218305-631052561.tar`, with no
RAW-hash or source-clock training overlap. It includes 18 per-eye crop-change
transitions, including the large 44-by-70-pixel move. It has rough recorded
scale support for 175 exposures but no independent reviewed rim labels. The
scale bounds are wide and not calibrated probabilities. The benchmark's
84 sparse validation exposures have no independent scale. These are reused
single-user development data, not pristine holdouts or new-user validation.

The canonical label evaluation is 16 reviewed native exposures prepared by the
native label tool at `outputs/bootstrap-rust-validation.waPTE4/rust-human/`.
All identities matched exactly. One exposure, low-light sequence 10163, occurs
in training; related-session overlap cannot be excluded merely because labels
were exported to standalone RAW files. Scale is absent, and those conservative
label groups have no attested temporal clock. No temporal accuracy is claimed
from them. Paired midpoints, legacy rims and triplet apexes are not equally
precise anatomical truth; the metric is unweighted nearest-rim distance to a
2048-edge ellipse, not gaze error in degrees or pupil accuracy.

### Coverage and localization

Counts below are fresh admitted outer fits / pupils accompanying admitted
outer fits, not proof of anatomical correctness. Every backend was replayed
on identical source exposures using current shared geometry settings.

| Subset | Current RGB | Fresh RGB control | Obelisk RAW | SAM3.1 |
| --- | ---: | ---: | ---: | ---: |
| 84 validation exposures | 38 / 21 | 35 / 18 | 38 / 24 | 44 / 25 |
| 258 shadow exposures | 243 / 240 | 240 / 239 | 245 / 230 | 241 / 204 |
| 16 reviewed exposures | 13 / 5 | 11 / 6 | 15 / 8 | 10 / 6 |

On the shadow clip's subject-right eye, current pupil coverage 122/129 falls to
110/129 with RAW. Subject-left improves 118/129 to 120/129. The latter's iris
is visibly no longer pulled up into the eyelid on inspected sequences 4449 and
4523; SAM and RAW agree closely there. Sequence 4479 after reframing also looks
consistent. This visual spot check is not a replacement for human labeling.

Pairwise human-label results use common accepted exposures; **the subsets differ
and their means must not be compared across rows**:

| Pair | Common frames / rim points | Baseline mean error | RAW mean error |
| --- | ---: | ---: | ---: |
| Current RGB vs RAW | 13 / 167 | 5.153 px | 5.136 px |
| Fresh RGB control vs RAW | 11 / 148 | 4.028 px | 5.161 px |
| SAM vs RAW | 10 / 133 | 5.867 px | 3.789 px |

Failures remain: low-light sequence 10124 worsens from 8.87 px to 15.29 px;
night sequence 533 worsens from 3.02 px to 4.61 px. Sequence 549 improves
8.06 px to 2.85 px; the difficult 569 improves 16.32 px to 12.94 px but remains
poor. RAW admits extra labeled exposures, but admission alone is not accuracy.

Only the registered phase-preparation API accepts general rectangular sizes;
the existing downstream ellipse projection assumes the live 3:2 crop aspect.
Corpus inference here uses 384x256 and 420x280; arbitrary-aspect geometry was not
validated. Odd-origin and odd-size phase sampling were separately unit-tested.

On the 72 test exposures, SAM-mask agreement improves for cold RAW versus cold
RGB: iris IoU 0.8034 to 0.8344 (35 supported frames), pupil 0.6434 to 0.6809
(27 supported frames). This is teacher agreement, not human-label accuracy.

### Area and motion

For the 163 common admitted shadow frames with independent scale, SN-FEIDA
median is 104.86 mm² current versus 103.91 mm² RAW; p95 is 166.38 versus
109.04 mm². On the exact same 82 eligible adjacent transitions, absolute area
change p95 falls 40.85 to 2.95 mm² (median 1.06 to 0.49). Among 16 commonly
admitted crop-change transitions, sensor-space center displacement median falls
1.97 to 1.23 px, but p95 remains 31.06 versus 24.76 px. These are consistency
diagnostics, not known localization/motion errors; real eye/head motion and
rough scale uncertainty remain. Both sides must be fresh and clock-attested;
rejected samples are not removed to bridge gaps. Nothing is normalized by the
candidate's own radius. The pupil dropouts and labeled regressions still count.

### Timing

RTX 5080, 16,303 MiB, driver 595.91.07, LibTorch 2.9.0 CUDA 12.8; two ROI
workers, synchronized student timings, first result per eye excluded. The
replay claim was `cpu=*;memory-bandwidth=*;block=*;gpu=0;gpu-memory=0`, token
`l18d4be3874932416-5a898`, soft-exclusive **honored=false** because presence
services retained shared CPU/memory claims. The viewer was stopped and no
other GPU inference was observed. Treat timings as local indicative results,
not exclusive-host guarantees. All lease receipts are retained with the run.

| Validation worker, first pass | Current RGB median / p95 | RAW median / p95 |
| --- | ---: | ---: |
| CPU input preparation | 11.255 / 14.494 ms | 0.993 / 3.611 ms |
| Synchronized H2D | 0.049 / 0.093 ms | 0.117 / 0.151 ms |
| Synchronized neural forward | 0.539 / 0.681 ms | 0.551 / 0.759 ms |
| Mask materialization | 0.505 / 0.714 ms | 0.477 / 0.673 ms |
| Remaining downstream | 41.531 / 54.861 ms | 13.728 / 56.849 ms |
| Whole completion-paced worker | 53.5 / 69 ms | 15 / 59 ms |

Reversing order and restarting workers gives whole-worker medians 54.5 ms
current versus 16 ms RAW (p95 69/58 ms). SAM's first-pass median/p95 is
356.5/536 ms; fresh RGB's is 23/67 ms. Different proposals take different
downstream branches, so the entire worker difference is not all preprocessing.
The clean input-stage result is about 11.2 ms to 1.0 ms; forward inference is
essentially unchanged and RAW transfer is larger. Separate medians are not
additive, and these completion-paced paired replays measure neither offered-
load throughput nor camera-to-cursor latency.

For visual review, `subject-left-comparison.mp4` is source / current RGB /
Obelisk / SAM, with cyan pupils and gray rejected proposals. It shows all
129 subject-left exposures at approximately the mean source cadence, not exact
per-frame presentation intervals. `low-light-regression.png` preserves an
important failure alongside the successful examples.
