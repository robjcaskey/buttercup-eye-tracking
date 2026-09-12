# CUDA eye-mask student

`G` now has a separate `eye-student` entry immediately after `sam31`. It is
experimental and does not replace the SAM default. The detector is a small
six-head U-Net trained in Rust/LibTorch on CUDA from SAM3 pseudo-labels. It
predicts **2D semantic masks**, not measured 3D anatomy or directly supervised
gaze. The existing conic/contact/gaze solver still reconstructs the 3D result.

The six fixed labels are outer iris disk, iris annulus, pupil disk, visible
sclera, upper eyelid, and lower eyelid. An arbitrary newly typed text prompt
cannot be compiled into this fixed-vocabulary model; the UI directs the user
back to SAM for prompt editing and object search.

## Shared live contract

Only mask inference is replaced. The student runs in native CUDA workers with
the existing replaceable latest-frame mailbox and atomic stereo source groups.
It takes the exact same native RAW10 decoding and configured SAM preprocessing
(`mild-blur` for the initial weights), with a 384×256 RGB input. The pupil and
outer masks come from one forward pass on one exposure. Each eye has its own
worker and source/pupil history; CUDA streams keep them independent.

After inference, both detectors use the same ordered-contour extraction,
multi-section flat-tire exclusion, constrained conic fitting, pupil-component
selection, RAW ring/glare checks, and source-coordinate conversion. Student
outputs then enter the same ROI presence authority, scale support, joint conic
solver, camera-facing contact, sign history, laser, screen projection, and
calibration paths. `uses_mask_geometry()` names that shared contract rather
than checking for the SAM enum everywhere. The student remains a distinct
authority: changing `G` invalidates old source/pose/calibration authority.

The pupil head supplies three nested logit level sets (0 and ±1) to that same
RAW candidate selector. These are correlated alternatives from **one** exposure,
not three observations or extra votes. At most one pupil is selected. Outer
mask inference is unchanged. `BUTTERCUP_EYE_STUDENT_PUPIL_LEVELSETS=0` restores
the single-threshold ablation; it is not a relaxation of any admission gate.

An unused detector is resident but idle. Switching between loaded backends
does not synchronously destroy a CUDA worker in camera ingest. No SAM encoder,
prompt decoder, or mask-memory network executes for a student frame. Its
frame-local mask is not labeled as a video-memory prediction. Existing pupil
history may guide a new RAW fit, but a held result is not new evidence.

`CAND` in this mode is the student's mask candidate. Its displayed score is
mean sigmoid activation inside foreground, **not SAM's object score**, a
calibrated probability, or gaze accuracy. RAW/conic admission remains separate.

## Inspecting Student results without overlapping diagnostics

Student's F cycle separates presentation layers. Its initial view shows only
the selected mask fill. Other views isolate all candidate mask fills, the mask
outline, the fitted outer ellipse, pupil fits, de-flat-tire keep/reject evidence,
conic arcs, and the virtual contact. Full diagnostics remains last. There are
also separate clean views for the exact inference source and the latest live
ROI, so inference delay is not mistaken for a bad fit on a newer image.

Sparse views suppress unrelated points, circles, scale guides and laser
decorations. Source-review views draw the proposal's own RAW pixels and crop
dimensions, including after the live ROI moves or resizes. F changes no mask,
fit, source clock or gaze authority; G remains the global detector selector.
Global preview defaults and per-preview overrides work the same as for other
detectors (Shift+Tab to choose edit scope, Backspace to inherit again).
Linked Views also has Student-specific F entries for paired ellipse-only,
mask-outline and pupil-only inspection, alongside compare, contacts and timing.

## Reproducible training

Runtime artifacts stay beneath `data`/`outputs`; none are source-tree assets.
The initial work directory is `outputs/eye-student.FQGerU`. The selector started
from `outputs/dual-eye-joint.MiR1Iw/replay-inputs-v2/frames.jsonl`, which addresses
native RAW bytes inside uncompressed capture archives without copying archives.
It verifies the selected bytes against their SHA-256 records and deduplicates
identical RAW. Entire source-clock lineages, including both eyes, stay in one
partition. Recorded gaze, display targets, and human annotations are not inputs.

The bounded pilot selects 730 exposures from 64 sessions, at most six exposures
per eye/session: 574 training frames from 51 sessions, 84 validation frames from
seven sessions, and 72 test frames from six sessions. This is a specified
subset, **not the full corpus** and not an independent population study.

The teacher uses the native SAM detector, canonical prompt bundle, and shared
image features for fixed prompts. Teacher exports are frame-local detector
observations, not a replay of SAM's video-memory propagation. Outer/pupil
training targets are actual SAM masks selected by the same RAW-qualified
selector used for geometry. Unavailable targets have zero loss weight; they
are not silently turned into negative masks. Other semantic heads require a
finite SAM score of at least 0.5. No completed ellipse is rasterized as a label.

The first experiment used highest-scoring pupil masks instead of RAW-qualified
pupil candidates. Despite seemingly reasonable teacher-mask IoU, **zero pupil
fits passed the shared publication checks**. Those weights (`pilot.ot`) are a
rejected diagnostic, not a deployment candidate. This illustrates why mask
overlap alone cannot establish usefulness for gaze.

Training uses weighted binary cross entropy plus Dice loss, paired image/mask
affine augmentation, horizontal flips, and mild photometric augmentation.
The held-out validation set selects the checkpoint; test results do not enter
the loss or checkpoint criterion. Pseudo-labels inherit teacher mistakes,
especially uncertain lids/reflections, so this is not proof of anatomical
accuracy. The corrected label-selection experiment reuses the pilot holdout;
it is a development holdout, not an untouched final benchmark.

With the same LibTorch environment as `scripts/run-viewer.sh`:

```sh
cargo build --profile live --features sam31 --bin buttercup_eye_student
python3 scripts/prepare-eye-student.py SOURCE_INDEX.jsonl outputs/RUN/frames.jsonl \
  --per-eye-session 6 --max-sessions 64
data/target/live/buttercup_eye_student export outputs/RUN/frames.jsonl outputs/RUN/teacher
data/target/live/buttercup_eye_student train outputs/RUN/teacher outputs/RUN/model.ot 120
data/target/live/buttercup_eye_student evaluate outputs/RUN/teacher outputs/RUN/model.ot outputs/RUN/eval.jsonl
BUTTERCUP_EYE_STUDENT_MODEL=outputs/RUN/model.ot \
  data/target/live/buttercup_eye_student replay outputs/RUN/frames.jsonl student outputs/RUN/student-live.jsonl
data/target/live/buttercup_eye_student replay outputs/RUN/frames.jsonl sam outputs/RUN/sam-live.jsonl
# Same-clock pairs exercise both workers through the actual atomic group API;
# unmatched eyes remain single-eye submissions, never silently dropped.
data/target/live/buttercup_eye_student replay-paired outputs/RUN/frames.jsonl student outputs/RUN/paired.jsonl
python3 scripts/report-eye-student.py outputs/RUN/eval.jsonl \
  --replay outputs/RUN/student-live.jsonl --replay outputs/RUN/sam-live.jsonl
```

Resource-intensive commands require the host resource-coordination claim.
Record its predicate, token, and honored status for performance comparisons.
Use fresh output paths. Model manifests bind architecture, input shape, fixed
labels, and preprocessing; incompatible/missing assets fail explicitly.

The default model path is `data/models/eye_student_v1.ot`, with a sibling
`eye_student_v1.json` manifest. Override it with `BUTTERCUP_EYE_STUDENT_MODEL`.
An explicit `--segmentation eye-student` starts the student directly without
loading SAM weights; the optional native LibTorch/CUDA build is still required.

## Validation boundaries

### Initial qualified checkpoint, September 11, 2026

The measurements in this subsection predate enabling the three pupil level
sets. Keep them as the matched single-threshold baseline, not current defaults.

The installed experimental checkpoint is `qualified.ot`/`qualified.json` from
the work directory, copied without replacing prior models to
`data/models/eye_student_v1.ot` and its manifest. It has **213,162 parameters**,
866,637 serialized bytes (about 846 KiB), and was selected at epoch 35 of 120
using validation mask IoU. Weights SHA-256:
`b780957a199fe1c1a8b4171fd95371e72836e6ece296797c3101cdcc334cf3de`.

The corrected 72-frame development test partition has teacher/student outer
RAW admissions 35/40, with 29 common admissions, six teacher-only and 11
student-only unverified admissions. Accepted pupil counts are 27/26. On the
29 common outer fits, median center disagreement is 2.83 native pixels and
p95 is 15.18 pixels; those are **teacher disagreements**, not human errors.
Mask IoU against the corrected teacher is 0.812 outer/0.665 pupil on this
partition. The separate validation partition is weaker: accepted pupils
22/12. Do not conceal that regression behind aggregate mask agreement.

`report-final.json` also compares the actual live workers, with identical 84 native
validation exposures and outer+pupil requests, including the real SAM video
tracker (rather than the stateless export):

| Measurement | SAM live worker | Student live worker |
| --- | ---: | ---: |
| Median elapsed per exposure, excluding first use of each eye | 259.5 ms | 34 ms |
| p95 elapsed | 428 ms | 48 ms |
| Median encoding/preprocessing stage | 119 ms | 11 ms |
| Admitted outer observations / 84 | 44 | 38 |
| Admitted pupil observations / 84 | 25 | 13 |

There are 33 common admissions, 11 SAM-only and five student-only unverified
admissions. All returned source identities were verified. The approximately
7.6× median processing-time reduction is useful, but **pupil coverage is worse**
on this replay. This does not justify replacing the default SAM mode.

Both replay resource claims were soft-exclusive with `honored=true`, predicate
`gpu=0;gpu-memory=0;cpu=*;memory-bandwidth=*;block=*`, no reported conflicts:
reference token `l18d469d0695793c3-341263`, candidate token
`l18d46a3375e74e05-34207c`. Claim evidence is not proof of a perfectly idle host.
The reference ran first, then the candidate; this is one sequence of runs,
not a randomized hardware performance study. Model-file reduction does not
measure total CUDA context, allocator, or application memory.

The final focused suite passed 96 tests (four explicit corpus tests ignored), and
the two source-selection Python tests passed. The full viewer suite passed
1,039 tests with 39 failures and 29 ignored. All 39 failure names are already
present in `outputs/sign-interlock-removal.qRbBoq/tests-final.log`; missing
older RAW fixtures and pre-existing geometry/UI assertions remain unresolved.
The source-tree allowlist audit and a no-CUDA build check passed. No camera/firmware changes, human label
edits, or default SAM replacement were made.

Report matched teacher/student RAW admissions, losses and student-only
observations, pupil availability, ellipse disagreements, and latency. A
student-only RAW-admitted result is **unverified**, not automatically a rescued
true positive. Neither teacher overlap nor ellipse agreement substitutes for
canonical human-label localization error.

Use **scale-normalized frontal-equivalent iris disk area (SN-FEIDA)** as defined
in [the area/motion document](flat-tire-area-and-motion.md). The reporter only
normalizes using source-provided independent `pixels_per_10mm` hints, never the
student's fitted radius. These are rough MediaPipe-derived scale supports, not
calibrated anatomical measurements. The pilot has very sparse scale support;
its widely spaced exposures cannot establish motion stability or meridian-sign
accuracy. Full source-time motion sequences and additional human labels remain
necessary before any default/promotion decision.

Teacher-export timing queries six prompts and is not the ordinary two-product
live SAM workload. Cached-input student timing excludes RAW preprocessing.
The completion-paced replay uses the actual live workers, includes native
preprocessing and shared fitting, verifies returned source identity, and reports
model warmup separately. It does not measure camera/display delay or offered-load
frame dropping.

### Bounded pupil alternatives and 3D-path validation

On the same 730 pilot frames with the original `qualified.ot` weights, adding
the bounded pupil alternatives changed accepted pupil counts from 163 to 218
on train, 12 to 21 on validation, and 26 to 30 on the development test split.
Outer admissions were unchanged. No previously fitted pupil was lost. On the
same teacher/student-common pupil cases, mean center disagreement changed
5.836→5.815 px (train), 6.724→6.604 px (validation), and 9.083→9.090 px (test).
This is not anatomical ground truth, and the small test regression is retained.
The actual 84-frame live-worker pupil count improved from 13 to 21 (SAM: 25).
These diagnostic runs overlapped teacher export, so their times are not speed
benchmarks.

`replay` and `replay-paired` additionally export the source-native **measured**
retained points, disconnected conic segments, censored points and RAW-qualified
pupil. They never generate dense rim samples from a completed ellipse. These
records feed `buttercup_stereo_conic_eval` without a second image interpretation.
The paired replay on `outputs/dual-eye-joint.MiR1Iw/label-source-frames.jsonl`
verified 27 same-clock two-eye groups and 22 single-eye sources (76 exposures),
including exact returned timestamps, epochs, sequences and group sizes.

The canonical label inventory has 16 reviewed labels; 10 have exact native-RAW
matches in that 76-exposure neighborhood index. Labels were opened for scoring
after inference, never supplied to the network/solver. With `qualified.ot` and
the three pupil alternatives, mask-fit admission at those 10 labels was SAM
7/10 and student 9/10. On the seven commonly admitted visible-label frames,
equal-frame mean RMS localization error was SAM 5.009 px versus student 4.494 px.
One of the ten labeled RAW images was present in the pilot's pseudo-label
training set, and related recording sessions are not a population holdout.

Shared joint-3D fit admission was SAM 8/10 versus student 10/10, but on the eight
commonly admitted labeled frames localization was **worse**, 8.165→10.496 px.
The difficult ROI-move target at sequence 317 accounts for much of the error
(29.840→46.954 px); sequence 10235 also regressed (8.500→14.184 px). A solver
producing a result is not proof that it localized the eye correctly. These are
component replay results, not calibrated cursor or 3D-pose ground truth.

All 124 SAM and 120 student contributing-eye normal evaluations across the
joint and monocular solves were camera-facing; the minimum facing cosine was
0.498 and 0.531 respectively. These normals reuse the existing constrained 3D
solver, not a separately supervised neural pose head. The 76 exposures provide
no independent metric-scale support, so SN-FEIDA is **unavailable** here; do not
normalize by either model's fitted radius. Relevant artifacts are
`labels-*-live.jsonl`, `labels-student-v1.jsonl`, `labels-*-conics*.jsonl`, and
`labels-score-*.json` beneath the work directory. The scorer's historical `SAM`
key denotes the input mask reference even when its input is the student; the
filenames identify the actual backend.
