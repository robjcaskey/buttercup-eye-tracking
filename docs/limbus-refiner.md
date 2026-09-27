# Experimental optical limbus refinement

Training and reuse of model-derived targets are governed by the
[bootstrapability contract](../bootstrapability.md). The experiments below do
not yet constitute a current-checkout cold-bootstrap proof or a new-user
onboarding validation.

**Tweaked Contact Geometry** follows the original contact view: SAM3.1 ROI
`F 8/11`, Butter Obelisk/Student ROI `F 11/15`; linked SAM `F 7/8`, linked
Obelisk/Student `F 9/11`. Both use immutable inference-source RAW and dimensions,
including after a newer ROI crop moves or resizes. Other detectors omit it.

This is a separate trained model with a different purpose from the SAM mask
student: move already observed limbus samples a small distance along their
outward normal, while distinguishing the surface landmark from deeper visible
optical continuation. It does not replace SAM, recover a missing iris, or
promise a perfect fit. Selecting F changes **presentation only**, not tracking,
calibration, mouse, laser or eye-presence authority.

For a lean comparison of the actual toggle's effect, select **LIMBUS EDGES
BEFORE / AFTER** instead: cyan original edges, pink dashed accepted corrections,
using the worker's saved before/after on one source frame. It does not run a
preview refinement, draw a completed hidden ellipse, or include the contact
meridians. Rejected attempts remain the original edge. See the
[viewer controls](viewer-workspaces.md#limbus-edges-before--after).

## Shared geometry plumbing and experimental authority

`--limbus-refinement off|experimental` (environment
`BUTTERCUP_LIMBUS_REFINEMENT`) selects a startup-only global analysis stage,
separate from F. **Default is off**: this checkout's model still lacks the
required cold-bootstrap proof. The switch is experimental comparison plumbing,
not permission to bypass `bootstrapability.md`; live use of the unverified
checkpoint requires an explicit experimental exception. No verified/default
promotion or training was performed for this integration.

When explicitly enabled, the native SAM video and Obelisk workers refine once
per source before publishing their shared `OuterResult` and `ProposalMasks`.
The accepted ellipse and corrected retained arc points consequently feed
tracking, pupil projection, single-eye/stereo conic solving, signed contact
pose, gaze, calibration, focus-follows-eyes, mouse and laser consumers. There
is no separate cursor/laser adjustment and the model supplies no signed gaze.
The independently measured pupil is not moved or replaced by this model.

The CPU-only pass retains the existing flat-tire exclusions and sparse arc
indices. It requires baseline RAW admission, bounded source-local shifts,
new candidate RAW admission, and compatibility with any measured pupil.
Missing weights, unsupported corrections or failed gates preserve the exact
baseline and its support. These abstentions are not new refined observations.
The source timestamp, crop origin, prompt generation and epoch are unchanged.
Legacy three-adapter SAM consensus is not wired to this native-video stage.

The F renderer reuses the worker's immutable decision, including rejection,
rather than refining the effective ellipse a second time. It labels shared
versus preview-only geometry. The Model panel reports the selected source's
refinement status. Recording predictions and sequence replays include source-
identified decision/status, original/candidate ellipse and CPU time. With
authority off the renderer can still compute its original preview-only field.

Worker physical context is explicitly unavailable: current proposals do not
carry independently measured source-aligned scale/focus. Do not substitute
the fitted radius or a newer UI estimate. Recalibrate when evaluating a changed
geometry pipeline; an old screen mapping is not evidence of its accuracy.

Matched evaluation uses the ordinary source/session and contact replay, e.g.:

```sh
BUTTERCUP_LIMBUS_REFINEMENT=off data/target/live/buttercup-eye-viewer \
  --offline-sam-sequence-eval outputs/RUN/base.json CAPTURE subject-right 200 120 1 native student
BUTTERCUP_LIMBUS_REFINEMENT=experimental data/target/live/buttercup-eye-viewer \
  --offline-sam-sequence-eval outputs/RUN/candidate.json CAPTURE subject-right 200 120 1 native student
data/target/live/buttercup_report_limbus_refiner live outputs/RUN/report.json \
  outputs/RUN/base.json outputs/RUN/candidate.json
```

The Rust report rejects mismatched source/crop/clock/motion inputs and compares
fresh signed coverage, RAW admission, matched human localization when present,
independently normalized SN-FEIDA steps and ROI-reframe subsets. Gaze-vector
change is labeled **difference, not target accuracy**. Coordinate streams and
corpus reports remain runtime data; no checkpoint is copied into Git.

### Shared-path integration check, September 12, 2026

Reports and exact input/report hashes: `outputs/limbus-shared-eval.rSIlhH`.
No weights were trained, replaced or promoted. Compared the installed refiner
and RAW Obelisk `outputs/raw-student.DUYOAP/raw.ot` with refinement off/on on
120 consecutive exposures per eye (indices 200–319) from
`outputs/micro-clock-compare.G6mlnW/capture`, plus the 37-frame canonical
triplet subset described below. SAM native-video also ran on the same right-eye
120 exposures and the same 37 triplet frames. Baseline/candidate jobs ran in
parallel under shared CPU/GPU coordinator leases (not exclusive benchmarks).

| Matched subset | RAW admissions, off → on | Fresh signed contacts, off → on | Mean absolute SN-FEIDA log step, off → on |
| --- | --- | --- | --- |
| Obelisk, right 120 | 120 → 120 | 117 → 117 | 0.02307 → 0.02185; 82 supported pairs |
| Obelisk, left 120 | 108 → 108 | 104 → 104 | 0.02672 → 0.02493; 68 pairs |
| SAM, right 120 | 117 → 117 | 113 → 113 | 0.02441 → 0.02231; 81 pairs |
| Obelisk, 37 labeled-triplet sources | 37 → 37 | 4 → 4 | 0.03533 → 0.03305; 18 pairs |
| SAM, same 37 sources | 33 → 33 | **7 → 2** | 0.09600 → 0.09430; 14 pairs |

On 14 matched canonical labeled frames Obelisk's equal-frame mean visible-rim
RMS changed **7.628 → 7.343 px**: eight improved, five regressed, one unchanged.
The worst increase was +0.220 px at sequence 226. Sequence 10124 remained badly
localized at 17.85 px; sequence 10206 still had 17.14 px error. SAM's 11 matched
labeled admissions changed 8.259 → 8.130 px, with a worst increase of +0.529 px.
These labels overlap development material; this is not held-out accuracy or
an independent model selection result.

Right-eye reframe comparisons had ten independently scale-supported pairs:
Obelisk mean absolute area step 0.04060 → 0.03933; SAM 0.03401 → 0.03065.
No left-eye reframe pairs were available. Left-eye area p95 worsened slightly
despite the better mean (0.07906 → 0.08023). Scale is a source-timed RAW texture
similarity estimate, not calibrated millimeters; missing/unsupported scale is
not filled from the ellipse. Continuous clips have no human limbus labels.

The short SAM low-light sequence lost five signed outputs at sequences 10206,
10207, 10216, 10217 and 10218, including two unchanged-ellipse abstentions whose
preceding trajectory had changed. Thus modest localization improvement does
**not** establish better sign acquisition or justify default authority. The
five sources had no admitted pupil cue. At 10206, baseline motion residuals
were [30.240, 20.176] px, exceeding the 5.034 px separation threshold; refined
residuals were [26.860, 23.885] px, below that same threshold. This is reduced
sign coverage, not proof the original sign was correct: the baseline's labeled
rim error on that source was already 15.650 px (refined 16.178 px). Do not lower
the acquisition threshold or copy the baseline sign merely to restore coverage.
No target-based accuracy or sign-ground-truth claim was made for those short
clips. On continuous clips, median
baseline-to-refined gaze-vector changes were about 0.67–0.80 degrees; these are
differences, not accuracy improvements.

CPU refinement averaged 1.08–1.33 ms per attempted Obelisk source (p95
1.42–1.57 ms), excluding model loading, main segmentation and downstream gaze.
Shared-host contention means these are indicative costs, not latency guarantees.
One conflicting-pupil correction, unsupported local fields and shape-bound
failures retained their exact baselines. Source/epoch/prompt identities were
unchanged; excluded arcs were not reinstated.

Focused publication, fallback, UI, source-render and model-bound tests passed,
including the real RAW/model Obelisk render fixture and CPU report tests.
The broader `limbus` name-filter run passed 86 tests, skipped six and failed
four unrelated Driving/RAW tests because their legacy fixture files are absent.
It is not reported as a passing full viewer suite.

### Full recorded-target efficacy, September 12, 2026

**Obelisk improves on this recording; SAM has a serious persistent sign
regression. Neither outcome authorizes default promotion.** Full source-matched
off/on replays and reports are in `outputs/limbus-target-efficacy.Mf1eGC`, with
final reports `student-target-v2.json` and `sam-target-v2.json`. Each backend
processed all 1,094 subject-right exposures from the same 106-second, five-site
micro-motion/optical-clock recording above. Viewer binary SHA-256 was unchanged
before and after all four runs:
`e54dafb5c35c01fdf96fed82b655c21cee1470a24d7f3e7d1d9d52a98ce4ed31`.
Baseline/candidate pairs ran concurrently under shared coordinator leases;
SAM and Obelisk pairs ran sequentially. No live camera or training was used.

The new portable Rust `targets` report reads native OIM1 presentations, including
fractional pixel coordinates and explicit target removal. It retains the
recording's monitor translation/rotation/size, rejects changed monitor geometry
or spliced sessions, and verifies each source's sequence/timestamp/clock epoch.
It does not recalibrate either arm or feed targets into predictions.

```sh
data/target/live/buttercup_report_limbus_refiner targets outputs/RUN/targets.json \
  outputs/RUN/base.json outputs/RUN/candidate.json CAPTURE/metadata.oim1 \
  outputs/micro-clock-compare.G6mlnW/clock-summary.json \
  outputs/micro-clock-compare.G6mlnW/blink-review.json
```

Use the clock/blink inputs belonging to the evaluated capture, not these paths
for arbitrary new recordings. The reporter hashes all input files. The imported
empirical clock-offset band is 45.058 ms wide, assumes unit clock rate, and is
**not** a calibrated confidence interval or an independently measured display
scanout/exposure delay. The first verified in-record optical receipt is source
2136. The same pre-existing RAW visual blink review is used unchanged for both
arms, padded by 200 ms before/500 ms after; it is assistant visual review, not
canonical human blink labels. Large site relocations require one second of
acquisition across the entire clock band and 0–400 ms response-lag sweep. Tiny
microsteps remain eligible. Exclusion precedence yields 127 blink, 52 acquisition
and 91 pre-clock sources, leaving **824** eligible exposures.

These are conditional angular discrepancies between fresh signed gaze and
commanded targets under the captured fixed-reference-eye monitor model—not
independent anatomical gaze/visual-axis truth or proof of fixation. The 200 ms
reference lag is not a fitted physiological latency. Each off/on pair uses the
intersection of fresh signed sources so missing predictions cannot improve its
matched score. The two backend intersections differ; the table is **not** a
same-census SAM-versus-Obelisk ranking.

| Backend | Matched target sources | Mean error, off → on | Median, off → on | p95, off → on |
| --- | --- | --- | --- | --- |
| RAW Obelisk | 823 | 3.337° → 2.941° | 3.272° → 2.673° | 6.253° → 5.772° |
| SAM native video | 798 | 3.023° → 8.151° | 2.325° → 2.678° | 7.402° → 46.612° |

Own-admission eligible counts were Obelisk 823 → 823, SAM 801 → 799. Across the
entire recording, RAW admissions were unchanged (Obelisk 1,088; SAM 1,052), while
fresh signed contacts were 1,080 → 1,080 and 1,039 → 1,037 respectively. Obelisk
mean target error improved at four sites but regressed at site 3
(4.349° → 4.511°); site 1 p95 also worsened (2.853° → 3.080°). SAM's final site
mean jumped from 4.383° to 26.853°. Across 0–400 ms reference lags and both clock
band endpoints, aggregate mean errors varied by less than 0.002° in each arm;
clock offset within this tested range does not explain the regression. This
does not establish reliable relative tracking of the 1–5 px microsteps.

The first large SAM off/on directional divergence occurs at **2998**, following
the reviewed blink at 2995. At 2997 refinement changes the axis ratio from
0.97439 to 0.99832: the ellipse is nearly circular, so its major-axis direction
is poorly determined. From 2997 to 2998 the fitted major-axis change falls on
opposite sides of 90° (baseline 91.17°, refined 88.02°). The sign tracker's
nearest transported-direction assignment consequently takes opposite branches.
Both retain branch label 1 and sign epoch 2: this is an implicit physical-
identity reassignment, **not** an explicit sign vote/epoch change. At 2998 the
baseline gaze y is −0.347; refined y is +0.388. There are 114 fresh source pairs
with off/on angular disagreement above 18.19° (dot product below 0.95), continuing
through the final source 3113. Blink-window exclusion removes affected scores
near the blink but cannot undo poisoned temporal state afterward. This localizes
a follow-up sign-association issue; no sign-history policy was changed here.

SN-FEIDA still improved, illustrating why area alone is insufficient:

| Backend | Independently supported area pairs | Mean absolute log step, off → on | Reframe pairs; mean step, off → on |
| --- | --- | --- | --- |
| Obelisk | 912 | 0.024295 → 0.023322 | 127; 0.027524 → 0.026269 |
| SAM | 867 | 0.038814 → 0.034460 | 115; 0.048549 → 0.042947 |

SAM's reframe p95 worsened from 0.14693 to 0.15405. These area statistics use the
full source census, separate from the target exclusions, with independent RAW
texture scale and no dropout bridges. This full recording has no canonical
limbus labels; the labeled localization check remains the small development-
overlap subset above. The data are Rob-only, with no new-user, held-out or
stereo-target validation. Completion-paced replay also does not measure live
queue latency, realtime drop policy or a user's post-refinement recalibration.

The refiner applied on 1,069 Obelisk and 1,010 SAM exposures. Mean CPU pass cost
was 1.194 / 1.213 ms respectively (p95 1.553 / 1.417 ms), excluding loading and
the rest of the pipeline; these shared-host observations are not benchmarks.
The portable report's 15 tests pass, including native metadata truncation,
session/frame attestation, hidden targets, fractional coordinates, tilted
monitor geometry, integer clock precision and rejection of held/unsigned gaze.
A baseline-versus-itself check reproduces the earlier SAM report's 801-source
mean, median and p95 exactly. Per-site quantile conventions differ from the old
temporary analyzer (the new report uses midpoint median and nearest-rank p95).

## Representation and implementation

The **optical limbus normal-displacement field** is a sparse collection of
source-pixel offsets and visibility distributions. It is not a measured
anatomical 3D heightmap, corneal thickness, or a triangulated surface.

| Output role | Meaning |
| --- | --- |
| `rim` | Generic reviewed localization; legacy band midpoint is a lower-weight fallback |
| `band_inner`, `band_outer` | Iris-side and sclera-side edges of an uncertain transition band |
| `iris_onset` | Iris-side onset of a reviewed optical triplet |
| `surface_apex` | Explicitly labeled apparent surface landmark, preferred for refitting |
| `subsurface_limit` | Deeper visible continuation; diagnostic only, never a refit target |

These are outer-limbus optical roles, not the pupil aperture's inner boundary.
A band midpoint is never relabeled as an anatomical apex. Unreviewed, guessed,
assistant, and backup annotations do not supply visible training labels.
Explicit possibly-occluded landmarks supervise the not-visible class; unknown
coordinates and targets outside a patch do not become invented coordinates.

The native Rust model has 24,358 parameters: 276 inputs, 64 ReLU hidden units,
and six 17-bin outputs (16 normal positions plus not-visible). A 16×16 tensor
is sampled at two-native-pixel spacing, covering roughly 32×32 native RAW
pixels. Patches are oriented normal/tangent to the current ellipse. RAW10 is
decoded with the repository's decoder; a local 3×3 tent suppresses mosaic
phase before bilinear sampling. This is a scalar linear-RAW model, not a
claimed color demosaic or a display/lightbox screenshot model.

Twenty context channels accompany the image: measured brightness, contrast,
sharpness energy, saturated/dark fractions and radial contrast; apparent
ellipse size, axis ratio, pi-invariant angular position around the ellipse,
sampling pitch; optional independent scale, rough camera range and their
allowances; and optional focus position with missing-data indicators.

Tiny patches do **not** independently identify signed gaze or absolute camera
distance. Viewing angle is conditioned on the baseline ellipse's unsigned
axis-ratio proxy. Physical range uses external coarse scale and the same
uncalibrated 4000-pixel focal prior as the live joint solver, with a 25% focal
allowance added to the propagated scale span. This is a defeasible engineering
estimate, not calibrated distance truth. Relative motion scale is never
converted into millimeters or camera range. The corpus provides no aligned
VCM supervision, so motor-position channels are disabled; focus awareness here
means **measured local RAW sharpness**, not a learned focus-motor calibration.
The live scale context is the held coarse acquisition estimate on the EyeFrame,
not an independently measured range on each asynchronous SAM exposure.

Training runs on CUDA using Rust/LibTorch, with RAW gain/blur and normal/tangent
jitter augmentation, 50% optional-context dropout, AdamW, and seed 73017.
Inference is native Rust CPU code without a GPU round trip. Exported portable
logits were checked against Torch (maximum difference below 2.4e-6 in the
grouped models and 9.6e-7 in the installed model).

Module boundaries:

- `limbus_refiner.rs`: RAW patches, model, six-role field, bounded corrections.
- `limbus_refiner_train.rs` and `buttercup_limbus_refiner`: offline CUDA training,
  exact-source evaluation, weak-context preparation, and portable export.
- `limbus_refiner_view.rs`: presentation-only caching and contact rendering.
- Existing `conic_solver::robust_contour_fit`: common baseline/control/refined fit.

## Bounds and source alignment

At most 96 **observed, already de-flat-tired retained points** are queried per
new SAM source allocation. Missing or excluded arcs are not filled with sampled
points just because a predicted ellipse crosses them. Missing patch pixels
cause abstention, never border clamping.

The heuristic acceptance criteria are visible mass >=0.85, normal spread <=4
native pixels, contrast >=0.015, saturation <20%, consistent supported
onset/apex/subsurface ordering, and a requested displacement within ±4 native
pixels. At least ten local corrections are required for a refit. The final
ellipse must satisfy the shared fitter's validity bounds, center displacement
<=4 pixels, each axis change <=6%, and a 64-position whole-rim displacement
bound of 4 pixels. These are engineering supports, not calibrated probabilities.

Supported apex wins over generic rim. The subsurface limit never pulls the
fitted surface outward. On abstention, the exact source's original contact is
retained and explicitly labeled as original—not counted as a fresh refinement.

Each eye caches by immutable SAM proposal allocation. Background pixels,
contours, patch coordinates and clipping dimensions come from that proposal,
not a newer ROI crop. Contact meridians require a camera-facing, sign-resolved
surface on exactly the same source timestamp. Otherwise the view shows only
the rim and says `WAIT SOURCE SIGN`; no sign decision is manufactured. The
existing convex-contact renderer and checks are reused.

Gold shows the refined rim/apex, teal/orange the inner/outer band, blue the
iris-side onset and purple deeper optical continuation. Some landmarks will
be absent when their distributions are too uncertain. The small offsets may
be hard to see without comparing the preceding original-contact view.

## Corpus and results

Artifacts: `outputs/limbus-refiner.3Qc8Ru`. Weights are external runtime data at
`data/models/limbus_refiner_v1.json`, a link to `final-model.json` in that run.
Override with `BUTTERCUP_LIMBUS_REFINER_MODEL`; restart after changing a model
(the view intentionally loads once). No weights or captures belong in Git.

There are **16 reviewed native labeled images in four conservative source-time
groups** (8/4/3/1 images), not sixteen independent sessions. They contain 205
generic rim, 87 inner-band, 87 outer-band, 42 onset, 38 apex and 36 subsurface
visible observations, plus explicit possibly-occluded evidence. No independent
physical scale is matched to these sixteen labels. Their source timestamps
exist, but original cross-session clock identity is not independently attested;
nearby timestamps were conservatively grouped rather than random-splitting
patches. Human geometry is never an inference feature or candidate selector.

Another **78 scale-bearing native RAW sources from ten clock lineages** provide
weak SAM-only generic-rim supervision at 3% label weight. They are from the
existing student teacher's training partition and exclude exact human RAW
identities and every human timestamp neighborhood within 300 seconds. They
provide no invented apex, subsurface depth or signed-gaze labels. Physical
context is supported only approximately, over this narrow corpus range.

Four-fold development evaluation holds out an entire source group, uses a
different whole group for checkpoint selection, and trains on the other two
plus the nonoverlapping weak sources. The selected epochs were 1/1/1/5 from
60-epoch trials. The final interactive model uses all reviewed development
data and one epoch (the median validation choice), 55,957 augmented patches.
Its training fit is **not** a held-out score. These comparisons informed
development; they are not a sealed population generalization test.

Fresh SAM baseline queries were made without labels. Four images had no ellipse;
two further diagnostic ellipses failed RAW admission. On the ten RAW-admitted
held-out images, the model accepted refinement on six and abstained on four:

| Matched six-image metric | Baseline | Unmodified refit control | Learned refinement |
| --- | ---: | ---: | ---: |
| Equal-frame mean generic-rim RMS error, native px | 4.5568 | 4.5568 | 4.3109 |

Five improve and one regresses: night sequence 521 worsens by **0.131 px**.
Keeping the untouched baseline on the four abstentions yields a mean error
change of -0.148 px over all ten, but those four are not fresh model outputs.
The wide, badly placed seq317 diagnostic ellipse remains wrong; seq10124 has
no baseline ellipse; seq10163 remains a large localization failure; seq224's
suggested change fails the global shape bound. This cannot rescue a bad SAM
localization or prove pupil/gaze accuracy.

### SN-FEIDA and adjacent motion

Use **scale-normalized frontal-equivalent iris disk area (SN-FEIDA)** exactly
as defined in [the area/motion document](flat-tire-area-and-motion.md):
`pi * major_radius^2 / independent_scale^2`. It is neither pupil aperture area
nor curved tissue surface area. No candidate radius supplies its own scale.

The scale-bearing weak training frames were too sparse for adjacent <=500ms
comparisons. A separate **37-frame canonical RAW before/target/after subset**
was therefore imported from the existing archived SAM contour experiment.
The importer rechecked exact native bytes against the independent RAW-motion
streams, plus timestamp, sequence, sensor origin and dimensions. Twenty-four
frames have bounded relative scale; each chain retains its own reference.
Thirty baseline frames fit, 28 refine, and **ten matched adjacent pairs** have
both fresh candidates and the same independent scale reference:

| Mean absolute natural-log SN-FEIDA step, same ten pairs | Value |
| --- | ---: |
| Archived baseline | 0.021391 |
| Unmodified refit control | 0.021375 |
| Learned refit | 0.016427 |

On the six comparable labeled refined frames in this archived-contour subset,
equal-frame RMS changes from 4.2560 to 4.0497 px; seq247 worsens by 0.163 px.
These sources overlap development training and use archived, not newly queried,
SAM contours. This is a conditional implementation diagnostic, not an
independent temporal-model test. There are **zero supported ROI-reframe pairs**
in those ten; reframe invariance is unit/render-tested but not established by
this area result. Missing candidates, gaps >500ms and scale-reference changes
are never bridged. Scale-only area bounds omit fitted-radius/optical uncertainty.
Stable but wrong or frozen ellipses do not validate the model.

Removing physical context on the 78 weak-source training examples changes
accepted refinements from 76 to 73; on 73 common candidates, mean absolute
major-radius difference is 0.645 px (max 1.313 px). This proves context is used,
**not that its distance inference is accurate**. These are SAM-only labels.

The sparse CPU refine pass (including patch extraction and both control/refined
conic fits, excluding model loading, RAW IO, SAM, and drawing) measured mean
**0.829 ms**, median 0.845 ms, p95 1.268 ms over 30 source frames. The shared-host
soft-exclusive `cpu=*;memory-bandwidth=*;block=*` request was honored, token
`l18d470994edbc510-351de3`; see `eval-sequence-control.log`. This is a small local
run, not an end-to-end latency or sustained-load guarantee.

## Reproduction and next evidence

With the same LibTorch environment used by `scripts/run-viewer.sh`:

```sh
cargo build --profile live --features sam31 --bin buttercup_limbus_refiner
cargo build --profile live --no-default-features \
  --bin buttercup_prepare_limbus_refiner --bin buttercup_report_limbus_refiner
data/target/live/buttercup_prepare_limbus_refiner INVENTORY.json outputs/NEW_RUN/human
# Generate a label-blind SAM reference with the existing student replay tool:
data/target/live/buttercup_eye_student replay outputs/NEW_RUN/human/sam-inputs.jsonl sam outputs/NEW_RUN/sam.jsonl
data/target/live/buttercup_limbus_refiner prepare-aux TEACHER_DIR outputs/NEW_RUN/human/dataset.json outputs/NEW_RUN/aux.json
BUTTERCUP_LIMBUS_AUX=outputs/NEW_RUN/aux.json data/target/live/buttercup_limbus_refiner train outputs/NEW_RUN/human/dataset.json outputs/NEW_RUN/sam.jsonl outputs/NEW_RUN/fold.json 60 TEST_GROUP VALIDATION_GROUP
data/target/live/buttercup_limbus_refiner evaluate outputs/NEW_RUN/human/dataset.json outputs/NEW_RUN/sam.jsonl outputs/NEW_RUN/fold.json outputs/NEW_RUN/eval.json
```

Use `train-all ... EPOCHS` only after choosing a schedule from grouped validation.
`BUTTERCUP_LIMBUS_ABLATE_CONTEXT=1` is an explicitly reported evaluation ablation.
The preparation tool's `--sequence-area-report` imports existing matched
motion evidence into an **evaluation-only** dataset; the trainer refuses it.
All generated files use new runtime destinations. `buttercup_report_limbus_refiner`
has `cv` and `sequence` modes, preserving identity and missingness.

```sh
data/target/live/buttercup_report_limbus_refiner cv outputs/NEW_RUN/cv-report.json \
  outputs/NEW_RUN/fold0-eval.json outputs/NEW_RUN/fold1-eval.json
data/target/live/buttercup_report_limbus_refiner sequence outputs/NEW_RUN/sequence-report.json \
  outputs/NEW_RUN/sequence-eval.json
```

Preparation/reporting now share `src/training_refiner_data.rs`, with native
Rust landmark/clock/area tests replacing the Python equivalents. These tools
build with `--no-default-features` and do not require Python, LibTorch or CUDA.
The existing Rust/CUDA patch trainer is shared, offline foundation work, not a
personal adaptation trainer. Optional user/scenario refinement training and
inference must be CPU-only. Follow the
[bootstrapability contract](../bootstrapability.md), including the declared
DAG preflight, before training or reusing derived material; successful
component commands alone do not prove cold bootstrapability.

Validation includes native landmark/clock/area tests, Rust geometry and
subsurface-exclusion tests, SAM/Obelisk UI-cycle tests, portable no-CUDA checks,
and an opt-in actual-model RAW render test. Set
`BUTTERCUP_LIMBUS_UI_CORPUS_DIR=outputs/limbus-refiner.3Qc8Ru` for the latter;
the displayed sign in that offline renderer fixture is synthetic, **not gaze
truth**. Production sources still require a real same-exposure resolved sign.
The full unrelated viewer suite was not claimed passing.

This version is stateless across exposures except for presentation caching.
It does not yet learn parallax, refraction/translucency, anatomical height, or
temporal de-flat-tire exclusion. Promotion needs more independently labeled,
source-registered motion/reframe sequences, focus/illumination coverage and
bounded optical/pose supervision. Always compare matched baseline/candidate
corpus results alongside tests, inspect regressions, and keep SN-FEIDA paired
with localization, dropouts and independent scale—not an area-stability-only
training objective.
