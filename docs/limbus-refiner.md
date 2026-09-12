# Experimental optical limbus refinement

SAM3.1 now has **Tweaked Contact Geometry** at ROI `F 6/8`, immediately
after the original contact view. Linked SAM ROIs also offer it at `F 4/4`.
It is not in the Eye Student or other detectors' F cycles.

This is a separate trained model with a different purpose from the SAM mask
student: move already observed limbus samples a small distance along their
outward normal, while distinguishing the surface landmark from deeper visible
optical continuation. It does not replace SAM, recover a missing iris, or
promise a perfect fit. The preview changes **no tracking, calibration, mouse,
laser, eye-presence, or de-flat-tire authority**.

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
python3 scripts/prepare-limbus-refiner.py INVENTORY.json outputs/NEW_RUN/human
# Generate a label-blind SAM reference with the existing student replay tool:
data/target/live/buttercup_eye_student replay outputs/NEW_RUN/human/sam-inputs.jsonl sam outputs/NEW_RUN/sam.jsonl
data/target/live/buttercup_limbus_refiner prepare-aux TEACHER_DIR outputs/NEW_RUN/human/dataset.json outputs/NEW_RUN/aux.json
BUTTERCUP_LIMBUS_AUX=outputs/NEW_RUN/aux.json data/target/live/buttercup_limbus_refiner train outputs/NEW_RUN/human/dataset.json outputs/NEW_RUN/sam.jsonl outputs/NEW_RUN/fold.json 60 TEST_GROUP VALIDATION_GROUP
data/target/live/buttercup_limbus_refiner evaluate outputs/NEW_RUN/human/dataset.json outputs/NEW_RUN/sam.jsonl outputs/NEW_RUN/fold.json outputs/NEW_RUN/eval.json
```

Use `train-all ... EPOCHS` only after choosing a schedule from grouped validation.
`BUTTERCUP_LIMBUS_ABLATE_CONTEXT=1` is an explicitly reported evaluation ablation.
The preparation script's `--sequence-area-report` imports existing matched
motion evidence into an **evaluation-only** dataset; the trainer refuses it.
All generated files use new runtime destinations. `report-limbus-refiner.py`
has `cv` and `sequence` modes, preserving identity and missingness.

Validation includes Python landmark/clock/area tests, Rust geometry and
subsurface-exclusion tests, SAM-only UI-cycle tests, portable no-CUDA checks,
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
