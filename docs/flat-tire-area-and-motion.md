# Recent occlusion memory, motion, and iris disk area

The September 5, 2026 experiment does **not** justify enabling temporal
re-exclusion in the live fitter. The bounded API and offline evaluator exist,
and 29 matched alternatives were evaluated on 107 unique native RAW frames.
Soft penalties preserve coverage but trade small area changes for localization
regressions. Several hard hysteresis settings lose eight additional frames.
The live stateless defaults remain unchanged.

## Canonical diagnostic: SN-FEIDA

Use **scale-normalized frontal-equivalent iris disk area (SN-FEIDA)**, variable
`sn_feida`, for the inferred unobstructed **outer limbus disk** after frontal
rectification and independent linear image-scale normalization. Keep the full
name on first use; “iris area” and “post-affine area” are too ambiguous.

Assume a circular planar limbus under weak perspective, with image semi-axes
`a >= b > 0`, major-axis rotation `R`, and independently estimated linear image
scale `s > 0`. Under this model, `cos(theta) = b/a`. The rectification

```text
W = R diag(1, a/b) R^T
projected_disk_area = pi a b
frontal_equivalent_disk_area = pi a b / (b/a) = pi a^2
sn_feida = pi a^2 / s^2
```

maps the inferred outer ellipse to a circle of radius `a`, before dividing
linear coordinates by `s`. Tilt direction/sign is unnecessary for this scalar.
In-plane translation, rotation, and the sign ambiguity of an ellipse axis do
not change its value. The formula is an engineering model, not evidence that
real imaging obeys exact weak perspective or that limbus size never changes.

The reference for `s` must be independent of the tested candidate radius. A
relative RAW-motion chain can set its first exposure's linear scale to one and
report area in that reference image's squared-pixel units. A calibrated
pixels-per-millimeter scale would instead give square millimeters. Reference
changes must carry a new identity; do not compare the absolute values across
unrelated references.

Do not set `s = a`, fit a separate scale to make each candidate match a desired
area, or use an arbitrary warp that maps every candidate to the same disk. An
area-preserving ellipse-to-circle warp gives radius `sqrt(a*b)` and preserves
`pi*a*b`; it does not compute frontal-equivalent area. The viewer's spherical
contact-cap approximation using the mean projected radius is also not this
measurement. This experiment changes no contact rendering.

These quantities remain distinct:

| Quantity | Meaning |
| --- | --- |
| Outer limbus disk | Inferred entire outer disk, including occluded portions and the pupil aperture |
| Visible semantic mask | Only pixels selected by the current mask; can lose area through occlusion |
| Pupil aperture | Inner opening, with its own boundary and dynamics |
| Iris annulus | Region between outer and inner boundaries under an explicitly specified geometry |
| Curved iris tissue surface | Requires a surface model/reconstruction; not determined by `sn_feida` |

Pupil or iris-annulus contraction does not by itself prove shrinkage of the
outer limbus disk. Conversely, this work establishes no empirical law forbidding
outer-limbus change. Depth/image scale uncertainty, projection error, optical
distortion/refraction, boundary misidentification, and anatomical variation are
different hypotheses to test. No anatomical population range or physiological
constant is asserted here.

SN-FEIDA is a north-star **diagnostic**, not an objective that can overrule
evidence. Hard geometric/anatomical validity remains a prerequisite, including
camera-facing convex contact surfaces and applicable projected-limbus limits.
Estimated size and motion bounds remain defeasible; the effective rotation
pivot is movable and must not be made a fixed anatomical truth by an area prior. A constant eyelid
ellipse, frozen prediction, or persistently wrong limbus cannot pass simply
because its area is stable.

### Small changes and defensible uncertainty bounds

“Same scale” means compatible scale support, not identical depth, optics or
anatomy. In logarithmic coordinates the diagnostic obeys
`delta log(sn_feida) = 2 delta log(a) - 2 delta log(s)`. A 1% residual error in
linear scale therefore produces about a 2% area error. With the illustrative
pinhole relationship `s = f/Z`, a small forward/back displacement gives
`delta log(s) ~= -delta Z/Z` when focal length is held fixed. Apparent contraction
inside a coarse scale bucket can consequently be unresolved depth/scale change;
it is not automatically evidence of tissue contraction or a bad fit.

For positive bounded supports `a in [a_min,a_max]` and `s in [s_min,s_max]`, a
conservative area support is

```text
sn_feida_min = pi (a_min / s_max)^2
sn_feida_max = pi (a_max / s_min)^2
```

These endpoints are support bounds, not a claimed 95% confidence interval.
When radius and scale are correlated, propagate their joint feasible samples
or covariance rather than multiplying independent confidence claims. Partial
arcs, blur, uncertain boundary identity, camera distortion and approximation
error must widen support or cause abstention. A narrow fitted residual alone
does not establish a narrow hidden-rim or frontal-area interval.

Freeze candidate-independent scale/size support before evaluating a frame.
Transport it only through source-aligned, bounded, independently supported
motion; retain native sensor origins so crop movement is not mistaken for
head translation. Keep translation, in-plane rotation, disk tilt and image
scale separate, using the actual affine determinant for area transport.
Bound accumulated drift and reset incompatible source/reference chains. Let
repeated new observations revise soft scale and movable-pivot estimates, but
do not let rejected candidates widen their own hard bounds or held overlays
refresh evidence. The scene's existing admission/recovery protections are
described in [the geometry architecture](geometry-architecture.md#scale-admission-and-recovery).

Compare small area changes with this uncertainty and with label localization,
coverage and source-timed motion before attributing them to anatomical change.
Pupil-aperture/annulus dynamics and inferred outer-disk change remain separate
measurements; this benchmark does not identify a physiological cause.

## Bounded recent-exclusion API

`src/outline_conic_segments/recent_exclusion.rs` is declared by the outline
module but has no call from the live stateless fitter. One memory instance
belongs to one physical ROI/source-clock lineage. The protocol is:

1. `begin_frame` admits one new source exposure and resets incompatible state.
2. All current alternatives query the same previous-exposure snapshot.
3. `observe_current` commits once, using independently computed current-frame
   occluding chords and fresh good arcs from one selected evidence set.

There are 48 optional angular regions and at most 128 bad plus 128 good sample
visits per commit. No images, unbounded candidate history, or trajectories are
stored in this memory. Each region holds the most recent independent bad
source timestamp, normalized radial position, and hysteresis state. Correlated
alternatives do not cast separate votes. A memory-weighted rejection is never
fed back as a fresh bad observation.

Coordinates use symmetric image-axis whitening
`R diag(1/a, 1/b) R^T (point-center)`. This is only a region-correspondence
coordinate system, **not** the independent scale `s`. It avoids the pi-axis
sign ambiguity and arbitrary axis direction of a circle. Queries consider the
same angular bin and its two neighbors, within normalized radial distance
0.20. A fresh supported curved arc gets weight one immediately and clears
overlapping previous exclusions after the frame commits. Good current
evidence wins a conflict with a current bad observation.

The reset/abstention rules cover a changed ROI/clock/epoch/geometry lineage,
stale or repeated source time/sequence, invalid geometry, a horizon-length gap,
incompatible scale provenance or missing/known scale transition, more than
25% adjacent scale/radius/aspect change, more than 0.35 prior major radii of
sensor-space center shift, and excessive change of the pi-invariant
anisotropy/orientation descriptor. Ordinary crop translation is handled through
the sensor origin. All numeric compatibility bounds are engineering choices;
they are not calibrated motion or anatomy probabilities.

Cached/reordered results cannot refresh or clear evidence or advance time using
host completion time. Fresh later exposures may confirm a still-present chord,
but no observation persists beyond the finite horizon without new evidence.
Out-of-order calls abstain without rewinding the remembered exposure.

For age `t`, decay parameter `tau`, horizon `H`, and maximum penalty `p`:

| Function | Penalty before `H` | Expiry |
| --- | --- | --- |
| Exponential | `p exp(-t/tau)` | Zero at `H` |
| Linear | `p max(0, 1-t/tau)` | Zero at `min(tau,H)` |
| Finite horizon | `p` while `t < tau` | Zero at `min(tau,H)` |

Soft weight is `1 - penalty`. Config validation bounds `p <= 0.95` and
`0 < tau <= H <= 2 seconds`. The tested settings use `tau` of 100, 250, and
500 ms, `H = 3*tau`, and `p` of 0.50 or 0.75. Each function/time also has a
`p=0.75` hard hysteresis arm: enter at 0.60 and leave below 0.20. A fresh good
arc still overrides hard exclusion. Hard exclusions can reduce usable support
below the fit threshold, producing an explicit dropout.

## Matched offline experiment and independent scale

`src/bin/buttercup_flat_tire_eval.rs` evaluates exported native contour evidence.
The final evidence is
`outputs/flat-tire-area-20260905/matched-comparison-v3.json`.
Earlier `sequence-comparison`, `v1`, and `v2` reports are development artifacts,
not the final census or determinant-consistent scale comparison.

| Source subset | Unique RAW frames | Mask candidates | Baseline fitted candidates | Selected fitted frames |
| --- | ---: | ---: | ---: | ---: |
| Existing right-eye sequence, 918–952 | 35 | 420 | 134 | 35 |
| Existing left-eye sequence, 918–952 | 35 | 420 | 21 | 14 |
| Canonical annotation triplets | 37 | 444 | 71 | 30 |
| Total | 107 | 1,284 | 226 | 79 |

The 14 canonical target labels are sequences 222, 223, 224, 226, 247, 195,
10091, 10124, 10163, 10181, 10193, 10206, 10217, and 10235. Their original
before/target/after RAW bytes remain under their captures' `annotator/archive`
directories; only runtime symlinks and a prediction-free index were constructed.
Sequence 225 appeared identically in two archives. Exact metadata and RAW-byte
equality deduplicated that exposure and joined the overlapping source lineage.
The generic-stroke backup and nested assistant annotation were excluded.
This is a specified subset, not a full-corpus benchmark.

Every arm fixes the query to the first baseline-RAW-eligible ellipse in the
export's semantic order. The exported baseline remains untouched. A separate
`matched_polish_no_memory` control and all 27 decay arms use the identical
eight-iteration robust weighted polish on the same baseline-retained samples.
This separates the effect of polish from the effect of temporal weights.
No area target or label-derived selection enters fitting or memory updates.
All label files are opened for scoring only after every candidate is fixed.
The candidate audit includes all 226 initially fitted alternatives; these are
not 226 independent eyes or exposures.

The polish checks current support/spread and inherited geometric limits;
it cannot recover a mask that had no baseline ellipse. It is an offline
conditional refit, not a replay of the live candidate-arbitration or SAM video
memory pipeline. The RAW-support statistic is median lateral outer-minus-inner
contrast from 3x3 native RAW means, normalized by 1023. It is an additional
current-image diagnostic, not a reimplementation of the live RAW admission
gate. There is no frozen-output fallback.

All source exposures have native timestamps and ROI geometry. They do not
provide calibrated cross-session clock alignment or row-exposure timing. The
70-frame capture's recorded motion telemetry has zero matched features and
zero supported scale layers, so it supplies no usable independent scale.
The older night-sequence report has a different source and was not substituted.

Existing `manual35-baseline.json` and `low0` through `low7-baseline.json` under
`outputs/benchmarks/log-luma-ab-20260901` do provide native global-similarity
evidence for some exact labeled-subset frames. The estimator consumes broad
RAW patches before iris segmentation and takes no candidate radius. Imported
records require exact timestamp, sequence, ROI dimensions/origin, and bytewise
RAW identity. Only reliable, supported, bounded residual/motion links enter a
scale chain. This is independence from the candidate fit, not statistical
independence from every iris pixel or a calibrated depth measurement.

For the native matrix `A = [[1+d,-r],[r,1+d]]`, the linear scale multiplier is
`sqrt(det A) = hypot(1+d,r)`. Using just `1+d` would mistake part of rotation
for an area-scale change. Chains use a fixed reference exposure, reset across
unreliable/missing links, and re-anchor after one second or a summed heuristic
fractional allowance above 0.25. The allowance is summed conservatively across
correlated steps, not presented as calibrated confidence. Its maximum on the
matched frames is about 0.202. Comparisons never cross reference identities.

There are **24/107 frames with independent relative-scale support**, **20 with
both that scale and a selected ellipse**, and **11 adjacent normalized-area
comparisons**. There are 58 adjacent unnormalized-area comparisons. Neither
metric bridges a dropout, source/geometry change, or gap longer than 500 ms.
Missing scale remains unavailable. Relative SN-FEIDA values across distinct
references must not be pooled as absolute anatomical areas.

## Results and interpretation

The label metric is each fitted frame's mean Euclidean distance from visible
canonical `iris_edge` points to the inferred ellipse, then an equal-frame mean.
Eleven of 14 target frames fit in these displayed arms. No visible-point error
is imputed for the three missed low-light targets. These sparse visible labels
do not certify the inferred hidden rim.

All area-step entries below are dimensionless absolute natural-log changes,
not percentages. “Pixel area” is unnormalized frontal-equivalent area.

| Arm | Fits / 107 | Label mean px | Mean SN-FEIDA step, 11 pairs | Pixel-area mean step | Pixel-area p95 step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exported baseline | 79 | 5.71808 | 0.035378 | 0.143882 | 0.468428 |
| Matched polish, no memory | 79 | 5.64472 | 0.036631 | 0.145766 | 0.458963 |
| Linear 100 ms, p=0.50, soft | 79 | 5.64513 | 0.036630 | 0.145771 | 0.458963 |
| Exponential 250 ms, p=0.75, soft | 79 | 5.66755 | 0.036265 | 0.146520 | 0.457898 |
| Finite horizon 250 ms, p=0.75, soft | 79 | 5.67581 | 0.035992 | 0.148254 | 0.456997 |
| Exponential 250 ms, p=0.75, hard hysteresis | 71 | 5.68522 | 0.035675 | 0.140220 | 0.588546 |

The finite-horizon `p=0.75` family has the lowest mean normalized-area step
among tested **soft** decays: 0.035992 versus 0.036631 for the matched control,
about a 1.7% reduction in that diagnostic. Its 100/250/500 ms versions tie on
these 11 normalized transitions, so the corpus does not identify an optimal
duration. The 250 ms arm worsens label mean by 0.03109 px, increases mean
unnormalized area step, and slightly reduces mean lateral RAW contrast
(0.014165 versus 0.014359). The exported baseline has a still lower mean
normalized-area step of 0.035378. This is not a justified production win.

Linear 100 ms with `p=0.50` is the least disruptive tested soft memory in
localization: only +0.00041 px over the matched control. Its effect on normalized
area is negligible. Native cadence is approximately 96–100 ms, so a 100 ms
linear penalty is almost gone by the next exposure. Its good localization
ranking is not evidence of effective long-lived occlusion recovery.

Hard hysteresis makes the dropout trap visible. The displayed hard arm's
unnormalized mean falls from 0.145766 to 0.140220, but only 45 common pairs
survive. On **those same 45 pairs**, the control mean is 0.139513, so the hard
arm actually worsens the matched mean while losing eight frames. Its p95 also
worsens. Dropping troublesome frames is not stabilization.

Individual label regressions also matter. The no-memory polish improves
sequence 10193 from 7.62714 to 6.37910 px, while worsening sequence 10206 from
14.28073 to 14.61512 px. The finite-horizon 250 ms arm worsens sequence 222
from the matched control's 2.72329 to 2.95965 px, but improves sequence 10235
from 7.59928 to 7.53048 px. Sequences 10091, 10124, and 10163 have no selected
ellipse in any arm. Stable area cannot conceal those localization errors or
missing coverage.

The report additionally records `log(A_candidate/A_control)` and its **signed
change**. A shared unknown `s^2` cancels exactly from those ratios. This measures
implementation effect, not physical-area consistency. Subtracting two
`abs(log area step)` values does **not** cancel unknown scale; that subtraction
is only an unnormalized matched implementation comparison unless independent
scale is actually available.

## Reproduce and continue

Build and test the standalone evaluator without GPU/model dependencies:

```sh
cargo test --no-default-features --bin buttercup_flat_tire_eval
cargo build --profile live --no-default-features --bin buttercup_flat_tire_eval
```

The first command now includes 33 tests: ten bounded-memory contracts, three
offline numerical checks, and the shared geometry/timing/RAW tests included by
the standalone harness. The full viewer's known failures are a separate test
suite; this document does not claim they have been repaired.

The final comparison command was:

```sh
data/target/live/buttercup_flat_tire_eval \
  outputs/flat-tire-area-20260905/matched-comparison-v3.json \
  outputs/module-extraction-20260905/right-outlines.json \
  outputs/module-extraction-20260905/left-outlines.json \
  outputs/flat-tire-area-20260905/canonical-outlines-v2.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/manual35-baseline.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/low0-baseline.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/low1-baseline.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/low2-baseline.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/low3-baseline.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/low4-baseline.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/low5-baseline.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/low6-baseline.json \
  --scale-report outputs/benchmarks/log-luma-ab-20260901/low7-baseline.json
```

Outputs use `create_new`; select a new output filename for a rerun. The
`--prepare-label-capture OUTPUT_DIR CANONICAL.labels.json ...` mode creates the
deduplicated runtime capture view. The exact 14 canonical label paths and
archived RAW paths are retained in
`outputs/flat-tire-area-20260905/canonical-triplets-v2/frames.jsonl`.
That view was exported with the existing viewer's offline-only command:

```sh
# With the native LibTorch/NVIDIA library paths configured, as in run-viewer.sh:
BUTTERCUP_SAM31_PREPROCESS=mild-blur \
BUTTERCUP_SAM31_PROMPT_BUNDLE=data/models/sam31_semantic_prompts_cuda_bf16.pt \
data/target/live/buttercup-eye-viewer --offline-sam-outline-export \
  outputs/flat-tire-area-20260905/canonical-outlines-v2.json \
  outputs/flat-tire-area-20260905/canonical-triplets-v2 subject-right
```

No viewer restart, recording, annotation UI launch, live memory enablement,
model change, or label write was performed. Runtime artifacts stay under the
checked `outputs` link. New source paths are the recent-exclusion child module,
the standalone evaluator, and this document; the source allowlist tracks them.

The standing resolve is to run matched baseline/candidate corpus evaluations
alongside geometry theory and synthetic tests, in parallel where practical,
and then actually analyze localization, coverage, current support, source-time
alignment, independent scale, and regressions. Unit tests and launching a
replay are necessary tools, not substitutes for that analysis. Future live
integration needs a reproducible benefit on those joint criteria, longer
sequences with independent scale, explicit handling of candidate association
and uncertainty, and current-frame RAW admission after the new fit. These
results provide no reason to bypass that evidence requirement.
