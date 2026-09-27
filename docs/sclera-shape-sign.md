# Sclera shape as independent iris-sign evidence

Estimating the globe's **position and surface normals** can resolve an iris
tilt ambiguity. Estimating only its radius of curvature cannot: both iris
interpretations can belong to globes of the same radius at different centers.
This is an offline investigation, not an enabled or validated live sign solver.

## What would be sufficient

Write a unit-radius circular iris interpretation as center `C` and outward
normal `n`. Under the explicitly assumed spherical globe model,

```text
d = sqrt(R² - 1)
G = C - d n
```

For an allowed globe-radius interval, each interpretation supplies a line
segment of possible 3D globe centers. Its perspective projection is also a
line segment when the segment remains in front of the camera. Let `delta` be
the minimum distance between the two projected center segments. An independent
center measurement with bounded error `epsilon < delta/2` cannot be compatible
with both families. It must still be compatible with at least one. If departure
from the model can displace each family by at most `eta` pixels, the sufficient
condition becomes `epsilon + eta < delta/2`.

This is a conditional geometric separation result, not an estimator accuracy
claim. It depends on the conic, intrinsics, valid circle reconstruction and
surface-family bounds. A geometric globe center is not an effective rotation
pivot. As the two interpretations converge, this measurement ceases to separate
them; their distinct physical meaning also diminishes.

On the area-admitted ambiguous frames in `roi-area-first-focus-v1`, using the
same nominal intrinsics and globe radii 1.8–2.4 times iris radius:

| Provider | Ambiguous frames | Smallest sufficient error radius | Median |
| --- | ---: | ---: | ---: |
| SAM3.1 | 202 | 45.13 native pixels | 58.56 pixels |
| Butter Obelisk | 193 | 46.96 native pixels | 62.27 pixels |

Thus an independently justified error below 45 pixels would separate all of
these particular candidate families. We have **not** measured that accuracy.
An aspherical or decentered globe, conic localization error and camera error
require additional allowances. These engineering radius intervals are not
certified anatomical bounds for Rob.

Broadening the globe-radius family to 1.5–2.8 iris radii reduces the smallest
sufficient center-error allowance to 33.80 pixels for SAM and 35.17 for Obelisk.
These bounds still exclude conic/camera error and departures from the sphere
model; those must be subtracted from the allowance, not silently ignored.

## Why lighting can supply the missing measurement

For a diffuse, unshadowed surface sample observed under known lighting
directions and strengths, after subtracting ambient/background illumination,

```text
I = rho L N
rank(L) = 3  =>  g = L⁺ I = rho N  =>  N = g / ||g||
```

The unknown positive albedo `rho` may differ between surface samples. It cancels
when normalizing each recovered vector. The exposures must observe the same
surface pose, with linear radiometry and stable albedo. Three independent
lighting equations are needed here; three copies of the same light direction
are insufficient. An unchanged ambient term can instead be removed using a
reference exposure, provided motion and exposure changes are handled.

Let `q` be the unit camera ray through a measured point, and `P = t q` its
unknown 3D position. For the visible surface of a sphere,

```text
N = (P - G) / R
(I - q qᵀ) (G/R) = -(I - q qᵀ) N
```

Stacking these linear equations gives the center in globe-radius units without
knowing metric eye size. The projected center is scale invariant. Two distinct,
nonparallel rays suffice for algebraic rank; many spatially separated samples
are needed for useful conditioning under noise. The native synthetic assay
uses actual sclera sample layouts, both hypothetical centers, varying albedo,
known lights and explicit RAW-code noise. It is an oracle measurement study,
not evidence that these lighting measurements exist in the recording.

Unknown natural lighting is a harder inverse problem. Shape-from-shading can
recover useful constrained local shape distributions, but the information
depends on lighting, shape assumptions and noise. It should retain alternatives
when the observed patch is uninformative. [Xiong et al., From Shading to Local
Shape](https://arxiv.org/abs/1310.2916)

The general unknown-shape/unknown-light problem is not uniquely determined by
shading. The generalized bas-relief ambiguity gives different Lambertian
surfaces, albedos and lights with identical images under its orthographic
assumptions. This is a counterexample to unconditional recovery, not a proof
that these particular constrained sphere candidates must always be
indistinguishable. [Belhumeur, Kriegman and Yuille, The Bas-Relief
Ambiguity](https://vision.ucsd.edu/publications/1999/bas-relief-ambiguity)

Reflections are a second route. Corneal reflections can encode illumination
under an assumed corneal geometry; estimating a different light from each iris
hypothesis does not provide an independent shape measurement. [Nishino and
Nayar, Eyes for Relighting](https://www.cs.columbia.edu/cg/pdfs/19-Nishino_TOG04.pdf)
Dense display-pattern reflections have also been used to recover corneal and
scleral normals and estimate an optical axis. That experimental setup uses
calibrated display/camera geometry and two cameras observing the same eye to
resolve depth/normal ambiguity. Our camera's two eye ROIs do not supply that
stereo observation. [Wang et al., Single-Shot
Deflectometry](https://arxiv.org/html/2308.07298v3)

## What upper and lower lids add

The two 3D iris interpretations project to the same ellipse `E`. Given the same
observed lid opening `L`, the visible iris `E ∩ L` is identical. Knowing only
those two image curves cannot universally distinguish the interpretations.
A constant-brightness sclera patch contained inside both globe silhouettes is
a counterexample even with perfect lid masks. Useful extra information is a
relationship between the lids and globe position, a true exposed sclera point
outside one entire allowed globe family, or independently measured normals.
Such anatomical/lighting assumptions must be stated and checked.

### Planar-lid oracle and the visibility failure

`surface-lid-oracle` tests an additional **assumption**, not an established
anatomical fact: each lid margin follows a plane/sphere intersection. It lifts
the same image points onto each candidate globe, fits an orthogonal least-squares
plane to each lid, and measures the projected circle's first-order pixel
residual. A common radius is searched for both lids; their planes may differ.
The declared finite radius grid is not a continuous planarity exclusion proof.

The experiment uses the native iris interpretations of all 202 SAM and 193
Obelisk ambiguous area-admitted rows. Each interpretation becomes a synthetic
generating truth in turn, at radius 2.15 iris radii, with three lid configurations.
Separate scenarios perturb pixel positions by up to 1, 2 or 4 pixels per axis,
or bend the curves on the sphere by 0.01, 0.05 or 0.1 iris radii. These are
synthetic cases, not independent observations or physical sign labels.

The first pictures exposed a serious limitation: many generated openings would
hide iris contour points reported visible in the source segmentation. The
subsequent audit requires at least 70% of those points to be covered by both
synthetic curves' horizontal extent, and at least 90% of compared points to be
inside the opening, allowing 3 pixels at its margins. These source contour
points are themselves unreviewed predictions; passing is only a necessary
compatibility check, not proof of correct eyelid position.

In [`roi-planar-lid-oracle-v4`](../outputs/roi-planar-lid-oracle-v4/summary.json),
only **7 SAM and 1 Obelisk noise-free configurations** pass that visibility
check. Before the check, containment appeared to separate 675/717 SAM and
577/610 Obelisk configurations with sufficient generated curve support and a
10-pixel total error allowance. Those attractive fractions are **not** evidence
for nearly-all real-frame disambiguation. Most underlying synthetic openings
conflict with the actual visible iris. Planarity residuals alone select neither
branch in the noise-free study under the declared 2-versus-5-pixel thresholds.

The exact generating sphere's noise-free circle residual is at most
`9.62e-12` pixels. No continuous containment bound excludes its known synthetic
generator beyond the declared pixel perturbation. These are implementation
controls, not anatomical validation. Updated pictures show both a
[visibility conflict](../outputs/roi-planar-lid-oracle-v4/oracle-90874-sam-1-0.png)
and a [compatible synthetic opening](../outputs/roi-planar-lid-oracle-v4/oracle-91401-sam-0-2.png).
Even the latter is not a detected lid boundary.

### Continuous single-point exclusion bound

The oracle also checks a simpler sufficient condition that does not require
planar lids. A reliable exposed-globe point must intersect its globe. For unit
camera ray `q`, sphere center `G(d) = C - d n`, and `R(d) = sqrt(1+d²)`, define

```text
h(d) = angle(q, G(d)) - asin(R(d) / |G(d)|)
```

Positive `h` places the ray outside the sphere silhouette. For the full radius
interval 1.5–2.8, let `M = |C| - d_max |n| > 2.8`. A derivative bound is

```text
L = |n|/M + (1/M + 2.8 |n|/M²) / sqrt(1 - (2.8/M)²)
min_d h(d) >= min_grid h(d) - L * grid_spacing/2
```

The implementation uses 33 equally spaced `d` samples and a small numerical
allowance. With the nominal 4000-pixel focal length, a pixel displacement of
`epsilon` changes the ray direction by at most `epsilon/4000`. Therefore a
positive lower bound greater than `epsilon/4000` excludes **every** radius in
that candidate family, conditional on `epsilon` bounding the combined point,
camera and model errors. An observed point outside the entire family suffices;
different pixels cannot be allowed to choose incompatible radii.

The actual corpus has no measured combined error bound or reviewed lid/contact
labels. A negative bound means this witness is inconclusive; it does not prove
that the candidate is correct. No real sign choice is admitted by this oracle.

```sh
data/target/live/buttercup_roi_focus surface-lid-oracle \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 NEW_OUT
```

## Continuous containment experiment

`surface-sign` searches one shared globe radius continuously for each pose.
For a pixel ray, the sphere-intersection discriminant is a quadratic in
`d = sqrt(R²-1)`. Interval endpoints partition the radius range. Sweeping these
intervals maximizes containment without giving different pixels different
radii. A direct intersection control checked 101 radii per synthetic pixel with
zero membership disagreements.

Only area-admitted inputs from sources with remaining ambiguity enter this
experiment: 239 SAM and 230 Obelisk provider/exposure rows, all source 239,
epoch 172. The ambiguous subsets are the 202 and 193 above. The same existing
Obelisk sclera and lid masks supply observed pixels for both signs and both
providers; they remain unverified predictions, not anatomy labels.

At nominal radius range 1.8–2.4, 98% versus 80% containment thresholds select
2/202 SAM and 5/193 Obelisk ambiguous rows. **None** survives all eight
mask/erosion/radius settings, including a broader 1.5–2.8 radius range. Several
source-matched images visibly admit both globes. This is evidence against this
simple containment procedure being sufficient, not against all sclera cues.

## Independent shading profile

`surface-lighting` uses upstream area-admitted inputs exclusively. It fits an
independent sphere's image center and size together with four illumination
coefficients, `I = a + b·N`, using exact perspective ray/sphere intersections.
The candidate iris signs do not set the free sphere center or its light. The
fitted ellipse supplies the broad search window and image-size unit only.

Measurements are individual RAW10 green photosites, separately at the two
absolute sensor CFA phases, selected with two sclera-mask thresholds and a lid
veto. There is no demosaicing, spatial averaging, fitted white balance, sign
label or training. Shape trials must contain the same complete sample set.
The mask is still an unverified model-derived diagnostic input.

Compare constant intensity, an image-plane gradient, independent sphere shading
and both iris-attached globe families. Shape/radius and light are chosen on
training blocks; predictive errors use the other 24×24 sensor-coordinate
blocks. Swap the split and combine the held-out errors. Full-data profile
support at explicit excess-error thresholds is a sensitivity plot, **not a
confidence interval** or continuous exclusion proof. Report negative ambient
or shadowed directional fits rather than interpreting them as physical light.

Unknown lid shadows, skin contamination, pigmentation, tear-film reflections,
glasses and complex nearby lighting can all invalidate the approximation.
Passing an error threshold is not sign accuracy. The cohort lacks reviewed
physical sign labels, human limbus localization labels and independent scale.
Neither fit changes the input ellipse or its frontal-equivalent area. The
upstream area gate uses unnormalized `pi*a²`, not full SN-FEIDA.

## Reproduction

### Independent outer-motion prerequisite

`surface-center-motion` checks whether independently measured image motion can
support a temporal globe-center fit on this same area-admitted cohort. It is a
**motion diagnostic only**, not an implemented or validated temporal sign
estimator. No excluded or neighbor-insufficient exposures are inserted to fill
gaps, and a rejected transform is never treated as identity motion.

The native shared matcher has an opt-in support mode restricted to the upper
and lower 12.5% of **both** ROIs, with a 12-pixel patch/CFA safety margin. This
avoids the middle 75%, including the iris and most corneal reflections. It
compares the original radius-4 patch/10-column feature layout with a radius-8
patch/20-column layout. The larger footprint gathers more spatial evidence;
neither variant relaxes the established match, spatial-coverage or residual
gates. Sensor coordinates account for ROI reframing. Source gaps above 300 ms
break history. Whole-ROI motion is retained as a matched comparison, never as
a fallback for the independent outer-motion estimate.

Runs `roi-surface-center-motion-v5` and `roi-surface-center-motion-v6` contain
239 SAM and 230 Obelisk provider/exposure rows, representing 243 unique RAWs in
source 239, epoch 172. Each includes hashed RAW identities, eligibility,
source time, correspondence coordinates, rejected diagnostic fits and review
overlays. The inputs and whole-ROI comparison are identical between variants.

| Provider | First observations | Gaps over 300 ms | Outer-match rejection | Accepted outer motion |
| --- | ---: | ---: | ---: | ---: |
| SAM | 2 | 23 | 214 | 0 |
| Obelisk | 2 | 26 | 202 | 0 |

Both footprints have the same outcome counts. Among the ambiguous rows, SAM
has 2 first observations, 19 gaps and 181 matching failures; Obelisk has 2,
22 and 169. Whole-ROI matching accepts 66 rows per provider, but those matches
may include moving iris/lid/reflection texture. They do not establish the
independent head transport needed here. No temporal center fit was attempted
after this prerequisite failed; there is no new sign-choice or accuracy result.

The source-matched overlays show inconsistent outer correspondences. Increasing
patch support did not fix them. This is a limitation of these two matchers on
these sampled intervals, not evidence that head motion is absent or impossible
to measure. The area gate also does not establish that a fitted ellipse is
correctly localized. A/B are local candidate labels and can exchange order
between exposures; their index is not a temporal association.

Declared-motion controls recover both zero motion and a `(4,-4)` pixel
translation despite a `(12,4)` sensor-crop reframe, for both patch footprints.
All four are accepted within predeclared 0.15-pixel translation and 0.002
similarity-coefficient tolerances. These synthetic controls validate the
coordinate/support implementation; they do not validate real sign accuracy.
Eight existing native-global motion regressions and the existing exact patch
arithmetic parity check pass. Live matching uses its original footprint and
support policy; no camera or live viewer is restarted by this assay.

```sh
cargo build --profile live --no-default-features --bin buttercup_roi_focus
data/target/live/buttercup_roi_focus surface-center-motion \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 NEW_OUT 4
data/target/live/buttercup_roi_focus surface-center-motion \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 OTHER_NEW_OUT 8
```

As with the shading experiments, the full goal remains unmet: this work has
not disambiguated nearly all 202/193 cases or measured independent sign accuracy.
An independent globe-center measurement, sufficiently constrained illumination,
or a validated anatomical relationship remains necessary to make that claim.

### Lid-mask audit and fresh SAM comparison

The subsequent complete lid audit is
[`roi-surface-lids-audit-v1`](../outputs/roi-surface-lids-audit-v1/summary.json).
It evaluates 239 SAM and 230 Obelisk **iris-fit rows**, representing 243 unique
area-admitted RAW exposures. Both use the same historical Obelisk anatomy heads.
The lower-lid output largely duplicates the iris: its median binary overlap
with the iris head is about 0.763 IoU. Every exposure fails the newly explicit
0.5-IoU semantic-collision gate. This gate rejects an unusable aperture; passing
it would not establish anatomical accuracy. The displayed aperture endpoints
in the original pilot are therefore **not valid canthus measurements**. No lid
center prior from those masks is admitted as sign evidence.

This is a concrete limitation of those predictions, not a demonstration that
true eyelid geometry or shading lacks useful information. The initial shading
experiment also used these lid heads as a veto, so its mask-dependent results
must retain that qualification. It did not use the inferred canthus positions.

Fresh anatomy inference now has a reproducible CPU recipe using the pinned SAM
checkpoint. A simple device conversion of the BF16 CUDA graph failed its
predeclared parity check and was rejected. The accepted FP32 CPU adapter
replaces the upstream forced-BF16 fused linear/activation helper with FP32
linear and the corresponding activation, and constructs positional caches on
CPU. It preserves checkpoint weights and has its own recipe hash. Eager versus
serialized CPU output has zero maximum difference on both seeded export
checks. This is arithmetic validation, not segmentation or sign accuracy, and
it does not establish equivalence to the original BF16 CUDA graph.

Two six-prompt sets were run on records **90874, 91540, 91921 and 92377** (both
eyes, four different times in source 239). They are selected only from the
already area-admitted inputs. [Literal anatomical prompts](../outputs/roi-anatomy-sam-review-v1/anatomy-90874.png)
often select skin or iris for the requested structure. [Explicit descriptive
prompts](../outputs/roi-anatomy-sam-detail-review-v1/anatomy-91921.png) improve the
visible white region in the inspected examples, but neither set gives reliable
upper and lower lid boundaries. Up to three proposals per prompt are retained;
their RAW, mask and RAW-to-SAM adapter hashes pass native review replay. The
second and third proposals also contain semantic mistakes. These observations
are visual sanity checks, not reviewed human labels.

The matched shading follow-up is
[`roi-surface-sam-lighting-pilot-v2`](../outputs/roi-surface-sam-lighting-pilot-v2/summary.json).
It evaluates all eight admitted provider/RAW rows for those four exposures,
including three SAM-ambiguous and four Obelisk-ambiguous rows. The other 461
admitted provider rows lack these fresh anatomy outputs and are explicitly
outside this pilot. It uses the highest-scoring nonempty `white of the eye`
proposal, chosen before inspecting either sign, with no unreliable lid veto.
Mask sampling is converted to the existing 384×256 mask grid; measured
brightness still comes from individual native green photosites. The input
ellipses, area eligibility and RAW hashes are identical to the matched baseline.

SAM mask levels 128 and 153 are a zero-logit mask and a stricter sensitivity
check. They are not calibrated confidence or equivalent to Obelisk's historical
179/230 levels. Applying 179/230 directly to these SAM masks produced missing
support; that failed comparison remains in `roi-surface-sam-lighting-pilot-v1`.
The revised levels supply usable observations in all eight rows, but **zero
stable sign choices** and zero cases where free-center shading beats the planar
gradient in every mask/CFA variant. The historical masks likewise give zero
stable choices on this same subset. [An early fit](../outputs/roi-surface-sam-lighting-pilot-v2/photometry-90874-sam.png)
has broad center support; [a later fit](../outputs/roi-surface-sam-lighting-pilot-v2/photometry-92377-obelisk.png)
also prefers lighting outside the nonnegative unshadowed approximation.

After sharing the mask-threshold settings across individual and paired shading,
all 469 historical containment rows and four evenly sampled historical shading
rows reproduce the previous results exactly. The new mask experiment does not
change the area gate or demonstrate SN-FEIDA/localization improvement. There
are still no independent scale measurements or reviewed physical sign labels
for this subset. No custom model was trained or promoted.

The goal of disambiguating almost all 202/193 retained cases remains unmet.
Useful next evidence must independently constrain illumination, globe position
or surface normals; a good-looking mask or an assumed lid-center offset alone
does not establish the needed accuracy. The conditional known-light sufficiency
result above remains valid under its declared model assumptions.

Build `buttercup_calibration_sign` with `--profile live --no-default-features
--features sam31` and the pinned LibTorch setup. `sam-export-anatomy-cpu` accepts
an optional JSON array of six literal prompts and records them in the export.
Use the same resource-coordination wrapper as other corpus work.

```sh
data/target/live/buttercup_calibration_sign sam-export-anatomy-cpu \
  data/models/sam31_multiplex.pt outputs/NEW_CPU_EXPORT \
  outputs/roi-anatomy-detail-prompts-v1.json
data/target/live/buttercup_calibration_sign roi-anatomy \
  outputs/roi-area-first-focus-v1 outputs/NEW_CPU_EXPORT outputs/NEW_MASK_RUN 4
data/target/live/buttercup_calibration_sign roi-anatomy-review \
  outputs/NEW_MASK_RUN outputs/NEW_REVIEW
data/target/live/buttercup_roi_focus surface-sam-anatomy \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_MASK_RUN outputs/NEW_MATCHED_LIGHTING
data/target/live/buttercup_roi_focus surface-lids \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_LID_AUDIT
```

### Eye-opening prompts and matched image-input experiment

The additional pinned CPU FP32 export
[`roi-anatomy-sam-opening-export-v1`](../outputs/roi-anatomy-sam-opening-export-v1/export.json)
uses six literal prompts: `eye opening`, `open eye`, `exposed eyeball`,
`iris and sclera`, `upper eyelid margin`, and `lower eyelid margin`. Its two
seeded eager-versus-serialized checks have zero maximum score and mask-logit
differences. That establishes export arithmetic parity, not semantic accuracy.

Three matched runs use 12 uniformly sampled RAWs from the 243 unique
area-admitted source-239 exposures (stride 21), with identical model weights,
prompts, RAW hashes, crop metadata and provider eligibility:

- [`roi-anatomy-sam-opening-pilot-v1`](../outputs/roi-anatomy-sam-opening-pilot-v1/summary.json):
  the established `quad_rgb` input, 124.09 seconds.
- [`roi-anatomy-sam-opening-gray-v1`](../outputs/roi-anatomy-sam-opening-gray-v1/summary.json):
  integer RGB luminance repeated into three channels.
- [`roi-anatomy-sam-opening-bilateral-v1`](../outputs/roi-anatomy-sam-opening-bilateral-v1/summary.json):
  the same luminance with a fixed 11×11 bilateral kernel, spatial sigma 3 native
  pixels and intensity sigma 20 quantized input codes. No temporal input,
  candidate geometry or learned denoiser enters this transform.

Every inference is CPU-only with eight threads. The grayscale experiments ran
concurrently with shared resource claims; their wall times are not an isolated
performance comparison. Native RAW photometry and the live/training adapters
are unchanged. These transforms are optional diagnostic inputs only.

Visual inspection of the actual hashed model input confirms strong color noise.
Removing it sometimes improves the `open eye` proposal (record 91543), but
fails on other frames: it still selects iris alone (90874), surrounding skin
(91402), sclera alone (92063), or iris plus upper skin (92613). Upper/lower
margin prompts also sometimes return sclera instead of lids. Higher SAM scores
are not evidence of correct anatomical semantics. No masks from this experiment
are promoted to training labels or reliable globe-center measurements.

The native `roi-anatomy-compare` command verifies source, model, input-transform
and mask hashes before rendering a fixed prompt and rank across all three runs.
The [matched comparison](../outputs/roi-anatomy-opening-input-comparison-v1/comparison.json)
retains every selected frame, including regressions. The separate
[rank review](../outputs/roi-anatomy-sam-opening-bilateral-review-v1/review.json)
also verifies and renders up to three proposals per prompt. This remains a
12-RAW single-user diagnostic, with no human lid localization error or physical
sign accuracy available.

```sh
data/target/live/buttercup_calibration_sign roi-anatomy \
  outputs/roi-area-first-focus-v1 outputs/roi-anatomy-sam-opening-export-v1 \
  NEW_OUT 12 gray-bilateral-v1
data/target/live/buttercup_calibration_sign roi-anatomy-compare \
  outputs/roi-anatomy-sam-opening-pilot-v1 \
  outputs/roi-anatomy-sam-opening-gray-v1 \
  outputs/roi-anatomy-sam-opening-bilateral-v1 NEW_COMPARISON 1
```

These commands require the existing CPU LibTorch runtime described above.
The iris-area gate still precedes all anatomy processing. It uses unnormalized
frontal-equivalent disk area because independent scale support is missing;
it is not full SN-FEIDA. This follow-up does not establish nearly-all sign
disambiguation on the 202/193 ambiguous subsets.

### Applying the same geometry to the actual mask proposals

`surface-lid-proposals` applies the oracle's plane fitting and continuous
silhouette bound to actual fixed-rank `open eye` masks. It verifies area
admission, provider eligibility, native fits, source timestamps, RAW and mask
hashes. It selects a component touching the fitted iris independently of either
sign, censors crop edges, and keeps the longest contiguous horizontal interval
with two boundaries. These boundaries remain SAM proposals, not measured
anatomical truth. The iris interpretations and their areas are unchanged.

The 12-RAW pilot contains **18 still-ambiguous provider/exposure rows** on 11
unique RAWs: 10 SAM and 8 Obelisk. The other 377 ambiguous rows are outside this
pilot, not successful recoveries. At mask threshold 128 and a conditional
10-pixel total error allowance:

| Model input | Both retained | Both excluded | One retained conditionally | Missing boundary support |
| --- | ---: | ---: | ---: | ---: |
| Established RGB | 8 | 10 | 0 | 0 |
| Grayscale | 15 | 3 | 0 | 0 |
| Grayscale + bilateral | 13 | 1 | 1 | 3 |

The apparent single choice is **not a valid recovery**: record 91402's mask
follows surrounding skin, covers only 57% of the SAM iris contour horizontally,
and loses sufficient boundary support at threshold 153. Its actual
[RAW overlay](../outputs/roi-lid-proposals-bilateral-v1/proposal-91402-sam.png)
shows the error. At threshold 153 every supported case retains both candidate
families; the three input variants have 14, 8 and 5 missing-boundary rows.

The planar-circle criterion makes zero conditional selections at either mask
threshold. On the more plausible opening of record 92362, the original-input
SAM curves fit the two candidates with 3.08 versus 2.91 pixel RMS residuals,
insufficient separation. [The matched geometry picture](../outputs/roi-lid-proposals-pilot-v1/proposal-92362-sam.png)
shows both centers with the same boundary evidence. Record 91543's denoised
opening also remains ambiguous (5.16 versus 4.14 pixels).

Reports are retained for the
[original input](../outputs/roi-lid-proposals-pilot-v1/summary.json),
[grayscale](../outputs/roi-lid-proposals-gray-v1/summary.json), and
[bilateral input](../outputs/roi-lid-proposals-bilateral-v1/summary.json).
No actual sign decision is admitted. Both-excluded cases diagnose an
incompatibility among mask, conic, camera and sphere assumptions; they do not
identify which assumption failed. The crop/visibility checks do not certify
mask semantics. No custom model was trained on these unreviewed boundaries.

```sh
data/target/live/buttercup_roi_focus surface-lid-proposals \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/roi-anatomy-sam-opening-bilateral-v1 NEW_OUT
```

Both native diagnostic binaries build successfully. The final shared-admission
and geometry extraction was replayed as `roi-planar-lid-oracle-v5`: its entire
16,590-row oracle stream has the same SHA-256 as v4,
`1583d8d6b046ccec63fdd93798bb54d1a8a19f41187d8466287bec47cf6a08d0`.
Source-tree audit, scoped Rust formatting and whitespace checks pass. The
live camera, viewer settings and model defaults were not changed.

### Original full lighting run

The full matched run is
[`outputs/roi-surface-lighting-final-v1`](../outputs/roi-surface-lighting-final-v1/README.md).
All 202 SAM and 193 Obelisk ambiguous rows have usable RAW photometry; none
produces a stable admitted sign choice. Independent free-center shading beats
the image-plane gradient by at least 5% in all four mask/CFA variants on 59 SAM
and 56 Obelisk rows. There is therefore some shape-related predictive information,
but it has not established the center accuracy required for sign selection.
At the 5%-of-constant-error profile tolerance, the median maximum coordinate span
is about 258–259 pixels, much larger than the sufficient error allowance.

A second native experiment couples illumination across exactly matching eye
exposures: 89 SAM pairs and 87 Obelisk pairs. It fits a shared directional vector
with independent ambient terms, and checks equal versus 0.75–1.25 relative
albedo. The same sample set is used for all four branch combinations, and
radius/light/gain selection remains inside each training fold. Sufficient
statistics from each eye/sphere are reused across combinations instead of
repeatedly processing pixels. None of the pairs yields a stable admitted choice.
117 admitted rows lack an admitted exact-source partner and remain unpaired.

The expanded known-light synthetic assay uses every retained row, both candidate
globes and six separate perturbation scenarios. Among the ambiguous subsets,
all 404 SAM candidate cases and 386 Obelisk cases remain below their broad-family
separation allowance in every tested scenario. Worst center errors are 1.41/1.20
pixels for bounded ±10-code variation, 25.05/25.45 pixels for the specified 5%
light-strength errors, and 29.48/29.36 pixels for the specified 10-degree light
direction offset. These are separate tests, not combined worst-case bounds.
They use a declared ideal shape, varying diffuse albedo, ambient subtraction and
known light correspondences. They do not measure real-world sign accuracy or
establish that the archived scene has these measurements.

Noise-free normal-to-center reconstruction and the opposing constant-illumination
ambiguity control both pass. Direct containment results exactly match the prior
`roi-surface-sign-v1` run after extracting the shared mask sampler. Native build,
formatting, tree audit and whitespace checks pass. Visually inspected RAWs include
records 90874–90877 and later records 91097, 91544 and 92059: some estimated
spheres track lid/sclera shading while the allowed center region remains broad.
No model was trained or promoted and no live solver setting was changed.

Build the existing `buttercup_roi_focus` binary with `--profile live
--no-default-features`. Claim shared host resources before material work.

```sh
data/target/live/buttercup_roi_focus surface-sign \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_SURFACE_RUN
data/target/live/buttercup_roi_focus surface-lighting \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_LIGHTING_RUN
data/target/live/buttercup_roi_focus surface-report outputs/NEW_LIGHTING_RUN
```

An optional final positive integer on `surface-lighting` selects an evenly
spaced pilot from the admitted rows; omitted or zero processes them all.
Fresh source, mask and RAW hashes are checked. Outputs retain per-frame scores,
abstentions, synthetic measurements, native RAW overlays and input hashes.

## Shared unknown lighting over neighboring source exposures

The native `surface-temporal-lighting` follow-up asks whether a common light
direction over one or two seconds helps even without measuring the lights.
It uses the same 469 area-admitted source-239/epoch-172 provider rows, including
the 202 SAM and 193 Obelisk still-ambiguous rows. These are overlapping conic
providers on 243 unique RAW exposures, not independent recordings. Both use
the same historical Obelisk sclera proposals for sampling; this is not a new
SAM-versus-Obelisk sclera-segmentation benchmark. Those unverified custom masks
remain diagnostic previews and are not training roots or reviewed labels.

For every retained native green sample the model is

```text
I_i(p) = ambient_i + gain_i (L dot N_i(p))
ambient_i >= 0, gain_i >= 0, L dot N_i(p) >= 0 when gain_i > 0
```

The unknown unit light direction is shared. Each exposure has its own ambient,
gain, radius and unrestricted branch choice. Shape/radiometry are fitted on
one set of absolute-sensor 24-pixel checkerboard blocks and evaluated on the
other. Two native green photosite phases, two mask thresholds, and two source
windows produce eight settings. Windows reset at gaps over 300 ms and require
five distinct usable source times; they never borrow area-rejected frames.
Other retained focus classes can provide photometric context, but their
heuristic classes never become sign truth. Photometry uses only present/past
source times; the upstream area gate itself is retrospective.

The branch-sequence optimization separates:
`min_L sum_i min_(branch,radius,ambient,gain) E_i`. The search uses 1,024 fixed
light directions and 27 radii from 1.5 through 2.8 iris radii. It is a finite
search with heuristic margins, not a continuous exclusion or calibrated
probability. A matched independent-light baseline uses exactly the same
grids, samples, nonnegative fit and held-out blocks.

The initial run selected the best unconstrained directional fit and then
rejected it if any sample was shadowed. This can miss a worse-fitting valid
alternative. The corrected search enforces unshadowed feasibility during
optimization; ambient-only remains feasible everywhere. Conservative normal
bounding boxes avoid most per-sample dot products, with exact checks when
the bound straddles zero. This changes the optimizer, not the physical model
or its selection thresholds. It does not model cast shadows, multiple lights,
spatially varying albedo, refraction, or specular reflections.

The frozen [matched comparison](../outputs/roi-surface-temporal-comparison-v1/README.md)
verifies identical source times, RAW hashes, area admissions and search
settings before comparing baseline `roi-surface-temporal-lighting-v2` with
corrected `roi-surface-temporal-lighting-v4`:

| Iris-conic provider | Still ambiguous | Usable reference setting | Insufficient fresh temporal context | No common sphere/sample support | Stable sign selections |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM3.1 | 202 | 108 | 55 | 39 | 0 |
| Butter Obelisk | 193 | 110 | 47 | 36 | 0 |

The reference setting is the two-second window, mask threshold 179, native
green phase zero. Neither baseline nor candidate makes a selection there.
The corrected shared-light fit has about **8.3% higher median held-out MSE**
than its matched independent-light fit on each provider's usable subset;
the shared-light assumption did not improve predictive fit. Relative to the
old shared-light search, the median best-candidate MSE ratio is 1.0, with
individual ratios spanning 0.960–1.120 for SAM and 0.877–1.139 for Obelisk.
These are photometric errors, not gaze errors.

Only Obelisk record 91535 makes a provisional choice in any single setting:
threshold 230/green phase one, at both window lengths. Its companion green
phase does not pass the unchanged separation and gradient-control margins.
Both prefer B by raw error; this is loss of sufficient margin, **not an
observed opposite-sign flip**. The native
[phase-one overlay](../outputs/roi-surface-temporal-lighting-v4/temporal-preference-91535-obelisk-v3.png)
and [phase-zero overlay](../outputs/roi-surface-temporal-lighting-v4/temporal-preference-91535-obelisk-v2.png)
use the correct separate photosites and fits from the same RAW. This does not
count as a stable recovery.

Synthetic controls use the first forty usable SAM sampling layouts, known
branch truth, radius 2.15, an off-grid light and bounded ±3-code noise. Only
17 targets per scenario have the required temporal context. Among all forty:

| Synthetic scenario | Correct selections | Wrong selections | Abstained |
| --- | ---: | ---: | ---: |
| Upward-facing candidate family | 3 | 0 | 37 |
| Downward-facing candidate family | 16 | 0 | 24 |
| Uniform ambient illumination | 0 | 0 | 40 |

The asymmetric coverage is a remaining limitation even under ideal model
assumptions. Exact sufficient-statistics loss agrees with direct pixel loss
within 8.87e-11 MSE, and 122,880 normal-bound decisions agree with direct
feasibility checks. These arithmetic controls do not certify real masks,
lighting or shape. Adding the provisional-choice visual review leaves the
entire per-frame result stream identical between corrected runs v3 and v4
(SHA-256 `8672ac1f76c5877ddf7c7917df73bb5c073b3650a917a8be7e538e8d8ffea75e`).

```sh
data/target/live/buttercup_roi_focus surface-temporal-lighting \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_TEMPORAL_RUN
data/target/live/buttercup_roi_focus surface-temporal-report \
  outputs/roi-surface-temporal-lighting-v2 outputs/NEW_TEMPORAL_RUN \
  outputs/NEW_TEMPORAL_REPORT
```

An optional final SAM anatomy run can supply the fixed-rank `white of the eye`
proposal instead; that is a separate mask experiment, not this result. The
completed full-corpus runs took roughly ten seconds each under shared CPU
claims; this is not an isolated performance benchmark.

This follow-up leaves the central gap unresolved: no independent globe-center
error bound, reviewed lid/sclera localization or real physical-sign truth is
available. No model is trained or promoted. Conics remain unchanged, and the
area gate still lacks independent scale support, so no SN-FEIDA improvement
is claimed. A measured normal field under sufficiently independent known
lighting would supply the missing center measurement under the stated sphere
assumptions; estimating a curvature radius alone would not.

## Following scleral surface texture across exposures

A second way to constrain shape is to track actual material points. Let `N`
be the normal at a fixed point on a rigid spherical globe and `n` its iris
normal. Under the same proper rotation `Q`,

```text
(Q N) dot (Q n) = N dot n
theta = acos(N dot n) remains constant
```

Translation cancels when computing normals about each candidate globe center.
This yields a necessary constraint for each of the four source/target branch
combinations. It does not by itself prove rigid correspondence: relative
azimuths must also agree with one rotation, anatomical attachment is uncertain,
and conic/localization errors can dominate small motions. Tracking scleral
vessel texture for rotational measurement has precedent, but that does not
establish this dataset's feature identity or our sign accuracy.
[High-accuracy measurement of rotational eye movement by tracking of blood vessel images](https://pubmed.ncbi.nlm.nih.gov/25571446/)

`surface-sclera-motion` reuses the native RAW patch matcher, exact sensor origins,
forward/back checks and source ordering. Its optional diagnostic support API
selects features inside an allowed region before spending the cell budget;
it applies support at source, target, subpixel refinement and reverse search.
It returns correspondences only and cannot prime the live whole-ROI stability
counter. Ordinary live feature budgets and acceptance arithmetic are unchanged.

Both source and target patches must lie inside the same historical sclera
proposal, outside the fitted iris, with eight native pixels of support margin.
An integral mask makes this whole-patch check constant-time. These are proposed
sclera regions, not reviewed anatomical labels. The area gate still runs first.
Only source 239/epoch 172's 469 admitted provider rows, on 243 unique RAWs,
participate; gaps over 300 ms reset matching. The wider source context is
admitted even when its heuristic focus class is not ambiguous.

Matched feature-selection experiments preserve the patch cost and thresholds:

| Selection | Feature grid | SAM pairs with at least one strict match | Obelisk pairs with at least one strict match |
| --- | --- | ---: | ---: |
| Whole ROI, then sclera filtering | 10 by 8 | 7 / 214 | 7 / 202 |
| Whole ROI, then sclera filtering | 20 by 16 | 24 / 214 | 24 / 202 |
| Sclera support before selection | 20 by 16 | 44 / 214 | 43 / 202 |

The strict check is native score at least 0.75 and distinct-match margin at
least 0.10. Neither is a measured pixel-error bound. Sclera-first selection
improves track availability, but **no pair has more than two strict tracks**.
On the actual 202/193 ambiguous target subsets, 181/169 have a fresh admitted
predecessor, and only 35/36 have at least one strict track. The less restrictive
native matcher yields median 12/11 proposed interior matches on those pairs;
their material identity and precision are not established. More candidate
points do not establish more independent evidence.

The [native report](../outputs/roi-sclera-texture-masked-v3/README.md) includes
the complete real and synthetic results. The synthetic assay supplies known
rigid correspondences on each recorded pair's geometry, with a shared globe
radius of 2.15 iris radii and torsion 0.05 radians. All four branch combinations
are generated separately, then profiled over radii 1.5–2.8 without using truth
to choose a fit. Source/target visibility in the proposed masks and at least
six synthetic points are required. This is an ideal measurement experiment,
not reconstructed real motion or human labels.

| Provider | Perfect correspondences: correct pair ranking | Bounded 1 px/axis perturbation | Bounded 2 px/axis perturbation |
| --- | ---: | ---: | ---: |
| SAM | 802 / 802 usable cases | 751 / 802 | 671 / 800 |
| Obelisk | 764 / 764 usable cases | 684 / 763 | 608 / 762 |

There are 856 SAM and 808 Obelisk synthetic cases per perturbation level;
54/44 lack six visible points. Additional noisy cases become unavailable
when a perturbed ray misses the true nominal sphere, rather than being
silently assigned a different sign. For ambiguous targets alone, perfect
correspondences rank correctly in all 689/642 usable cases out of 724/676;
one-pixel perturbations reduce that to 642/689 and 569/641. These are forced
rankings in a declared ideal model, not reliable real sign admissions.
Floating-point projection residuals in the perfect controls remain below
0.00044 degrees. Some competing families are separated by only hundredths
of a degree, so a unique numerical minimum is insufficient confidence.

The initial control run v2 stopped on a noisy point leaving the sphere;
the completed v3 explicitly records these support failures. No noise-free
control failed. The actual matched-track stream is identical between
masked v1 and v3 (SHA-256
`90f9462432deb28a123daad41ed210c55c3ca19784a772aa9c71d08c4a1bb200`).
The unchanged whole-ROI assay is identical before/after sharing the matcher
and integral support mask (SHA-256
`60551bb565724122cfca6a85a3eb46cb87192468aa132de7bbe961b9f48e0879`).
Nine native motion/support regression tests and the exact patch-cost parity
test pass. The earlier lid-oracle result stream is also unchanged after sharing
the area-admission loader (v5/v6 SHA-256
`1583d8d6b046ccec63fdd93798bb54d1a8a19f41187d8466287bec47cf6a08d0`).

```sh
data/target/live/buttercup_roi_focus surface-sclera-motion \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_SCLERA_RUN sclera
data/target/live/buttercup_roi_focus surface-sclera-report outputs/NEW_SCLERA_RUN
```

Use `whole` or `whole-dense` instead of `sclera` for the matched feature-selection
baselines. Native RAW overlays show selected points and 20-times motion, clipped
to the image panel. Visually inspected pairs include 90892→90894 and
91887→91889, with additional masked-selection views at 90885→90887.
The evidence supports investigating a full rigid rotation with held-out
reprojection and persistent texture identity. It does **not** yet supply
near-all disambiguation: real sign truth, reliable material tracks, globe-model
error and reviewed localization remain unresolved. No model was trained or
promoted, and no SN-FEIDA improvement is claimed.

## Full rigid rotation with spatially held-out texture

`surface-sclera-motion ... rigid` extends the polar-angle diagnostic to one
proper rigid rotation for all surface points. With right-handed iris bases
`B_source` and `B_target`, its rotation is

```text
Q = B_target R_z(torsion) transpose(B_source)
X_target = G_target + R Q N_source
```

The source camera ray intersects the hypothesized front globe surface to give
`N_source`. Both globe centers follow their native conic interpretations. The
same radius, profiled over 1.5–2.8 iris radii in 0.05 increments, applies in both
exposures. Torsion minimizes squared surface-normal chord error analytically;
the training pixel error then selects the radius and source interpretation for
each target interpretation. This is not an exact joint pixel-error optimizer.

The test uses all tentative interior matches, whose quality was insufficient
to establish material identity in the preceding experiment. Each point is held
out in turn. Any training patch within a square of radius 16 native pixels of
it in **either** exposure is also excluded; a separate run uses 24 pixels.
At least three training matches per fold, six held points, and 80% held-point
coverage are required. Both target families must produce projections on that
same held set. An unavailable competing fit causes abstention, not selection
of the supported family. Translation and 2D similarity use identical folds.
No held-point residual participates in parameter fitting or outlier removal.

Fixed conditional-choice margins require winner RMSE at most 2 pixels, the
other family at least 1 pixel and 50% worse, winner no more than 0.5 pixels
worse than the better baseline, and at least 80% fold agreement on source
family and training target preference. A stable choice must agree across both
16/24-pixel exclusion distances and both mask thresholds, 179/230. These are
engineering filters, not calibrated uncertainty or physical sign truth.

The [frozen matched report](../outputs/roi-sclera-rigid-v1/README.md) verifies
that every RAW identity, source time, area admission, native correspondence,
prior polar score and prior synthetic-control result exactly matches
`roi-sclera-texture-masked-v3` after removing the newly added diagnostic fields.
All 469 provider rows on the same 243 RAW exposures participate; the actual
still-ambiguous target subset is:

| Provider | Ambiguous targets | Fresh admitted predecessor | Usable reference folds | Median best rigid RMSE | Median translation RMSE | Stable conditional choices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM | 202 | 181 | 114 | 8.96 px | 8.83 px | 0 |
| Obelisk | 193 | 169 | 113 | 9.03 px | 9.08 px | 0 |

Reference means threshold 179 and 16-pixel exclusion distance. Of the fresh
but unusable targets, SAM has 64 missing globe-family projection support,
two with fewer than six matches, and one with insufficient separated folds;
Obelisk has 51, four and one respectively. At threshold 230 only 41/33 targets
are usable with the 16-pixel exclusion and 25/22 with 24 pixels. No setting
makes even a provisional choice. The best real reference error is still
6.14 pixels for SAM and 5.55 for Obelisk. Favoring one family by a small relative
error improvement would not solve this absolute mismatch.

The same synthetic rigid correspondences provide a positive arithmetic and
conditional geometry check, on all source-admitted context pairs:

| Provider / per-axis coordinate perturbation | Cases | Usable reference | Correct target ranking | Stable correct conditional pairs | Stable wrong conditional pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM / none | 856 | 709 | 709 | 553 | 0 |
| Obelisk / none | 808 | 675 | 675 | 471 | 0 |
| SAM / 1 px | 856 | 662 | 662 | 362 | 0 |
| Obelisk / 1 px | 808 | 599 | 594 | 265 | 0 |
| SAM / 2 px | 856 | 630 | 623 | 0 | 0 |
| Obelisk / 2 px | 808 | 567 | 554 | 0 | 0 |

Here stability refers to the two exclusion distances on the original synthetic
mask support, not the real-data four-setting check. All four true source/target
branch combinations are generated. Noise-free true-target held-out RMSE stays
below 0.000234 pixels. With one-pixel perturbations its median rises to about
1.48 pixels, and with two-pixel perturbations to about 2.94 pixels; the latter
therefore fails the fixed two-pixel quality requirement. The synthetic fit
matches its generator's sphere/conic assumptions exactly. These numbers do
not establish near-all real recovery or measured tracker precision.

Source-matched review images include
[Obelisk 90878→90880](../outputs/roi-sclera-rigid-v1/rigid-90878-90880-obelisk.png)
and [SAM 90877→90879](../outputs/roi-sclera-rigid-v1/rigid-90877-90879-sam.png).
Orange circles show matched target locations; blue crosses show full-data
predictions for visual review only. Gray curves are projected native 3D globe
silhouettes, not observed anatomical boundaries. The inspected points cluster
in bright lateral patches and do not visibly establish distinct vessels.
Large, incoherent residuals remain despite the low-dimensional rotation fit.
Conic error, mistaken patch identity, lighting/reflection changes and sphere
model error are not separated by this diagnostic.

```sh
data/target/live/buttercup_roi_focus surface-sclera-motion \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_RIGID_RUN rigid
data/target/live/buttercup_roi_focus surface-rigid-report \
  outputs/roi-sclera-texture-masked-v3 outputs/NEW_RIGID_RUN
```

The completed corpus/control run took 17.2 seconds under a shared CPU claim,
not an isolated benchmark. Native builds and source-tree audit pass. Report
input hashes and the running executable hash identify the frozen evidence.
No live solver, camera connection, calibration or trained model is changed.
The result still supports only a conditional route to independent globe shape:
reliable material-point measurements could help, but these tentative tracks
do not yet supply them. Shading alone also remains insufficient on the actual
area-admitted cohort; the missing independent center and real sign validation
are not replaced by synthetic success.

## Native lid measurements and review preparation

The next full replay uses the existing native RAW constrained-arch and
nautilus detectors directly. This removes the historical learned anatomy masks
from lid extraction while keeping the same 202 SAM and 193 Obelisk ambiguous
iris inputs and the area gate. There are 225 unique RAW exposures. The native
conic is only the search seed; neither 3D interpretation selects an image edge.
Live detector behavior is unchanged.

For each detector, the diagnostic repeats extraction with four eight-pixel
seed shifts and two ±10% radius changes, leaving the evaluated conic fixed.
Both margins must retain 80% common sampled support, at most four-pixel 95th
percentile vertical disagreement and eight-pixel maximum disagreement for
every perturbation. A path with at least six points and 75% within four pixels
of the fitted limbus is also flagged as coincident evidence. This does not
prove a lid is false: actual lid contact, a duplicate limbus, or conic error
are different explanations. Such a path does not independently locate a globe
center. Distances use 2,048 samples of the fitted ellipse.

The [native report](../outputs/roi-native-lids-report-v2/README.md) and original
[replay receipt](../outputs/roi-native-lids-v1/summary.json) give:

| Iris input / native detector | Rows | Both margins returned | Compatible with predicted iris visibility | Stable through all seed changes | Limbus-coincident margin | Pass complete independence checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM / constrained arch | 202 | 16 | 0 | 0 | 5 | 0 |
| SAM / nautilus | 202 | 127 | 1 | 0 | 5 | 0 |
| Obelisk / constrained arch | 193 | 13 | 0 | 0 | 3 | 0 |
| Obelisk / nautilus | 193 | 126 | 0 | 0 | 6 | 0 |

Every supported reference path leaves both broad globe-radius families
possible with a ten-pixel conditional error allowance. Planarity makes no
reference choice. Inspection of records
[90874](../outputs/roi-native-lids-v1/native-lids-90874-sam.png),
[91911](../outputs/roi-native-lids-v1/native-lids-91911-sam.png) and
[92620](../outputs/roi-native-lids-v1/native-lids-92620-sam.png) shows paths
following interior iris texture or the lower iris boundary. The visibility
test itself uses unreviewed predicted contour samples, not human evidence:
its conflicts cannot determine which estimator is wrong. The problem is not
just sparse output; returning more points did not establish independent lids.
Noisy or coincident paths must not become center labels for a sign network.

The replay took 16.6 seconds under a shared resource claim, not an isolated
benchmark. None of 2,765 nautilus calls reached its production eight-millisecond
budget; its largest recorded internal time was 4,539 microseconds. These
dropouts therefore were not observed budget exhaustion. ROI-clipped occluders
remain separate from anatomical margins. Reporting can be regenerated from
the frozen stream without rerunning the detectors; report v2 references the
original immutable stream and adds data-derived outcome counts.

```sh
data/target/live/buttercup_roi_focus surface-native-lids \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_NATIVE_LID_RUN
data/target/live/buttercup_roi_focus surface-native-lid-report \
  outputs/NEW_NATIVE_LID_RUN outputs/roi-area-first-focus-v1
data/target/live/buttercup_roi_focus surface-lid-review \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  NEW-REVIEW-NAME
```

The native preparation recipe creates a capture-local review under
`sam31-mouse-3d-1789300371-414277114/annotator/native-lid-review-v1`.
Its [manifest](../outputs/calibration-corpus/sam31-mouse-3d-1789300371-414277114/annotator/native-lid-review-v1/manifest.json)
contains twelve targets: six evenly spaced eligible ambiguous targets per eye.
The selected records are 90896, 91550, 91904, 92062, 92374, 92654, 90889,
91537, 91889, 91919, 92073 and 92387. Each has nearest admitted before/after
RAW exposures within 300 ms, with a provider admitted across the entire
triplet. Original packed bytes, source times, crop origins and hashes are
preserved. Predictions are omitted. This is diagnostic review preparation,
not a bootstrap certificate or authorization to train from unproven selection
ancestry; the training contract still applies before any training reuse.

The canonical `paired-limbus-annotator/server.py` specified in
[AGENTS.md](../AGENTS.md) now supports an explicit eyelid scope, implemented in its adjacent
`eyelid_scope.py`. Upper/lower margin points and `not_visible` states save to
separate `*.eyelids.labels.json` files using
`buttercup-raw-eyelid-margin-label-v1`. They never enter limbus ellipse fitting
or overwrite limbus labels. Guessed points remain guessed, absent portions are
not completed, and both lids must be addressed before `SAVE + DONE`.
The API withholds a recorded prediction until independent review is complete;
edits close that barrier again. Labels are bound to the exact native RAW hash.
No reviewed labels were created by these checks.

The browser review retains the RAW mosaic as default. An optional
[neutral preview](../outputs/roi-native-lids-v1/canonical-review-neutral.png)
suppresses the period-four CFA pattern with symmetric separable
`[1,2,2,2,1]/8` filtering for display only. It does not alter stored RAW or
annotation coordinates. The three source exposures stay at native 420×280
dimensions. Five synthetic serialization/preview checks pass, covering scope
separation, prediction hiding, RAW identity changes, incomplete/invalid marks,
legacy limbus compatibility, carrier suppression and unchanged feature center.
A headless browser check verified the twelve-frame set, scope controls,
upper/lower hotkeys and both previews without writing labels. No GUI was
opened for Rob and the verification server is not a new annotation application.

The full sign goal remains unresolved. Reviewed margins would let us measure
the current localization failure and test a lid-derived center model; they
would still not, by themselves, certify the hidden globe center or physical
sign. Independent scale, validated anatomical bounds or constrained lighting,
and physical sign validation remain missing.

## Provisional visual lid portions and explicit 3D alternatives

A further bounded check uses the fixed twelve-target native review set. The
assistant inspected all twelve RAW targets and their admitted before/after
exposures without candidate geometry, then recorded sparse upper/lower
boundary estimates separately from the canonical human-label directory.
The [estimate artifact](../outputs/roi-lid-visual-raw-v1/assistant-estimates.json)
explicitly marks them as unreviewed, ineligible for training and lacking
physical sign truth. Upper dark arcs can be folds rather than margins, lateral
endpoints are uncertain and the complete eye opening is not established.
These estimates must not become human labels or assumed anatomical evidence.

The native `surface-lid-visual` diagnostic tests both sphere families against
exactly the same points. For each candidate it finds one shared radius in
1.5–2.8 iris radii that maximizes coverage of every corner of square test boxes
around those points. The continuous radius search is the existing shared
ray/sphere coverage solver. A perspective sphere silhouette is convex, so
covering every corner covers each box and every straight segment connecting
them. This supplies a conditional existence witness; the box widths are
sensitivity assumptions, not measured annotation uncertainty.

The [pilot report](../outputs/roi-lid-visual-geometry-v3/README.md) covers twelve
RAWs and 23 ambiguous provider rows: eleven SAM and twelve Obelisk. SAM record
92062 is not silently added as ambiguous merely because its Obelisk counterpart
was selected. Both candidate families cover every provisional point and its
±5-pixel box on all 23 rows. At ±10 pixels they cover all boxes on eleven SAM
and eleven Obelisk rows; at ±20 pixels the counts are ten and eleven.
The remaining ten-pixel case, Obelisk record 92073, has 98.4% corner coverage
for one family. **Failing to cover an entire test box does not exclude that
family**: the true point could lie in the covered portion. No sign is admitted.

The native planar-lid diagnostic also makes no choice on these estimates.
The best individual lid-plane intersections lie approximately 0.59–0.95 globe
radii from the globe center. They do not establish a hinge through that center.
Those planes were fitted independently; their intersection is not an exhaustive
fit under a shared anatomical hinge constraint. Imposing such a constraint
without validating its anatomy would replace missing evidence with a prior.

The source-matched pictures now show model contour samples separately from
the fitted ellipse and assistant estimates. All 23 estimates conflict with
some recorded contour samples under the existing visibility comparison; these
are unreviewed predictions, so the conflict does not say which is wrong.
The [90896 SAM sheet](../outputs/roi-lid-visual-geometry-v3/visual-90896-sam.png)
and [92073 Obelisk sheet](../outputs/roi-lid-visual-geometry-v3/visual-92073-obelisk.png)
also show oblique native 3D reconstructions at a common scale. Exact pixel rays
lift the proposed points onto each sphere; every drawn point is checked to
reproject within one millionth of a pixel. This validates projection arithmetic,
not physical geometry or the estimated margins. Inspection confirms that the
different centers and iris tilts can accommodate the same proposed image
points. Sparse estimates cannot establish that **all** true lid points would
also fit both alternatives.

```sh
data/target/live/buttercup_roi_focus surface-lid-visual \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/calibration-corpus/sam31-mouse-3d-1789300371-414277114/annotator/native-lid-review-v1 \
  - outputs/NEW_RAW_REVIEW
data/target/live/buttercup_roi_focus surface-lid-visual \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/calibration-corpus/sam31-mouse-3d-1789300371-414277114/annotator/native-lid-review-v1 \
  outputs/roi-lid-visual-raw-v1/assistant-estimates.json outputs/NEW_GEOMETRY_RUN
```

RAW bytes, context hashes, source identity and upstream area eligibility are
checked. No rejected frames are inserted, the conics and area gate are
unchanged, and there is no claim of SN-FEIDA improvement or independent scale.
Native compilation, source-tree audit and formatting checks pass. The canonical
human-label directory remains empty. This pilot gives no justification to
train a sign network from the provisional margins. The requested near-complete
real-frame disambiguation remains unverified; an independently accurate globe
position/normal measurement, or a validated relationship from lids to that
position, is still the central missing evidence.

## Embedded measurement audit and next capture

The native `surface-evidence-audit` command checks the actual archive used by
the 202/193 ambiguous inputs, including its manifest, OIM1 metadata, native
frame index and session sidecar. The
[audit receipt](../outputs/roi-sclera-measurement-audit-v1/audit.json) contains
21,132 metadata records and 2,290 frame-index records. Only `SOLID WHITE`
lightbox states occur: 6,682 metadata occurrences disabled and 66 enabled,
including the initial snapshot. These are repeated state records, not fresh
exposure counts. There is no optical-clock record for this capture. Camera
intrinsics, camera-to-scene transform and sensor-to-host clock model remain
null in all 6,747 configuration changes and the initial configuration snapshot.
Monitor pose remains estimated/preset with null uncertainty. The sidecar's
display pose is also null; EDID dimensions do not fill that gap.

This closes the possibility that already-recorded color/checkerboard sweeps
or calibrated display geometry were simply overlooked for this cohort.
It does not imply that fixed-screen brightness provides no signal, or that
every restricted candidate test needs full photometric-stereo rank. It means
the tested sufficient known-light measurement is not supplied by this archive.

The [measurement specification](sclera-measurement-requirements.md), also
copied to `/tmp/requirements` as requested for missing recording evidence,
describes a sufficient acquisition recipe and the separate independent
reference needed for validation. It preserves native RAW, successful-presentation
identity, two-second acclimation, distraction allowance and temporal uncertainty.
No live capture, training or camera connection was started. The canonical
human-label directory for this capture still has no reviewed labels.

```sh
data/target/live/buttercup_roi_focus surface-evidence-audit \
  outputs/roi-area-first-focus-v1 outputs/roi-lighting-resegment-v1 \
  outputs/NEW_MEASUREMENT_AUDIT
```
