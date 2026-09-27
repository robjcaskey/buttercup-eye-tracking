# Reflected-light and sclera shading diagnostic

This offline experiment tests whether the two circular-iris interpretations
predict different lighting across visible sclera. It reruns the exact ambiguous
RAW exposures through pinned SAM3.1 and the existing Butter Obelisk RAW16 model,
then compares the two poses of each fresh ellipse. This is historical model
comparison, not training, promotion or a cold-bootstrap certificate.

Corneal reflections can encode incident illumination when corneal geometry is
specified; reconstructing that geometry and separating iris texture remain
important limitations. See [Nishino and Nayar, Eyes for Relighting](https://www.cs.columbia.edu/cg/pdfs/19-Nishino_TOG04.pdf).
Diffuse shading integrates lighting over directions, so it cannot recover all
the information present in a specular reflection. See [Ramamoorthi and Hanrahan,
radiance and irradiance](https://graphics.stanford.edu/papers/invlamb/).
The model below is our bounded approximation, not a reproduction or validation
of those papers' complete methods.

## Native commands

Build `buttercup_calibration_sign` with `--features sam31`, using the installed
LibTorch runtime as in `scripts/run-viewer.sh`. Use a coordinated GPU/CPU claim
around material work. The first command uses the checked SAM export on CUDA
and explicitly loads Obelisk on CPU. The second needs no Torch/CUDA dependency.

```sh
data/target/live/buttercup_calibration_sign roi-resegment \
  /tmp/buttercup-roi-ellipses-with-area.bin \
  outputs/roi-focus-neural-eval-v2 \
  outputs/calibration-sign-sam-cold-20260918-v1/sam-export \
  outputs/raw-student.DUYOAP/raw.ot outputs/NEW_SEGMENTATION
data/target/live/buttercup_roi_focus lighting \
  outputs/NEW_SEGMENTATION outputs/NEW_LIGHTING
```

The optional final integer on `roi-resegment` limits a pilot to the first N
exposures. Frozen RAW hashes, native source indexes, crop/clock/sequence/range
identities and model hashes are checked. The generic `eye_student_v1.ot` asset
is an older RGB model; the command requires the RAW16 architecture and matching
weight-manifest hash. It calls shared SAM inference, Obelisk preprocessing and
inference, shared contour fitting and unchanged native RAW support gates.
Stored masks retain failed fits. No missing fit is replaced by a held result.

## Geometry and measured samples

For each fresh ellipse, keep both shared perspective-circle interpretations
`(C, n)` in iris-radius units. A spherical surface of radius R intersecting the
iris plane in that unit circle has center `G = C - sqrt(R²-1) n`. Use separate
globe and corneal radii; treating the entire eye as one reflective sphere would
erase the corneal bulge. The nominal globe radius is 2.0 and corneal radius 1.3.
Sensitivity settings are globe 1.8/2.0/2.2, cornea 1.2/1.3/1.4. These are
engineering priors, not measured anatomy or certified anatomical bounds.

Intersect each source pixel's camera ray with the front sphere. Its outward
surface normal is `N`; the unit direction from that point to camera is `V`.
For a specular highlight the direction toward the light is `L = 2(N·V)N - V`.
The weighted mean of connected bright corneal samples supplies one dominant
direction per pose. The reflection half-vector identity is checked numerically.
Multiple lights, a large nearby source, glasses and an aspherical cornea can
invalidate this approximation. The reflection is not a calibrated light probe.

Quantitative measurements use unaveraged RAW10 green photosites, separately at
absolute sensor CFA phases (2,0) and (0,2). No demosaiced display brightness,
gamma correction or per-frame white balance enters the fit. Missing physical
black/white level, exposure and gain calibration remain missing.

Sclera proposals come from Obelisk's existing visible-sclera channel, shared
for both segmentation providers and both pose hypotheses. Evaluate thresholds
0.7 and 0.9; restrict samples to the same lateral annulus outside the ellipse,
excluding clipped codes. This is an unverified learned mask, not human sclera
truth. At least 32 samples and adequate vertical span are needed. Both poses
are evaluated on the same common surface support. If fewer than 90% of the
observed samples fit both globes, show the diagnostic shading but refuse a
photometric sign choice. Record per-candidate containment and omitted fraction.

## Fit, controls and admission

Fit `I = ambient + gain * max(0, N_sclera · L)` with nonnegative gain, robust
residual weights and one shared observed sample set. Train on alternating
24x24 sensor-coordinate blocks and score the other blocks, then swap. Compare
against constant brightness and a simple image-plane brightness gradient using
the same splits. A better fit alone is not proof of orientation.

The nominal conditional choice requires error separation of at least 10% of
the constant-control MSE and at least 20% improvement over that control, plus
the support/containment gates. A stable choice requires at least 24 of the 36
shape/mask/CFA settings to qualify for the same candidate and none to qualify
for its opposite. A surface-specific choice additionally improves on the
image-plane control by at least 5%. All individual settings and abstentions
remain in `lighting.jsonl`; thresholds are heuristic, not probabilities.

The initial pilot failed containment on many settings. Its follow-up exposes
common-support shading even on those failures without relaxing the 90% sign
admission gate. Keep both pilot outputs to distinguish rendering/diagnostic
coverage from admitted evidence.

Review sheets show original RAW, measured sample locations, both predicted
shading fields, candidate normals and inferred light directions. White/cyan/pink
plots compare observed/A/B brightness across image height. Color backgrounds
are display-only demosaicing. Up/down means camera coordinates, not world
vertical. Unknown eyelid shadow, skin contamination, sclera reflectance,
refraction and source geometry remain potential explanations for disagreement.

Both signs share each input ellipse and its frontal-equivalent area. This
experiment adds no independent scale and makes no SN-FEIDA improvement claim.
The corpus is Rob-only, with no verified physical sign labels in this subset.
