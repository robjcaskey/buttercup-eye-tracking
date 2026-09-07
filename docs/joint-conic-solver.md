# Source-aligned joint conic gaze solving

Development status: implemented core and live adapter; **full-corpus validation
and boundary-selection fixes remain in progress**. Do not equate the synthetic
proofs, build success, or partial replay with demonstrated gaze accuracy.

## One target, not two gaze points

`conic_solver/joint.rs` optimizes one latent camera-frame fixation together
with bounded nuisance geometry for each available eye. Each eye's gaze ray is
the normalized vector from its fitted center to that **same target**. Actual
outer-limbus, inner-limbus and pupillary-boundary samples constrain perspective
projections of a nested 3D circle family. There is no arithmetic or weighted
average of independently solved gaze points in this path. The synthetic
comparison with such an average is explicitly a prohibited-method control.

Prefitted conics initialize alternate circle-normal hypotheses. They do not
add a second observation factor for the same pixels. Exact intrinsic-normalized
conic decomposition supplies both camera-facing mirror starts, including
off-axis and unequal-focal-length synthetic cases.

The bounded model currently has three target parameters and eleven parameters
per participating eye: center XYZ; outer, inner and pupil radii; two pupil-plane
decentration coordinates; inward pupil depth; and two surface-axis alignment
angles. These angles distinguish the imaged surface normal from the fixation
ray. They are not a calibrated clinical visual-axis offset. The distinction is
consistent with established eye-tracking models, but the numeric bounds here
are provisional engineering choices, not derived population intervals.
[Primary model discussion](https://pubmed.ncbi.nlm.nih.gov/16761839/).

## Sparse evidence and constraints

- Retained SAM contour runs supply actual, non-flat-tired source samples.
- RAW pupil profiles search the proposal's own untouched exposure; a fitted
  ellipse is only a search guide. Flat/saturated/no-edge data yields no arcs.
- Alternatives from one profile sector share a correlation group. The solver
  selects one alternative per group, never votes once per resampled pixel.
- Point residuals use a robust loss; grossly incompatible whole groups have
  a capped cost and zero pulling force. Their rejection remains in diagnostics.
- Native sensor origins remain attached, so a crop translation is not measured
  eye motion. Neither a rendered ellipse nor a held prediction becomes evidence.
- All projected circles must face the camera, be in front of its optical plane,
  obey the provisional projection envelope and retain bounded nested radii.
- Independent effective-pivot support is optional and movable. The live prior
  currently leaves it absent rather than deriving a prior from the chosen sign.

Core caps are 24 hypotheses, 16 refinements, 32 groups per eye, four alternatives
per group and 16 points per arc. The live/evaluation callers use 16 hypotheses
and 12 refinements. These are work budgets, not latency guarantees.

An important optimizer correction projects trial radius updates onto the
**already bounded** nested-radius set. Simply rejecting a trial whenever an
unobserved inner radius crossed the observed outer radius could freeze both
eyes' refinement. The projection does not widen any anatomical bound.

## Source time and live routing

`binocular_coordinator/source_pairing.rs` retains at most 32 sparse packets per
ROI across a 1.5-second source-time span. Only identical source timestamps in
an explicitly matching stream epoch pair. Local ROI sequence numbers may
differ. Delayed, duplicated, conflicting, expired, or previous-epoch results
cannot create a new synchronized pair. Missing eyes remain optional.

`gaze_target_solver/joint_tracking.rs` runs the shared objective on those packets.
A previous shared target can initialize a later optimization; it is not a
residual, smoothing operation, or fresh sample. Host completion/redraw time is
not used as exposure time. The 2 ms row-time allowance is an explicit engineering
assumption, not an attested bound on the sensor's rolling exposure.

With the existing `3` hotkey enabling the second ROI, SAM frames now use this
joint path. The second ROI remains off by default. Provider changes advance
both calibration-authority generations, preventing an existing monocular fit
from silently being treated as calibration of a new observation model.
Unavailable joint evidence does not fall back to an unrelated native pupil.

The contact uses the fitted **surface normal**. Cursor calibration, plotter
mapping and laser direction use the **shared fixation ray**. The estimated
camera-frame target/centers are exported in millimeters, separately from the
existing eye-reference monitor frame in inches. No measured transform between
those frames is asserted. Scene export includes both source keys, participating
eyes, group residuals/rejections, intrinsics/prior provenance and alternate cost.

Sign admission is still heuristic: a separated competing-basin cost margin is
required along with both-eye support or noncoplanar pupil support. An outer
ellipse alone cannot acquire sign merely because one numerical start won.
This is not a calibrated probability or a completed temporal-sign model.

## Scene support is defeasible

The current live/evaluation pinhole starts with focal lengths 4000 native pixels
and principal point [4000,3000] on the 8000×6000 sensor. It is **not** checkerboard
calibration. Available MediaPipe scale support is preferred; otherwise a nominal
64 mm interocular span, or a 350 mm monocular range, supplies a broad prior.
Depth uncertainty follows the viewing ray rather than incorrectly fixing an
off-axis point to a narrow Cartesian XY column.

Neither fitted iris radius nor the candidate target sets its own independent
scale. The live model does not yet include corneal refraction, accommodation,
learned user-specific vergence, calibrated axis alignment, or anatomical torsion.
Posterior covariance remains absent. A finite metric target is conditional on
these priors, not a measured location.

## Corpus inventory and current evidence

Runtime artifacts live under `outputs/dual-eye-joint.MiR1Iw`, never in Git.
`inventory-stereo-corpus.py` scanned 198 archives and 87 extracted indexes.
`prepare-stereo-replay.py` examined every index with stereo timestamps, including
incomplete archives and recovered staging copies. Exact RAW hashes plus source
lineage and geometry identify duplicates; matching index text alone does not.

The authoritative `replay-inputs-v2/manifest.json` accounts for 387,519 unique
available ROI exposures: 166,626 same-read RAW pairs and 54,267 single-ROI reads.
Missing payload receipts and truncated auxiliary metadata are explicitly
reported; they are not replaced with predictions. The `sam-pilot` export and
two full-range `sam-all` jobs together cover the requested index ranges, but
the two full jobs are **still running**, not validated complete.

The first inspected matched comparison uses 20,000 exposures (source indices
100–10099 and 193759–203758), 12,994 reads, 7,006 RAW pairs and 26 capture entries.
Only 3,204 reads supply both current extractor packets. This is stateless,
fresh-SAM component replay; it does not reproduce live frame dropping or SAM
video-memory publication byte for byte. Labels and recorded gaze never select
the candidate.

| Metric on this subset | Before radius-step fix | After |
| --- | ---: | ---: |
| Available shared-target solves | 7,633 | 7,638 |
| Both eyes contributing | 3,067 | 3,110 |
| Joint-minus-monocular withheld-error regressions over 1 px, right | 705 | 436 |
| Same diagnostic, left | 1,000 | 586 |

The last two rows compare each algorithm with its own matched monocular arm;
they are **not** independent gaze/localization accuracy or identical acceptance
sets across versions. Reports also retain all rejected-arc residuals and compare
exactly common accepted groups separately. All 20,000 exposures lack recovered
independent scale hints, so absolute **scale-normalized frontal-equivalent iris
disk area (SN-FEIDA)** is unavailable there. Do not replace it with candidate
radius normalization. See [its definition and caveats](flat-tire-area-and-motion.md).

The post-fit label audit finds 16 reviewed native labels after excluding
assistant, backup and unreviewed documents; ten match available stereo RAW
bytes and geometry exactly. None falls in this first 20,000-exposure subset.
Those labeled exposures still need evaluation. Visible and guessed landmarks
are separate, and limbus localization is not gaze ground truth.

The full viewer suite currently has 945 passing tests, the same 41 failures as
the pre-change baseline, and 24 ignored tests. Live-adapter tests additionally
check source deduplication, different ROI sequences on one clock, crop transport,
missing-eye behavior, provider changes, radius units and shared gaze mapping.
No live user calibration or desktop-pointer trial has been performed for this
new path.

## Known failures and remaining work

Native RAW inspection of pair [7358,7359] exposes a serious failure: a few
compatible upper-eyelid arcs can leave a fitted ellipse above the iris, despite
other groups being rejected. Pair [5460,5461] also contains rejected upstream
fits and large contour disagreements. These are not successful localizations
merely because an accepted subset has a small residual.

Still required: stronger spatial/boundary support accounting; useful partial
evidence when a full upstream conic cannot be fitted; post-fit human-label
scoring; independent source-motion/scale/SN-FEIDA sequence analysis; inspection
of all completed corpus results and regressions; and live/source-replay checks
of every presentation/calibration consumer. Full implementation validation is
not complete until those checks are actually run and inspected.

Reproduce with the repository runtime environment:

```text
viewer --offline-stereo-sam-export frames.jsonl cache.jsonl START COUNT
buttercup_stereo_conic_eval output.jsonl cache.jsonl...
report-stereo-conics.py output.jsonl report.json --expected-manifest manifest.json
score-stereo-labels.py inventory.json frames.jsonl labels.json output.jsonl...
buttercup_raw10_preview --source-index frames.jsonl INDEX comparison.png output.jsonl
```

`--max-frames-per-cache N` permits bounded prefix comparisons of caches whose
writers have already completed that many records. It never counts the rest of
the corpus as evaluated. The coverage checker rejects missing, duplicate and
out-of-range source indices. RAW comparison panels are custom-decoded native
RAW / magenta SAM / green joint / yellow monocular; no ImageMagick is involved.
