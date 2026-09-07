# Meridian continuity and completed calibration

This follow-up removes the requested post-calibration sign-epoch interlock and
fixes a reproduced near-frontal branch-correspondence failure. It does **not**
establish that every recorded gaze sign is correct, repair a misplaced iris
ellipse, or validate physical monitor pose.

The preceding acquisition workflow and user-led calibration attempts are in
[sign-acquisition-trials.md](sign-acquisition-trials.md). At that earlier
handoff, completed mappings were preserved but their cursors paused after an
epoch change. The policy below supersedes that interlock.

## Three events that must remain different

| Event | Interpretation and behavior |
| --- | --- |
| A projected gaze component crosses zero | Ordinary continuous eye motion; not inherently a branch correction or a new epoch |
| Independent evidence changes the selected normal hypothesis | A sign correction; retain its epoch/provenance diagnostics |
| Eye, provider, SAM prompt, or authority generation changes | Different observation basis; do not silently reuse an incompatible completed mapping |

Completed calibration now uses the current **signed** gaze with its existing
mapping even if the sign epoch changes. This applies to the calibration result
cursor, its wireframe ray, the normal viewer mapping, and desktop gaze mapping.
The original training epoch is not relabeled as the new epoch. A real sign
reversal can consequently jump the cursor; removing the interlock is not a
claim that the old fit remains accurate under a changed sign convention.

Missing/unsigned observations and incompatible sources remain rejected.
Duplicate or older source timestamps do not advance the completed cursor.
Provider changes still suspend a completed calibration. Unfinished calibration
still discards mixed-epoch clusters; none of these changes admits an unsigned
sample to training or restarts an already completed sequence.

Scene metadata retains `training_sign_epoch` and the current surface epoch,
with `completed_sign_epoch_policy: continue-with-current-signed-gaze`.
`sign_diagnostics.near_frontal_continuation` identifies a correspondence
adjustment on that exact surface source. Held surfaces do not gain fresh votes.
No new user setting, camera change, or saved monitor update is involved.

## Reproduced geometric failure

An ellipse axis is unoriented: angles separated by pi describe the same
ellipse. At a circle, even that axis becomes unobservable. The old assignment
matched the two hypotheses by the nearest transported **unit transverse
direction**. When a physical crossing of camera-normal fell between exposures,
that assignment could reflect the trajectory back onto its old side. The
separate motion window eventually corrected it, but only after several more
frames and an epoch change.

The strengthened pre-fix radial test reached a maximum projected-normal error
of 0.27. Merely testing a trajectory that includes an exactly circular frame
missed this failure; arbitrary ellipse-axis parameterization could accidentally
carry the desired answer through the singularity.

`eye_scene_model/sign_continuity.rs` now performs a bounded local correspondence
check, called only for an already signed eye with reliable independent RAW
transport. It uses two prior source-timed samples and two current hypotheses;
there is no new image history, allocation, or combinatorial search.

The check requires increasing known source times, adjacent intervals between
1 ms and 750 ms, a current/prior interval ratio at most 2.5, and both prior and
current transverse normal magnitudes at most 0.12. It never seeds an unknown
sign, changes the conic, relaxes convex/camera-facing validity, or silently
rematches arbitrary far-from-frontal pivots.

For transport residual `r`, the pixel support margin is `max(2*r, 0.5)`. The
winning pivot must be within that margin plus 0.015 frontal radii of its
transported reference, and beat the alternative by at least the margin. When
the preceding branches are distinguishable, source-time projected-velocity
extrapolation must also prefer that candidate. The new and old projected
directions must straddle a meridian rather than merely continue on the same side.

An additional noisy, slightly off-axis test exposed an important limitation of
using a single previous branch endpoint. If the transported previous pivots are
closer than the support margin, their chosen near-circle direction cannot
reliably veto the next frame. In this case the reference is the midpoint of
their bounded pivot support, not the arbitrarily selected endpoint, and its
velocity extrapolation is not treated as decisive. The current winner must
still meet the separated-support and absolute-residual tests. Published gaze
rays are **not** averaged together. This also permits a genuine turnaround at
straight-on; approaching the camera axis does not force a crossing.

These are engineering support allowances, not calibrated probabilities or
anatomical limits. The effective pivot remains approximate and movable. The
check deliberately does not cover fast saccades that skip the entire small
near-frontal band, untrusted transport, long gaps, or incorrect segmentation.

## Synthetic and application regression tests

The four meridian tests exercise 312 modeled trajectories, 28 source frames
each (8,736 frames): horizontal, vertical and oblique motion in both directions;
straight-through crossings at five exposure phases; slight misses of the
camera axis; and turnarounds. Equivalent pi-separated ellipse axes alternate,
and an exactly circular frame has an arbitrary axis. The tests include an ROI
nudge, independently prescribed scale, head translation, small additional
effective-pivot drift, 400 ms delivery delay, 100/250 ms source cadence, and
0.1/1.5 px declared transport residuals. Pupil anchors appear only in the first
four frames; no target position selects a later sign. The generating eye depth
uses continuous physical scale, independent of the tracker's quantized area
family and fitted pivot.

All retain the physical branch and epoch. Maximum projected-vector error is
0.0100 for radial crossings and 0.018868 for the near-miss set, below the 0.02
test bound. These dimensionless errors are not screen accuracy or degrees.
They allow a small ambiguous source near coalescence, not a delayed growing
reflection. These tests vary declared transport uncertainty; they are not a
measured distribution of human image/segmentation noise.

Unit controls reject missing/duplicate/late clocks, long gaps, conflicting
velocity outside coalescence, large pivot residuals, noisy indistinguishable
current candidates, invalid numeric input, and a general far-from-frontal
rematch. Coalescent pivot support cannot privilege one arbitrary prior endpoint.
Application tests exercise actual signed feature reversals across several
epochs while preserving the completed fit, source freshness, wireframe ray,
and normal/desktop affine mapping. Provider and unsigned-source rejection and
unfinished-calibration restarts remain covered.

The complete SAM-enabled viewer suite reports **924 passed, 41 failed,
24 ignored**. The 41 failing names exactly match the preceding 916-pass suite;
this is not a green full suite. The optimized viewer build, whitespace check,
and source-tree audit pass.

## Matched corpus evaluation

Artifacts are under `outputs/sign-interlock-removal.qRbBoq/`. The baseline
binary is the previously live acquisition-enabled build, preserved before
this work. Final candidate reports are named `final-candidate-*.json` and
`full-final-candidate-*.json`; `matched-comparison-final.json` contains the
exact source-matched comparison. Earlier candidate reports are development
runs, not the final comparison.

The five complete recent recordings under `outputs/calibration-corpus/` are
`sam31-mouse-3d-` followed by these suffixes:

- `1788786652-056177751`
- `1788786721-601704506`
- `1788786787-329863888`
- `1788786821-286447565`
- `1788787050-568039862`

The complete older recording is
`outputs/raw-eye-hotkey/both-eyes-1788766324-951655107.tar`, extracted at
`outputs/sign-audit-20260907/capture`. This evaluation uses the whole archive,
not just the earlier first-1,000-per-eye subset.

| Subset / eye | RAW ROI frames | Fresh conic sources | Signed / contained, baseline = candidate | Epoch changes, baseline = candidate |
| --- | ---: | ---: | ---: | ---: |
| Five recent, right | 1,133 | 422 | 366 / 364 | 4 |
| Five recent, left | 960 | 252 | 245 / 212 | 2 |
| Historical complete, right | 8,719 | 3,136 | 3,006 / 2,413 | 6 |
| Historical complete, left | 6,461 | 2,930 | 2,764 / 2,180 | 9 |

There are 17,273 ROI frames and 6,740 distinct attempted conic sources, not that
many independent sensor exposures or people. The table uses the live
`MotionWindowFallback` policy. All four replay policies have identical
baseline/candidate fresh outputs and state, except the added false diagnostic
field; none triggers the new near-frontal rule. This is a regression check, not
a measured improvement in real-eye sign accuracy. "Contained" checks the
source/presentation ROI margin, not the complete calibration state machine.

Inputs are frozen recorded ellipses/pupil cues, with source-aligned native RAW
motion and independent relative scale. No new SAM inference or fitting is
performed. Replays start cold at actual discontinuities and use the recorded
publication schedule rather than applying predictions retroactively. They
exclude unarchived sources (30 publication rows across the subsets) and count
held observations only once. Missing fits remain missing. Historical metadata
lacks host-ready clocks for 12,023 replay publication rows, which use source
cadence; internal SAM consumption times and pre-recording state are unavailable.

### SN-FEIDA and localization

[Scale-normalized frontal-equivalent iris disk area (SN-FEIDA)](flat-tire-area-and-motion.md)
uses the RAW transform scale, never the candidate radius. It is supported for
4,951 fresh inputs, with 2,858 adjacent comparisons within 500 ms on the same
short scale-reference chain. Reference changes and unreliable motion break
the chain; unrelated references are not pooled as anatomical areas.

| Subset / eye | Scale-supported inputs | Same-reference pairs | Mean absolute log SN-FEIDA step |
| --- | ---: | ---: | ---: |
| Recent, right | 366 | 248 | 0.036712 |
| Recent, left | 205 | 133 | 0.040875 |
| Historical, right | 2,422 | 1,445 | 0.080014 |
| Historical, left | 1,958 | 1,032 | 0.063560 |

Candidate/baseline area and SN-FEIDA differences are zero. Output area versus
fixed input differs by at most about `1.46e-11 px²` from rounding. This is
expected for unchanged ellipses: area cannot establish sign correctness or
reward a frozen/wrong fit. The synthetic tests also preserve SN-FEIDA against
an independently prescribed scale. No human limbus/sign labels, independent
metric scale, calibrated uncertainty, or new localization score are available
for these recordings. RAW scale is independent of candidate radius, not of
every iris pixel.

## Native video review and remaining failures

The source-time audit selects events, not ground-truth motions. It finds 36
small component crossings in the recent captures; all recent surfaces are
outside the new near-frontal band. The historical archive has only four
individual near-frontal sources. None supplies the qualified signed,
source-contiguous sequence needed to validate the new rule in real imagery.

Six short native-RAW sequences were decoded with the existing custom
RAW10/Quad-Bayer code, using fixed per-clip display levels, exact conic exposure
alignment, and no held prediction in place of a missing fit. Their half-speed
MP4s and inspected frame strips are in the artifact directory:

- `recent-right-axis-crossing`: a small horizontal-component crossing with
  continuous iris geometry and epoch zero. A bright reflection crosses the
  pupil. This could include estimator jitter about zero, not necessarily a
  deliberate fixation change.
- `recent-left-crossing-roi-move`: the iris stays registered across a crop
  shift from sensor origin (4308, 3192) to (4292, 3180), with a small component
  crossing and no epoch change. The frames are visibly soft/noisy.
- `historical-right-vertical-crossing` and
  `historical-left-vertical-crossing`: vertical components cross from negative
  to positive at sequences 5527–5529 and 4385–4388, respectively, without
  changing epoch one. The left crossing includes an ROI shift. The iris remains
  visible, but upper-rim occlusion and noticeable fit/axis noise prevent treating
  these small component crossings as independently labeled physical gaze truth.
- `historical-left-epoch-3849`: an approximately 58-degree reported step and
  signed-to-unsigned epoch transition. The image conics are similar; this is
  not evidence of a legitimate physical near-frontal crossing.
- `historical-left-epoch-4759`: an approximately 113-degree reported reversal
  accompanies a conic visibly beside the iris, on bridge/skin. The reconstructed
  overlay was checked against the original published limbus points. Removing
  a cursor interlock cannot repair that upstream localization error.

The two earlier successful calibration archives closed before the live log's
post-success epoch changes. Their exact RAW transitions therefore remain
unclassified; the new synthetic failure must not be presented as their proved
cause. Replaying old published bad fits is also not proof that a fresh current
SAM run would reproduce them.

Independent evaluations ran concurrently under shared CPU/memory-bandwidth
claims. Logs retain coordination evidence; their runtime figures are not an
isolated performance benchmark. Desktop pointer injection and focus-following
were disabled, the AFK live viewer stopped, and saved monitor defaults left
unchanged. The optimized build is ready for a user-led retest. That retest still
needs a recorded deliberate vertical/horizontal sweep and a turnaround near
camera-normal, including post-calibration motion, before claiming human sign
accuracy or that all cursor dropouts are fixed.

## Reproduction

Use the normal SAM-enabled native-library environment and coordinate shared
resources before building/running these CPU evaluations. No model inference or
camera connection occurs in the offline command. For either build and either
eye, use a fresh output filename (the reporter uses `create_new`):

```sh
cargo test --offline --features sam31 --bin buttercup-eye-viewer meridian_crossing
cargo test --offline --features sam31 --bin buttercup-eye-viewer sign_continuity
cargo build --offline --profile live --features sam31 --bin buttercup-eye-viewer
data/target/live/buttercup-eye-viewer --offline-sign-acquisition-trial \
  outputs/new-meridian-report.json subject-right --arrival --cold \
  outputs/sign-audit-20260907/capture
```

Use the five extracted `capture-…` directories in the artifact directory as
the final arguments for the recent subset. The runtime `compare-replays.py`
matches clocks, source timestamps, inputs and fresh output/state, excludes
runtime microbenchmarks, and computes only same-reference SN-FEIDA steps.
`audit-crossings.py` and `render-crossing.py` retain event selection and native
review procedures. These local scripts/artifacts and captures are runtime data,
not checked-in source or binaries.
