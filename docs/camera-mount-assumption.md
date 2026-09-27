# Temporary camera mounting assumption

For now, Rob's normal upright, screen-facing setup uses **camera below eyes**.
The viewer defaults to this mode when no saved mounting selection exists.
`F8` cycles flexible → below eyes → above eyes when no expected mode is pinned.
`outputs/settings/camera-mount.json` may also contain `"expected_mode":"below-eyes"`.
Rob's current setup pins this expectation. F8 and control-socket changes that
conflict with it are rejected before saving or changing the active solver;
the UI displays the conflict. Deliberately reconfiguring the rig requires updating
the expected mode in that file, separately from cycling the solver mode.

Startup rejects a saved mode that conflicts with its expectation before camera
initialization. Calibration entry independently checks the same invariant before
arming RAW capture. Invalid or unreadable settings are errors, not silent defaults;
a missing file retains the operating default. Legacy settings without an expected
mode keep their previous behavior. Control and UI status expose both the actual
and expected modes. This guards an explicitly selected solver prior, not a measured
physical pose or a theorem that below-eye camera placement fixes gaze sign.

This is a temporary operating assumption while better evidence-based sign
selection is developed. It is not a universal camera-placement theorem or a
claim of nearly perfect signage. Long term, signed direction should be robustly
supported by independent pupil/limbus evidence, source-time motion and compatible
two-eye geometry, validated on matched RAW with known limitations. Improved
signage should eventually make this setup-specific prior unnecessary.

## Current native stereo behavior

Below-eyes now uses **screen orientation, then conditional continuation**.
`M` first displays a stationary top-center plus. Source exposures from the first
2.1 seconds are excluded. The solver intersects a screen-reference half-space
with the same density used for MAP selection and posterior integration. Each eye
needs six unique posterior-supported observations spanning at least one second,
with a coherent angular winner over competing fixations. A partner arriving for
the same exposure does not cast another vote. The live settling clock starts at
the current RAW frontier after a successful target submission, not at the timestamp
of an old inference result.

Once initialized, an eye uses a 45-degree continuation neighborhood around its
last supported direction. Routine tracking has no screen-position or sensor-Y
clamp. A 1.5-second support gap releases that eye's reference. During continuation,
every eighth fresh solve also checks the unrestricted MAP objective. Three fresh
strong contradictions release the reference instead of flipping it silently.
Existing posterior support and numerical-integration gates still apply. The
recorded posterior labels the prior explicitly; a remembered direction is never
counted as independent sign evidence.

While stationary calibration targets are presented (after orientation, before a
display fit), the tracker is in a **screen-fixation** phase. An eye that still
holds a reference continues unchanged. An eye whose reference expired falls back
to the same screen half-space and must win the same six-source, one-second vote;
nothing is cleared. This is needed for Butter Obelisk, whose multi-second contour
dropouts otherwise exhaust the one reorientation per target. On 2026-09-26 matched
Obelisk replays (below-eyes, counterfactual orientation, harness
`BUTTERCUP_SCREEN_FIXATION_TRIAL=1`) of sessions `1790075989` and `1790076170`
removed the repeated-acquisition failure. Ready rows for the affected eye rose
from 36 to 152 and from 143 to 217, with zero published normals on the mirror side
and no Ready-to-Ready jumps over 45 degrees. Session `1790076043` never lost its
reference and was unchanged. Outputs are under `outputs/obelisk-sign-flip-20260926`.
It does not add contour coverage: in `1790075989` both ROIs crop the lower iris,
and RAW Obelisk fits one eye in only 14 of 236 exposures.

A surviving reference is also kept alive by fresh, contributing,
continuation-conditioned fits whose posterior 90% radius is at most 30 degrees
(`continues_direction`). Calibration samples still require the 15-degree
`supports_direction` gate, and continuity alone never initializes a reference.
During screen fixation the periodic unrestricted audit also runs, so three
strong contradictions still release the reference. In the first live stereo
Obelisk session (`1790436305`), about half of the contributing fits had
15–35-degree radii. Without this change the reference expired during them and
was later re-earned on the mirror branch (36 mirror-side Ready normals and one
jump over 45 degrees for that eye). With it, that eye kept its reference: no
re-entry, Ready rows rose from 256 to 370, there were 2 mirror-side normals and
no jumps. The other eye still re-entered acquisition after a 9-second Obelisk
dropout, caused by specular glare across the iris.

While an on-screen stimulus is presented (`ScreenAndContinue`), every normal must
satisfy both the 45-degree continuation cone and the screen half-space. Relative to
the last normal alone, a chain of imprecise fits can walk a reference onto the
mirror branch. That walk was observed in stereo Obelisk session `1790436833`:
normal Y went from −0.46 to +0.62 in about 2.6 seconds with no jump over 45 degrees
between Ready rows. On that replay the half-space score of every Ready normal with
Y < −0.2 was at least +0.34, and every walked mirror normal was at most −0.38
(limit −0.34). With the bound, mirror-side Ready normals fell from 26 to 0. Routine
`Continue` tracking still allows off-screen gaze.

The stereo selection (Shift+3 / STEREO button) is now saved in
`outputs/settings/stereo-solver.json` and restored at startup. Before, every restart
silently calibrated with the monocular virtual-contact path. Session `1790436738`
was such a run: its pupil-anchor sign flipped the right eye vertically from target 6
onward, and target 9's mirrored feature made the display pose unsolvable.

Outside orientation and calibration, the live viewer enables **routine
acquisition** (`AcquireAndContinue`). An eye without a reference may earn one
only under the screen half-space and the same six-source, one-second coherent
vote. An eye that has a reference continues within the 45-degree cone without
the screen bound, so off-screen gaze remains allowed, and the periodic
unrestricted audit still releases contradicted references. Before this, a
restarted viewer had no way to establish direction outside `M`/`Shift+M`, so a
reloaded calibration was never applied. Offline replays keep the historical
behavior unless they enable it.

During calibration, a precision-only stall keeps the current stationary target
and its previously admitted samples. Uncertain sources remain ineligible. The
three-second recovery timer now requires a current, same-epoch native report
that the orientation reference is explicitly unavailable; an unknown legacy
reference state does not assert loss. Initial acquisition, one recovery per
target, provider/epoch invalidation and overall time limits remain enforced.
The UI says it is waiting for a precise sample instead of automatically asking
for orientation again after three seconds of low confidence.

`Shift+M` repeats orientation while retaining an available display mapping;
without a saved mapping it performs normal calibration. Moving the camera can
invalidate that mapping: use ordinary `M` to refit it. Reorientation alone does
not track a moving monitor/camera transform. The new policy belongs to the native
joint conic pipeline. Live monocular trackers use flexible geometry in BelowEyes;
they do not secretly keep the old sensor-Y clamp or receive this stereo prior.

### Conditional geometric bound

In camera coordinates X-right, Y-down, Z-toward-camera, the reference plane through
the eye E, camera and projected horizontal has unit normal proportional to
`[0, E.z, -E.y]`. The current implementation admits gaze normals n with
`h dot n >= -sin(20 degrees)`. The 20-degree allowance consists of 15 degrees
of projected-axis alignment uncertainty and 5 degrees of model error.
**Physical camera roll within 15 degrees does not certify this projected-axis
bound under arbitrary yaw.** This remains a conditional engineering assumption,
not a completed mathematical guarantee for Rob's entire requested placement range.
No EDID fallback or nominal camera intrinsics certifies an unmeasured camera pose.

### Legacy controls and provenance

The low-level BelowEyes `supports`/`branch` filter remains sensor-normal Y < -0.05;
AboveEyes remains Y > +0.05. These are explicit historical controls, and AboveEyes
still uses that legacy policy live. Native `JointTracker` defaults to the new
BelowEyes phase policy unless `legacy_mount_filter` is explicitly selected.
Older standalone contact/component replay paths retain their recorded legacy
mount filters. A matching `camera_mount_assumption` alone no longer identifies
the algorithm: compare direction-prior provenance and the experimental flags too.

The recorded calibration test uses `BUTTERCUP_SCREEN_REFERENCE_TRIAL=1` for the
new phase policy and an explicit legacy control otherwise. Recorded masks/RAW
without an actual top-reference stimulus provide a **counterfactual** orientation
trial, not proof that the user followed the new protocol. Offline mounting
selection still defaults to below-eyes; `BUTTERCUP_OFFLINE_CAMERA_MOUNT` accepts
below-eyes, above-eyes or flexible and rejects invalid values. Debug alternatives
are not authorized published sign evidence.

## Continuous-gaze experiment with a moving camera

The global **Continuous sign** button and `F9` toggle a separate module beside
the active gaze solver. It is independent of stereo, detector and preview mode;
`outputs/settings/continuous-gaze-sign.json` saves/restores the selection. It
starts disabled when no setting exists, and malformed settings are reported.
Single-eye/virtual-contact and joint paths expose source-matched diagnostics.
The single-eye adapter reuses the existing motion-compensated contact hypotheses
and requires a fresh reliable image-motion interval for votes. A local 2D similarity
does not attest arbitrary 3D camera movement: its status explicitly says local
motion only. Missing motion and nearly circular projections remain unresolved.
The binocular path uses up to four retained
unrestricted hypotheses (up to sixteen when the selected mount policy already
requests them), with the optimizer/refinement budget unchanged. Candidate scope
explicitly says whether the input was restricted by a mount or orientation prior.
Use flexible mode for an unrestricted experiment; the module does not bypass an
explicit expected-mode pin in saved settings.

For inter-limbus unit baseline b and unit normals nL/nR it tracks
`[nL dot b, nR dot b, nL dot nR, b dot (nL cross nR)]`. These values are unchanged
by common rigid rotations and translations, so moving the camera alone adds no
trajectory penalty to exactly transported geometry. Actual inferred geometry can
still change with intrinsics error, reflections, scale error or missing contours.
The baseline is between limbus centers, not measured eyeball rotation centers;
its small gaze-dependent movement is another model limitation.

A bounded three-second source-time window compares whole-candidate histories,
using current image cost plus a soft, capped continuity preference. Sources must
span one second and include six unique binocular observations. Same-time revisions
replace votes, old exposures cannot rewind history, and loss or a change of prior
scope resets it. No persistent candidate index becomes a branch identity. Close
candidate directions are reported as coalesced; separated similarly supported
histories remain ambiguous. It exports scores, not probabilities, and never
substitutes a fit or copies another fit's posterior. It does not authorize gaze
calibration and has not been promoted as an independent sign detector.

**Camera-facing convergence is not the only unresolved case.** For example,
symmetric up/down binocular normal histories can have identical descriptors
while differing by more than 40 degrees. Smoothness cannot choose between them.
This is a concrete ambiguity of this descriptor, not a claim that every other
image cue is identical. Absolute motion about the inter-eye baseline is missing.
Independent face/head pose, stable anatomical pivot evidence, corneal cues or a
measured camera/display transform can supply additional information. A moving
camera also changes screen mapping even if the optical-axis sign is correct.

Related primary work fits a consistent eye-motion model across images:
[Swirski and Dodgson, 2013](https://www.cl.cam.ac.uk/research/rainbow/projects/eyemodelfit/).
That paper uses a head-mounted camera; its stationary eye-center assumption cannot
simply be carried over to a freely moving remote camera.
[Dierkes et al., 2019](https://www.hcics.simtech.uni-stuttgart.de/publications/dierkes19_etra/)
uses multiple gaze observations to constrain eye position and accounts for corneal
refraction. This experimental invariant module is not an implementation or
validation of either complete eye model.

### Bounded validation and the remaining restart

Two Rob-only recorded segments, first 18 source seconds each (366/364 cached
worker rows), were replayed with flexible geometry and the diagnostic off/on.
The native solver re-read RAW boundary evidence. Geometry, posterior and admission
outputs were exactly unchanged in all 730 rows. Across 365 distinct analyzed
exposures, 248 had a preferred conditional trajectory, 94 stayed ambiguous, 22
were collecting and one lacked binocular evidence. Preferences are not ground-truth
sign labels. There are no human localization labels or attested free-camera poses
in this comparison; it does not establish tracking accuracy under camera motion.

The latest clip triggers the existing no-repeated-acquisition assertion in both
arms after all rows are emitted; the earlier clip passes. At 11.471 seconds, the
UI had seen no confidence-qualified direction for three seconds. Its last accepted
sample was at 8.424 seconds. The selected normals had moved about 17.6/16.5 degrees,
but retained alternatives spanned 48.9 degrees and posterior 90% radii were about
34.5/30.4 degrees. This was an unconditioned replay, not the new BelowEyes phase
policy, and the UI did not test whether the old branch was continuously reachable.

A source-timed reachability probe connects retained candidate normals when their
angular separation is at most `assumed angular rate * elapsed source time + both
exposure error bounds`. These normals must be expressed in a common frame or the
rate must bound relative camera motion too. Under assumed 300 degrees/second and
3-degree per-exposure errors, there is one reachable candidate at the restart,
but three intermediate exposures allow competing paths. At 600 degrees/second,
multiple candidates remain reachable at the restart. Tighter assumptions can
instead leave no compatible path: they are not validated physical bounds.

A continuous-path argument needs a justified motion/measurement envelope and
coverage of the alternatives between samples. A posterior 90% radius is not a
hard bound, and finite optimizer alternatives are not all possible geometric
solutions. Direction precision insufficient for a calibration sample must not be
confused with proof that a previously established sign is lost. Calibration now preserves the stationary target during precision-only stalls and
requires explicit, fresh reference loss for automatic reorientation. This does
not promote the diagnostic trajectory score into sign truth or relax admission.

Rust comparison/reachability recipes, native reports, source-matched RAW/3D
overlays and UI renders are under `outputs/orientation-continuation-20260916`.
The earlier BelowEyes phase comparison had 17 gained and 9 lost ready sources
after five seconds on the latest clip; the earlier clip was unchanged. Matched
SN-FEIDA steps and median normal/area differences were effectively unchanged.
It assumes the initial screen fixation counterfactually and does not prove a
successful live calibration. Glare-related fits remain unresolved.

## Why the previous comparison needs this correction

The September 14 refiner comparison used a sequence replay whose contact
trackers always defaulted to flexible, while the live viewer was already using
below eyes. On the nearly stationary calibrated pair 1973/1974, the candidate
selected the opposite vertical tilt under that unconstrained history. That is
evidence of flexible-mode instability, not proof of the same regression with
the operating prior enabled. Re-evaluate baseline and candidate under the same
declared prior before deciding whether the refiner helps.

The first sustained disagreement begins at source 1771, after a blurred frame
at 1770 with no contour. Relative to 1769, the new fitted axes straddle the
roughly 90-degree nearest-direction association boundary: baseline and refined
angles differ by about one degree. Flexible tracking associates them with
opposite tilt branches while retaining a resolved sign. The subsequent
pupil/motion history carries that disagreement into the still pair. This is
an unresolved association weakness, not evidence that a one-degree local fit
change physically reversed the eye.

A fresh 637-exposure replay with below-eyes applied to both arms removes the
opposite-direction result on 1973/1974. All original image-derived contours,
pupil fits, RAW admission decisions and independent motion measurements are
unchanged by the mounting setting. Fresh signed coverage is 561 baseline and
568 candidate exposures, with zero published signed normals outside the
declared mounting hemisphere. The remaining mean absolute SN-FEIDA log-step
regression (0.03623 to 0.03693 over 555 matched transitions) is unaffected:
constraining sign does not improve contour accuracy by itself. RAW overlays,
3D rays and full comparison receipts are under
`outputs/camera-mount-checks-20260914`.

The failed live session `1789393388-022296519` already had below eyes selected.
Its four sign restarts and incomplete collection therefore cannot be dismissed
as merely forgetting this option. The prior does not repair all contour or
branch-continuity failures. Keep coverage, sign restarts, conditional target
error, human localization evidence and independently scaled SN-FEIDA alongside
each other when assessing any candidate; lower area variance alone is not success.

## September 14, 15:26 calibration failure and correction

The newer session `1789413998-067497827` recorded 227 complete paired exposures
over 20.187 seconds. Its configuration was **above eyes, stereo enabled, Butter
Obelisk, refiner off**. It timed out in sign acquisition with zero calibration
samples. Metadata shows only the moving acquisition stimulus; none of the nine
stationary calibration points was presented. This archive cannot prove an
accepted complete calibration because those observations do not exist.

The stereo mounting filter previously ran **after** integration. Only the first
unconstrained optimized mode carried a posterior; selecting another mode dropped
that posterior and left a zero alternative-cost margin. The native replay
reproduced the resulting above-eyes failure with both refiner settings. This was
a pipeline bug, not evidence that every recorded outer contour was unusable.

Mounting now belongs in the shared conic solve and sampled target density. The
native tracker and component evaluator use that same implementation. Its enum
is shared from `src/conic_solver/camera_mount.rs`; the monocular tracker reexports
it, so parsing, signs and operating defaults remain one definition.

Fresh paired SAM and Obelisk worker replays, refiner off and augmented, followed
by the actual bridge and acquisition gates all acquired under below-eyes at
349 ms of source time after the fix. These are completion-paced diagnostics,
not measured end-to-end live latency. The deliberately above-eyes Obelisk
regression also acquired after the fix, without missing posteriors; this proves
the confidence handoff works, not that above-eyes matches Rob's physical rig.

On the same recording the experimental refiner increased mean absolute
independently scaled SN-FEIDA steps for both eyes and both detectors. It remains
experimental and is not promoted. No human contour labels or independent gaze
truth exist for this clip. Source receipts, the four detector exports, matched
before/after bridge outputs, native RAW motion, numerical reports and neighboring
RAW/3D visual comparisons are in `outputs/calibration-recovery-20260914-1526`.

Reproduce the archive preparation and immutable comparison with the native
viewer commands:

```sh
data/target/live/buttercup-eye-viewer --offline-calibration-prepare SESSION.json NEW_OUTPUT_DIR
BUTTERCUP_STEREO_LIVE_REPLAY=combined data/target/live/buttercup-eye-viewer \
  --offline-stereo-sam-export SOURCES.jsonl NEW_MASKS.jsonl 0 454 student
data/target/live/buttercup-eye-viewer --offline-calibration-summary MANIFEST.json NEW_REPORT.json
```

`student` is the internal backend name for Butter Obelisk; `sam` selects SAM.
Refiner mode and model use the existing shared environment settings. The runtime
run scripts retain the exact library/model setup. Keep new calibration collection
on the actual mounting setting and assess a fresh complete nine-point run; never
invent missing target visits or relax coverage to make this short clip pass.

## Physical below-iris position is a different constraint

For a planar circular iris disk of center E, radius r, unit normal n, camera
center C and unit world-up u, the camera being below the **entire** iris means

```text
u · (E - C) > r sqrt(1 - (u · n)^2).
```

The right side is the disk's vertical half-extent, obtained by projecting u
into the disk plane. Camera-facing separately requires `n · (C - E) > 0`.
Neither condition generally chooses one of the two circular sections of a
perspective conic. The native recorded examples in
`outputs/calibration-direction-sheet-20260915/native-poses.json` satisfy both
for both candidates if world-up is provisionally identified with sensor-up.
Actual physical evaluation requires camera attitude relative to gravity.

The historical low-level `below-eyes` filter requires sensor-normal Y < -0.05.
That can select one candidate, but it is stronger than camera-below-iris position.
It assumes target/camera orientation and must not be presented as a measured
physical restriction. In particular, a positional setup declaration alone is
not evidence that its rejected direction is physically impossible.

## Agreed screen-orientation contract (2026-09-15)

The replacement for the continuous sensor-Y prior uses screen fixation only
while establishing direction. Routine tracking must allow off-screen gaze, retain
competing hypotheses, resist unsupported branch switches, and treat near-coincident
branches by their angular disagreement rather than by an arbitrary branch label.
Neither a temporal preference nor a remembered sign is new independent evidence.

Rob's setup bounds: lens between user and display, 1–6 inches below its bottom
edge, up to 24 inches in front; eyes 8–48 inches from the lens and never less than
3 inches above its center; sideways roll within 15 degrees, flexible pitch/yaw.
Orientation may ask for a brief top-center fixation; physical display dimensions
should come from credible selected-monitor EDID, with missing dimensions retained
as unknown rather than silently certifying the nominal fallback. These bounds are
not yet an experimentally validated bound on the display-horizontal direction in
camera coordinates. The old continuous sensor-Y filter and the new orientation-only mode are different
constraints; their provenance must remain distinguishable.

Calibration now excludes 2 seconds acclimation plus 100 ms display-onset allowance
from each stationary target. This conservative 2.1-second exclusion also applies
to clocked runs until actual scanout timing is attested. Thumbnails still disappear
by 500 ms, leaving an unobstructed fixation during acclimation. The minimum target
hold is 4.6 seconds, preserving 2.5 seconds after exclusion. Original RAW arrival,
source-time settling and duplicate-source guards remain mandatory.

Distraction is usable gaze directed elsewhere, not absent, unsigned or otherwise
invalid signal. The collector groups usable observations by angular coherence and
compares the winning group with the strongest separate fixation, instead of pooling
all other looks into a single competing group or imposing a fixed 60% sample quota.
Require six unique usable sources and a lead of at least max(2, ceil(winner/4)).
Nearly tied coherent fixations wait for more evidence. RMS/localization gates still
apply. Tests cover 60/40 fixation/distraction, repeated returns among two brief
looks, competing cohesive groups, invalid features, and delayed/duplicate sources.

This sample-based comparison is not a proof about wall-clock percentages or a
classifier that knows which cluster corresponds to the displayed target. Burst-rate
bias, duration-aware fixation evidence, source-matched RAW validation with recorded
distractions, and a justified projected-axis bound remain required before claiming
the complete operating contract is implemented. Native orientation/continuation
and its UI description now use the phase policy described above.
