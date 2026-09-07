# Calibration sign-acquisition regression: matched offline trials

The initial offline trial did not justify any of the three changes as a
standalone live fix. Its results are preserved below. The subsequent
[live-retest follow-up](#live-retest-follow-up) adds an explicit recorded
acquisition phase and enables the sustained-motion fallback as part of that
workflow, without relaxing the sign or geometry support margins.

The later [meridian-continuity follow-up](meridian-sign-continuity.md) removes
the user-requested post-calibration epoch interlock and documents the subsequent
video review and source-time crossing regression tests. Earlier descriptions
of that interlock below record the policy at that stage, not the current policy.

The final reports are under
`outputs/sign-acquisition-20260907.mhqqmo/final/`. Reports in its parent are
development runs. The source baseline was `db05f4f`.

## What actually failed

The latest failed calibration, `sam31-mouse-3d-1788782054-740335557`, recorded
72 right-eye RAW frames and 30 distinct serialized right-eye surface sources.
None of those right-eye samples was signed. The first calibration point was
therefore never admitted. The left eye had signed samples, but calibration
was using the right eye. Recording itself worked.

The inspected native RAW frame contains the iris and a bright rectangular
reflection crossing the pupil. No pupil-void fit is serialized in that latest
attempt. The reflection is a plausible reason for loss of an independent
inner-boundary cue, not a proved causal diagnosis from one inspected image.

Including the two preceding attempts reproduces the right-eye failure in the
established tracker: 0 signed results from 56 distinct attempted surface sources.
There are 139 reliable adjacent RAW-motion estimates among 147 right-eye frames,
and 52 of the 56 attempts have a complete reliable source-to-source transport.
55/56 frozen input ellipses meet the source/current-ROI containment margin.
This is predominantly a sign-acquisition failure, not an eye-out-of-frame or
missing-RAW-motion failure. There is one recorded pupil anchor before the
reacquisition, not a sustained usable anchor sequence after it.

For the established motion-only bootstrap, the largest ratio of the branch
EMA residual gap to its required margin is **0.8655**; it must reach one. The
margin is `max(0.35 px, 4 * transport residual px)`. The multi-frame motion
window also fails to provide enough discriminating support. Simply having
motion estimates does not establish which of two convex normals is correct.

This identifies the failing state/evidence interaction, **not a bad commit**.
The prior successful live run already used the motion-sign window. In the
failed attempt the focus-following desktop feature made no focus changes.
Session history, pupil visibility, and captured motion differ; these are not
controlled before/after images from a single software revision change.

## Trial design and actual subset

All policies see exactly the same frozen fitted ellipse and pupil anchor from
each recorded source. Native RAW supplies candidate-independent global motion
and relative image scale. Neither labels nor the recorded sign choose a branch
in the cold trials. No SAM inference, segmentation refit, or artificial ROI
movement is performed.

| Subset | Right RAW frames | Left RAW frames | Distinct attempted conics, right / left |
| --- | ---: | ---: | ---: |
| Three failed attempts, complete recordings | 147 | 136 | 56 / 57 |
| Previous successful calibration, complete recording | 140 | 140 | 95 / 0 |
| Older motion capture, first 1,000 RAW frames **per eye** | 1,000 | 1,000 | 335 / 416 |
| Total | 1,287 | 1,276 | 486 / 473 |

These are 2,563 ROI frames, not 2,563 independent sensor exposures or people.
The successful clip's left ROI is recorded but publishes no contact surface;
it is a measured missing-evidence case, not an omitted successful solve.

The failed archives under `outputs/calibration-corpus/` are:

- `sam31-mouse-3d-1788782043-267030473.tar`
- `sam31-mouse-3d-1788782045-904584446.tar`
- `sam31-mouse-3d-1788782054-740335557.tar`

The successful archive is
`sam31-mouse-3d-1788779879-782299279.tar`. The older source is
`outputs/raw-eye-hotkey/both-eyes-1788766324-951655107.tar`, extracted as
`outputs/sign-audit-20260907/capture`. Its selected right-eye sequence range is
3532–4555, sensor timestamps 1690047087484494353–1690047192224563866;
left is 3532–4882, timestamps 1690047087484494353–1690047225347112203.
The ranges contain gaps. They span approximately 104.74 and 137.86 seconds,
respectively, so they are not a matched binocular time interval. Each arm is
matched within the same eye; there is no binocular sign transfer in this test.

The policies are:

1. `Established`: the unchanged historical baseline behavior.
2. `MotionWindowFallback`: allow the existing multi-frame motion decision to
   acquire an initial sign, as well as correct an already signed surface.
3. `MotionWindowOnly`: require that window for motion-only acquisition, keeping
   the existing pupil-anchor path.
4. `ReliableMotionSeed`: initialize the residual EMA from the first actually
   supported motion interval, rather than letting preceding unsupported contour
   observations make that first residual receive only 25% weight.

All retain convex/camera-facing geometry checks, duplicate/source ordering,
area-family handling, and the existing sign-change safeguards. No gate is
disabled merely to increase coverage.

## Results: availability is not sign accuracy

Entries below are signed fresh sources / distinct attempted sources, with
cold state at each genuine source discontinuity.

| Subset / eye | Established | Window fallback | Window only | Reliable-motion EMA seed |
| --- | ---: | ---: | ---: | ---: |
| Failed lead-in + final attempt, right | 0/56 | 0/56 | 0/56 | 0/56 |
| Failed attempts, left | 39/57 | 39/57 | 0/57 | 39/57 |
| Successful clip, right | 73/95 | 73/95 | 0/95 | 73/95 |
| Older motion, right | 320/335 | 320/335 | 177/335 | 330/335 |
| Older motion, left | 372/416 | 372/416 | 0/416 | 381/416 |

The fallback arm is a no-op on this subset. The window-only arm introduces
major availability regressions. EMA initialization adds ten and nine signed
sources in the historical right/left subsets but does not fix the failed
calibration. On historical left, its agreement with the recorded branch drops
from 296 to 285 despite greater availability; epoch changes go from five to
four. Without independently labeled gaze signs, neither change is an accuracy
win or loss. It is sufficient evidence **not** to promote it based on coverage.

Signed-and-contained counts in those same rows are respectively:
`0/0/0/0`, `37/37/0/37`, `73/73/0/73`, `233/233/116/237`, and
`236/236/0/245`. This statistic checks the source and presentation ROI margin;
it is **not** a replay of the entire calibration state machine or a prediction
that calibration would finish.

### History and timing controls

- Replaying only the final failed right-eye clip from cold state gives **27/28
  signed** sources, versus **0/56** with the recorded lead-in. That apparent
  success is a replay-start artifact, not a fix. Those 27 signed normals are
  opposite the live clip's provisional unsigned normals; neither is ground
  truth.
- Delivering the failed combined sequence at idealized source time also gives
  **0/56**. Publication delay alone does not explain this frozen-input result.
  It does not test the effect of a different SAM inference cadence or recover
  proposals that were never recorded.
- Cold replay of the successful clip gives **73/95** signed, all opposite its
  recorded signed branch. Seeding the first usable sample with the recorded
  branch and resolved flag gives **95/95**, with zero epoch changes and all 95
  agreeing with the recording. This is a *checkpoint sensitivity control*, not
  permission to assert that sign without evidence. It does not reconstruct all
  hidden pre-recording state.
- The corresponding seeded failed run has one signed but ROI-clipped source
  before reacquisition and zero eligible signed sources. An old sign is not
  carried through the new acquisition.
- Removing independent motion gives zero signed sources for every arm on the
  combined failed right-eye sequence and historical left-eye sequence. Existing
  recorded anchor fragments alone do not complete acquisition there.

Recent archives have exact integer source timestamps, source-epoch identity,
ROI origins and host-ready clocks. The replay delivers a surface on the RAW
frame that published it, not retroactively on its earlier exposure. It merges
the three recordings chronologically, rejects unarchived/stale/future sources,
uses production ROI continuity, and never counts held results as fresh votes.
Duplicate keys, if encountered, require identical geometry and RAW bytes.
There were no overlapping RAW exposure keys in these selected archives.

The failed right-eye median source-to-publication-frame lag is 406.4 ms
(range 304.4–708.7); successful right is 303.7 ms (201.1–508.2). These are
source-clock frame lags, not isolated GPU inference times. Older records lack
host-ready stamps: 779 right and 743 left replayed publication rows use source
cadence as a fallback. They have region-session identity and host arrival
ordering, but no new-style stream epoch. Exact internal SAM consumption times,
rejected/unpublished fits, and pre-recording tracker state remain unavailable.

## SN-FEIDA, localization, and uncertainty

Use [scale-normalized frontal-equivalent iris disk area (SN-FEIDA)](flat-tire-area-and-motion.md)
as the outer-limbus diagnostic. This trial changes sign selection, not the input
ellipse. Maximum output frontal-equivalent area difference from the fixed input
is approximately `1.46e-11 px²`, numerical roundoff. Area invariance is expected
and **cannot distinguish correct and incorrect signs**.

Independent relative scale is available for 714/959 distinct input conics,
yielding 412 successive same-reference comparisons no more than 500 ms apart.
The RAW transform's scale is `hypot(1+d, b)`, never the candidate's iris radius.
Chains reset on missing/unreliable motion, source discontinuity, after one
second, or after accumulating a heuristic fractional allowance above 0.25.
Each valid step adds `0.01 + transport_residual_px/100` to that allowance.
It is an engineering support allowance, not a calibrated confidence interval.

| Subset / eye | Scale-supported inputs | SN-FEIDA pairs | Mean absolute log area step |
| --- | ---: | ---: | ---: |
| Failed attempts, right | 49 | 34 | 0.01931 |
| Failed attempts, left | 46 | 30 | 0.02819 |
| Successful clip, right | 84 | 67 | 0.02769 |
| Older motion, right | 258 | 149 | 0.09042 |
| Older motion, left | 277 | 132 | 0.08805 |

These characterize the **fixed input conics**, not improved candidate fitting.
References are never pooled as absolute anatomical areas. Missing input
surfaces and missing scale are not backfilled. No human gaze-sign or limbus
labels, independently measured metric scale, or anatomical ground truth were
available for this subset. No new human-label localization score is claimed;
changing signs does not improve localization of the frozen ellipse. RAW motion
is independent of the tested radius, not independent of every iris pixel.

## Proposed next fix after the initial trial

The evidence favors addressing **initial sign observability and initialization**,
not accepting unsigned calibration points or repeatedly resetting until a noisy
interval happens to pick a sign. Preserve supported state across ordinary
calibration entry and compatible ROI nudges, but not genuine reacquisitions.
When the sign really is unknown and the pupil cue is occluded, a short explicit
small moving-target acquisition phase is the next candidate: collect enough
source-aligned, branch-discriminating motion before the stationary calibration
dwell begins. Keep both hypotheses and show why acquisition is waiting.

At the end of the initial trial, this was a proposed intervention, not an
implemented or validated fix. The follow-up implementation and live results
are documented below.
Stationary-target video cannot prove how the user will respond to a new moving
target. Test that interaction with a new capture and independent target/sign
evidence; do not use a saved monitor affine or the recorded sign as its own
ground truth. No precise source commit causing this session regression has
been isolated.

## Reproduce and verification

Build the viewer with the repository's usual native library environment:

```sh
cargo build --offline --profile live --features sam31 --bin buttercup-eye-viewer
```

The offline command uses the same LibTorch/NVIDIA shared-library paths as the
viewer launcher, but does not load models, run inference, connect to the camera,
or open a window. Each extracted directory must contain `frames.jsonl`,
`predictions.jsonl`, and the requested eye's `.raw10` stream. Example:

```sh
data/target/live/buttercup-eye-viewer --offline-sign-acquisition-trial \
  outputs/sign-trial-new.json subject-right --arrival --cold \
  outputs/sign-acquisition-20260907.mhqqmo/1788782043-267030473 \
  outputs/sign-acquisition-20260907.mhqqmo/1788782045-904584446 \
  outputs/sign-acquisition-20260907.mhqqmo/1788782054-740335557
```

All four arms run together with shared fixed inputs. `--source`,
`--seed-recorded`, `--no-motion`, and `--limit N` select diagnostic controls.
Output uses `create_new`; choose a fresh filename for each run. The offline
report is ordinary JSON and makes no change to the live recording protocol.

Added tests cover cold acquisition in both directions, equivalent ellipse-axis
parameterizations, an ROI nudge, delivery delay, duplicate/late frames, static
tilt, frontal geometry, missing transport, reliable-only EMA initialization,
lossless clock identity, and independent scale-chain gaps. The targeted sign
suite passed 21 tests. The complete SAM-enabled viewer suite reported
**909 passed, 41 failed, 24 ignored**; the 41 failing test names exactly match
the pre-existing `gaze-focus-complete-tests.log` failures. This is not a claim
of a green full suite. The optimized build and source-tree audit passed.

Independent replay jobs ran in parallel under shared CPU/memory-bandwidth
claims; their logs retain coordination tokens. Microsecond tracker timings are
diagnostic only, with fixed arm ordering and shared resources; they are not an
isolated performance benchmark. That initial trial changed no camera/firmware,
live default, calibration gate, saved monitor, label, or desktop-focus setting.

## Live-retest follow-up

The workflow bug is a circular dependency: the first calibration target asks
for a stationary fixation even when acquiring the sign needs informative eye
motion. The new `calibration_acquisition` module presents a smooth figure-eight
only when an otherwise usable iris surface has an unresolved sign. It remains
inside normalized coordinates 0.405–0.595 on both axes: the requested middle
20% of the screen. An already signed, usable eye keeps the existing fast
stationary calibration path.

Acquisition begins only after automatic paired RAW recording is confirmed.
It requires four distinct, increasing source timestamps in one source
generation and sign epoch, spanning at least 300 ms, with no gap above 750 ms.
There must also be sustained sign support: the existing multi-frame motion
window or four pupil-anchor confirmations. The live tracker now permits its
motion window to acquire an unresolved sign, not just correct an established
one. The window's geometric and uncertainty margins are unchanged. A later
sequence of four matching pupil anchors can validate a motion-only seed
without changing its sign or epoch; duplicates cannot train this confirmation.

Target positions do **not** choose a gaze sign. They are a stimulus for the
user; the solver still consumes only conic/RAW-motion/pupil evidence. An
isolated motion-EMA sign selection is insufficient to finish this new phase.
Motion sources are never put into stationary calibration clusters, including
the source that completes acquisition. The first stationary target starts a
new wall/source-time settling interval, so delayed in-flight SAM results from
the moving phase cannot contaminate it.

The phase has a 20-second bound within the existing recording/session bounds.
Failure remains explicit, stops the automatic capture, and does not fabricate
a sign, save a monitor, or restart forever. The small live thumbnail follows
the stimulus; explanatory text still waits three seconds. Unsafe clipping and
source mismatch are checked before labeling an unsigned surface as merely
needing direction acquisition. Completed monitor mappings remain preserved.

The existing recording streams now retain:

- The exact submitted moving target, with role `sign-acquisition`, followed by
  its removal and the distinct stationary calibration target. Rendering and
  recording share one target/animation definition and one presentation time.
- Acquisition phase, episode count, duration, fresh-source count, and whether
  sustained support has arrived, in scene metadata and the session sidecar.
- Immutable per-surface `sign_diagnostics`: support kind, selected branch,
  both pixel residual EMAs, source-motion residual, temporal margin, and anchor
  confirmation count. Held surfaces retain their original source evidence.

These are small coordinate/diagnostic additions, not another video stream or
gzip layer. Native ROI/global data and the existing OIM1 framing are unchanged.

The follow-up outputs are `outputs/calibration-acquisition-live.25p0O4/`.
A matched repeat on the same 2,563 RAW ROI frames reproduces the initial
baseline/fallback coverage and area results; the stationary failed recording
still cannot validate a new human moving-target response. A synthetic end-to-end
test drives the actual stimulus through known convex eye geometry, realistic
small angular excursions, 1.5-pixel transport residual, 250-ms conic cadence,
400-ms publication delay, and both signs. It acquires within eight seconds,
preserves frontal-equivalent disk area, and rejects late moving-phase samples
from target one. This is a model test, not measured human gaze accuracy.

Additional tests cover missing recording startup, recorded/drawn target
agreement, timeout, duplicate/late frames, source/sign changes, preservation of
an already usable or completed calibration, and later pupil corroboration.
The complete suite reports **916 passed, the same 41 pre-existing failures,
24 ignored**. There are no new failing names. The optimized build and source
audit pass. Human gaze-sign labels, calibrated uncertainty, and proof that the
new stimulus resolves the user's real reflection/occlusion case remain missing.

For the live retest: start SAM31 with the prior camera focus (530), press **M**,
and follow the moving plus if direction acquisition is needed. Once it becomes
stationary, hold each target as before. No separate S press is necessary: the
entire acquisition/calibration interval is automatically recorded for diagnosis.
If acquisition times out, retain that capture rather than repeatedly forcing
resets until one branch happens to be selected.

### User-led live results

The rebuilt viewer was relaunched after the desktop session restarted. Camera
focus stayed at 530 and settled; desktop pointer injection and focus-following
remained off. The user started these attempts without an agent-issued
calibration command. The second ROI's analysis remained disabled at launch;
paired native ROI capture is distinct from enabling a second eye's analysis.

| Capture suffix (`sam31-mouse-3d-…`) | Recorded duration | Result |
| --- | ---: | --- |
| `1788786652-056177751` | 38.876 s | Acquired direction, all nine targets and both fits accepted |
| `1788786721-601704506` | 11.457 s | Ended before completion, no timeout/error; target one accepted, target two had no stable cluster |
| `1788786821-286447565` | 21.291 s | Already signed; all nine targets and both fits accepted |

The first attempt spent 11.751 seconds in acquisition. The recorded handoff
shows **pupil-anchor** support, not a sustained-motion-window decision: a
previous motion-only seed was corroborated by later pupil evidence. Therefore
this demonstrates the new workflow and handoff working live, **not** that the
moving stimulus alone fixed the failure or that the motion fallback solved this
reflection case. The 705 submitted moving-target positions remained inside
the middle 20%. All nine subsequent stationary targets used sign epoch one;
89 distinct predicted sources were recorded after handoff, with no sign or
authority restarts during stationary collection. This count is not the number
of samples admitted to the nine final clusters.

Its 2D/3D fit RMS values were 0.0407/0.0404 normalized screen fraction; the
latest completed attempt's were 0.0447/0.0426. These are training-target fit
residuals, not independently measured gaze accuracy. Neither proves physical
monitor pose or correct gaze signs. Both completed archives pass
`scripts/validate-recording.py`: exact source references, finalized target
removals, paired RAW, native thumbnails, and no metadata-sequence gaps. The
first contains 739 ROI frames and the later one 362; these are not independent
sensor-exposure counts. Reports and the read-only handoff audit are under
`outputs/calibration-acquisition-live.25p0O4/`.

There is a remaining failure after **both** successful calibrations: the live
log reports a sign-epoch change and suspends the cursor while preserving the
completed mapping. These transitions happened after their automatic RAW
captures closed, so the archives do not contain the exact transitions' raw
evidence. They cannot be classified as true branch reversals versus
scale-family/reset transitions from those log messages alone. The safety gate
remained intact at that handoff; the saved monitor defaults were not overwritten.
Post-calibration sign stability and independent accuracy still require
evaluation, even though target acquisition and calibration completion now
succeeded in these live attempts. At that handoff the latest mapping was preserved
but its cursor was suspended, not a confirmed working live gaze cursor.
