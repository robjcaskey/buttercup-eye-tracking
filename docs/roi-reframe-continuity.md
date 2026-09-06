# Motion through in-sensor ROI reframes

Checkpoint before this work: `fa72f7a` (September 5, 2026).

## TODO: preserve continuity across ROI moves

The parallel implementation pass is in the worktree. Checked items identify implemented
and tested work, including explicitly opt-in prototypes; they do **not** mean
all four issues are solved in the default live model. The caller reset fix is
enabled. Crop-aware learned memory and stable photometry remain off because
matched corpus evaluation found regressions. The results and remaining gaps are
recorded below.
An ROI translation changes the coordinate window; it must not, by itself,
change eye identity, source time, physical scale, or the sign of the contact.
Some boundary evidence really does disappear, so the requirement is bounded,
evidence-aware continuity rather than identical predictions from different
exposures or invented observations of missing pixels.

### 1. P0: remove the live viewer's unnecessary session reset

- [x] Add a regression through the live caller and worker before changing the
  reset policy. Previously in `src/main.rs`, `source_discontinuity` treated any
  sensor-origin change as a new SAM session, clearing both eyes' gaze/contact
  trackers, RAW histories, proposals and results. The epoch change also defeats
  the worker's intended pupil-history preservation.
- [x] Share an explicit discontinuity policy between live viewing and replay:
  distinguish a compatible crop translation from identity/prompt changes,
  incompatible geometry, out-of-order sources and genuine source-time gaps.
  Handle duplicate presentations without counting them as fresh evidence.
- [x] Preserve sensor-referenced gaze/contact state on compatible moves,
  including pivot estimates, sign hypotheses, kinematic history, area history
  and pupil size/offset evidence. Audit downstream gaze-authority and calibration
  invalidation so a crop-only move cannot indirectly restart them.
- [x] Keep continuity decisions per eye. Moving one ROI must not clear an
  otherwise unaffected eye's state; shared exposure clocks must remain shared.
- [x] Handle asynchronous pre-move results using their original sensor geometry
  and source time, with existing age/identity checks and supported motion
  transport. Do not reject them solely because the ROI moved, reinterpret
  crop-local coordinates as current coordinates, or count them as new frames.
- [x] Acceptance: a compatible X/Y nudge does not change the tracking/sign
  session or clear the other eye; stale, wrong-identity and truly discontinuous
  results still fail admission. Exercise moves while SAM work is in flight.

### 2. P1: preserve useful SAM context without misregistering spatial memory

- [x] Inventory `LiveTrackerState` and `PriorFrameFeatures`: separate eye
  identity, query anchors, object pointers and pupil history from image features,
  mask memory and positional information. Record each item's coordinate space,
  source clock, validity and reset reason; do not assume all learned state has
  the same spatial dependencies.
- [x] Implement compatible non-spatial history preservation (pupil history in
  defaults; identity/pointer preservation in the opt-in experiments) with current association
  checks, immutable source ages and bounded miss budgets. Repeated nudges must
  not refresh old evidence or indefinitely lock onto an incorrect identity.
- [x] Prototype a bounded crop-aware memory path using sensor-addressed
  historical attention keys. Historical RAW re-encoding remains a future comparison.
  Account for positions, resampling, padding and valid overlap explicitly;
  blindly retaining or shifting feature tensors is not sufficient.
- [x] Keep a safe current-frame recovery path when memory cannot be transported
  reliably. Require fresh RAW support for newly admitted fits, and measure
  whether memory helps select the correct limbus rather than merely hold a fit.
- [ ] Acceptance: compare cold-reset, identity-only and crop-aware memory on
  matched sequences, reporting localization, coverage, sign continuity and
  latency/memory cost. Keep the existing identity experiment disabled by default
  until its low-light regressions are resolved and the live path is covered.
- [x] Add opt-in current-detector competitor arbitration before committing a
  propagated mask that conflicts with independently source-aligned motion/scale
  support. RAW ring support alone admitted large normalized-area outliers in
  this experiment; do not fix those by relaxing the contact jump gate.
- [ ] Make the prior's own fit uncertainty defeasible during arbitration. The
  September 6 bounded prototype rejects useful corrections as well as outliers;
  it must not turn a merely RAW-admitted previous ellipse into anatomical truth.

### 3. P1: stop crop boundaries from unnecessarily changing retained pixels

- [x] Instrument crop-to-crop white-balance gains, percentile bounds, clipping
  and processed values on common sensor pixels. Separate photometric changes
  from resizing phase, feature-grid position and boundary-context effects.
- [x] Implement and test bounded, robust running normalization driven by
  trustworthy/common RAW support instead of independently normalizing every
  crop. Preserve adaptation to genuine illumination changes without letting an
  entering glint, lid or background abruptly remap the whole iris.
- [x] Add sensor-anchored demosaicing and a bounded, border-excluding sensor
  sample lattice; test all CFA phases and same-source shared-pixel preservation.
- [x] Retain rejected lighting increments against the last accepted RAW
  reference, preserve unapplied rate-limited adaptation, and expire unsupported
  lighting references by source time. Compare actual legacy-to-legacy transforms
  in the diagnostic, not previous running-to-current legacy parameters.
- [ ] Separate anatomical motion from illumination changes in the lighting
  estimate. The v2 literal sensor-coordinate comparison still mostly abstains
  on the moving-eye sequence and periodically reinitializes.
- [ ] Resolve final model-grid phase/context effects. The prototype's final
  resized tensor remains crop-local; it is not an invariant learned embedding.
- [ ] Acceptance: derive shifted crops from the same RAW exposure, inject
  bright/dark boundary content, and measure overlapping-pixel changes plus
  fitted geometry changes. Test bright, dim, glare and real lighting-transition
  sequences; choose tolerances from measured baselines, not a brightness-only
  acceptance gate.

### 4. P1: handle genuine loss of visible context proportionately

- [x] Compute overlap and visibility in sensor coordinates for both the crop
  and independently validated eye support. A small outgoing border is not
  equivalent to losing the iris. In the demonstrated `(+32,+24)` nudge, crop
  overlap was 84.5% and the previous validated foreground remained 100% visible.
- [x] Represent newly exposed pixels as having no temporal history and departed
  pixels as historical-only. Preserve original timestamps; transported masks,
  held fits and predicted arcs must never become fresh observations.
- [ ] Weight current arc, motion and scale support by actual visibility,
  age, focus and spatial coverage. Keep broad-support requirements: a precise
  fit to a thin overlap must not claim whole-eye motion or calibrated certainty.
- [x] Define graded handling for full-eye overlap, partial clipping, severe
  clipping and no overlap: retain supported state, widen uncertainty or abstain,
  and reacquire only when association/support is genuinely insufficient.
- [ ] Wire graded focus/coverage uncertainty through all relevant arc/scale
  consumers. The shared assessment API is tested, but the live consumers in this
  pass are overlap-only motion sampling and SAM foreground association/traces.
  Do not substitute a per-eye tracking epoch for the actual shared sensor clock.
- [ ] Acceptance: small full-eye nudges retain continuity; large jumps and
  disappearing eyes do not fabricate certainty. Show recovery time and reasons
  for reduced support separately from global reacquisition and identity changes.

### End-to-end acceptance and rollout

- [x] Add the same-source crop-equivariance test as separate, identically warmed
  branches through shared live session/result-admission helpers, the production
  worker and contact consumption. This is a blocking offline harness, not a
  full asynchronous UI/AF/calibration simulation. Do not feed alternate crops
  of one exposure as successive new frames.
  Also test real successive exposures with simultaneous eye/head motion.
- [ ] Cover positive/negative X/Y moves, repeated nudges, delayed SAM results,
  one-eye moves, paired ROIs on one exposure, source gaps, clipping and missing
  eyes. Log reset reasons, retained state and source-relative latency.
- [x] Run matched baseline/candidate corpus evaluations alongside synthetic
  tests, in parallel where practical. Include the recorded small nudge, large
  jumps, fixed/cycling insets and brighter/low-light labeled subsets; explain
  every regression instead of relying on aggregate admission counts.
- [x] Report human-label localization, coverage/dropouts, independent RAW motion
  and scale support, sign changes, recovery latency and
  [SN-FEIDA](flat-tire-area-and-motion.md#canonical-diagnostic-sn-feida).
  Never use the candidate radius as its own scale reference, bridge unsupported
  observations, or reward a frozen but wrong ellipse. State missing labels,
  timing/scale limitations and heuristic uncertainty explicitly.
- [ ] Check real-time resource budgets, existing admission/safety regressions
  and a live-viewer smoke test before changing defaults. Enable changes
  incrementally with diagnostics that identify which layer lost continuity.

## Current integration and evaluation: September 6

Runtime artifacts from the initial team pass are under
`outputs/roi-reframe-team-20260905-5TRzMn/`; resumed evaluations are under
`outputs/roi-reframe-20260906-AfTUWE/`. Coding agents for the resumed pass used
Astra with medium reasoning. No capture, model or compiled artifact is checked
into the source repository.

### Enabled continuity and clock contracts

`roi_continuity` owns per-eye source admission and tracking epochs. Overlapping,
same-sized crop translations retain the live session, contact/sign/pivot state,
RAW history and pending-result eligibility. Actual discontinuities still clear
the affected eye. Duplicate/reordered sources are ignored without clearing the
session, renewing evidence age, or counting as fresh measurements.

Result presentation maps the result's original crop into sensor coordinates and
then into the destination crop; moving the viewport cannot make a stale result
fresh. Calibration requires supported, unclipped limbus evidence in **both** the
original and destination crops. The shared tests exercise populated host state,
in-flight results, one-eye moves and invalid source/presentation cases.

Independent RAW motion is captured as a bounded, value-owned timeline snapshot
when a SAM source is submitted, never when its answer arrives. Live snapshots
use the actual shared camera-connection clock epoch, separately from per-eye
tracking epochs. A snapshot cannot acquire later motion even if the live
64-entry timeline advances, evicts entries or resets.

Replay declares one clock domain per recorded `lineage` (or capture if absent),
not per ROI, crop, file member or tracking epoch. A changed domain invalidates
the entire replay tracking session, including SAM/contact state. This is a
conservative archive boundary, **not recovered hardware boot provenance**.
Original source timestamps remain unchanged. The labeled index has three
declared lineages; its 225-to-226 file-member change does not change the clock.

### Experimental policies: all remain OFF

| Flag | Bounded behavior | Important limitation |
| --- | --- | --- |
| `BUTTERCUP_SAM31_CROP_MEMORY=1` | Sensor-addressed overlap keys, rebased sine/axial positions, aged RAW-validated mask memories and separate object pointers | Historical features retain old crop context; they are not RAW re-encoded or exactly crop-equivariant |
| `BUTTERCUP_SAM31_STABLE_PHOTOMETRY=1` | Sensor-anchored RAW reconstruction/statistics and source-time-limited running normalization | Final resized model grid is crop-local; corresponding sensor pixels need not depict the same anatomy |
| `BUTTERCUP_SAM31_MEMORY_ARBITRATION=1` | Independent RAW motion triggers inspection of up to four current, identity-associated detector competitors before memory commit | Prior ellipse uncertainty is incomplete; hard conflict bounds can reject genuine corrections |

Crop memory uses the existing maximum 16 histories, at most six recent memory
frames plus conditioning in attention and 16 pointers. Source age is at most
900 ms. A moved memory needs at least 50% valid token support after the border
guard and at least 80% visibility of the prior RAW-validated foreground. Missing
or newly exposed cells are not padded into evidence. Fresh RAW gates and the
three-processed-miss release remain in force. Attention/decode errors fall back
to current-frame detection; compatible moves cannot keep old identity alive
indefinitely.

Photometry v2 keeps separate latest-observed and accepted-lighting RAW snapshots
(at most 4096 samples each). Rejected changes remain recoverable against the
accepted reference; source-time rate limits still apply. Unsupported accepted
references expire after 900 ms or loss of overlap. Duplicates do not refresh
either clock. Default and trace-only adapter bytes remain identical to legacy.
Mild-blur and balanced-quad-RGB are supported; other adapters explicitly use
legacy behavior. Linear white-balance gains largely cancel against per-channel
percentile normalization here: percentile range changes, sampling and crop
context matter more than the gain numbers alone.

Arbitration requires an exact, gap-free RAW-motion chain between the prior and
current SAM sources, matching eye/session/clock, at most 500 ms, eight links,
and bounded accumulated residual. Unknown motion remains unknown, never unit
scale. Its area and center allowances are engineering heuristics, not calibrated
intervals. Only a fresh RAW-admitted memory commit renews its reference. Tiny
incremental drift and a consistently wrong prior can still escape it.

### Matched findings and rollout decision

Both passes use `mild-blur`, the same frozen SAM bundles and untouched RAW10.
The small nudge has 35 right-eye exposures (918–952), 420x280 pixels, with a
`(+32,+24)` move at 927. The labeled cycling-inset test has 37 unique exposures
and 14 canonical targets, using 360x240 pixels from native 384x256 recordings.
It contains 10 brighter `example2` frames, 24 low-light frames, and three recent
regression frames. Labels are read only after inference completes.

| September 6 policy | Nudge RAW admitted / fresh contacts / signed fresh | Labeled-cycle RAW admitted | Labeled targets admitted |
| --- | --- | --- | --- |
| Default | 35 / 35 / 34 | 35/37 | 13/14 |
| Photometry v2 | 35 / 35 / 34 | 34/37 | 13/14 |
| Crop memory + arbitration | 32 / 32 / 31 | 28/37 | 9/14 |
| Both experiments + arbitration | 29 / 29 / 28 | 28/37 | 9/14 |

All default outer/pupil fits and RAW admissions exactly reproduce the prior
default on all 72 sources. A separate old-caller-reset ablation of the nudge
produced 32 signed fresh contacts versus the new caller's 34, with one needless
epoch change versus none. That isolates a host-state continuity gain, not an
improvement in limbus localization or proof of live UI timing.

The initial crop-memory prototype admitted all 35 nudge frames but lost four
fresh contacts: normalized-area jumps at 932, 945, 948 and 952 tripped the
existing contact gate. Mean absolute SN-FEIDA log step worsened from 0.1279 to
0.1534 on the same 34 independently supported transitions. The initial
photometry prototype lost one fresh contact and had stale lighting bounds:
rejected increments disappeared because only the last observed RAW snapshot
was retained.

Photometry v2 fixes that state-accounting error, restores the missing nudge
contact, and reduces maximum green-channel high clipping from 5.04% to 2.11%.
At frame 933, clipping falls from 4.72% to 0.73%. But only one of 34 transitions
accepts an illumination update; three reference expirations supply much of the
recovery. On 31 comparable sensor-sample transitions, running mean absolute
normalized RGB change is 0.05365 versus actual per-crop 0.05055. It is therefore
not yet a better motion/illumination separator. The v1 per-crop MAD diagnostic
used the wrong previous transform and must not be treated as a matched baseline.

Against default, v2 photometry's nudge mean absolute SN-FEIDA step improves
0.1279 to 0.1164 on all 34 pairs, but its maximum worsens 0.3595 to 0.4570.
In the labeled set it loses 10125 (no current mask) and 10193 (RAW ring score
2.309 below the unchanged 2.450 gate), and gains a **bad** 10124 fit at 40.39 px
visible-label RMS. The 12 jointly admitted labels improve in mean 9.064 to
8.753 px, yet seven worsen and five improve. The aggregate win is dominated by
10181 improving 12.00 to 4.89 px; the five brighter labels are essentially
unchanged in mean, 3.967 to 3.971 px. Neither lower clipping nor a slightly better
aggregate error justifies enabling this path.

Arbitration reduces nudge area jumps on the 30 jointly supported transitions
(mean absolute SN-FEIDA step 0.1161 to 0.0420), but loses current observations at
932, 933 and 952 when all eligible current alternatives conflict with the prior.
On top of crop-memory's existing low-light losses, it drops labeled frames
10181, 10206 and 10217. These have full prior foreground visibility and fresh
RAW-admitted detector candidates: rejection is caused by the new prior-conflict
gate, not the eye leaving the crop. The nine jointly admitted labels have lower
mean RMS (7.494 to 5.899 px) but worse median (4.788 to 6.898 px); two improve,
five worsen, two are identical. The dominant improvement at 10193 (30.54 to
8.23 px) already existed in the crop-memory-only run. The brighter five-label
mean worsens 3.967 to 4.661 px, and the recent target 195 worsens 3.20 to 7.68 px.

Combining the experiments loses six nudge observations and seven labeled-set
observations versus default. Only two of nine jointly admitted labels improve;
seven worsen. Its same-13-pair labeled SN-FEIDA mean absolute step worsens from
0.0229 to 0.0418. This is not a rollout candidate.

Area stability after arbitration is partly mechanical: the gate directly
screens a related radius-change diagnostic. It is **not independent accuracy
validation**. Missing observations are excluded, never replaced by held fits;
all area comparisons above use the same source pairs and independently admitted
RAW texture scale, not each ellipse's own radius. The physical nudge and jump
sequences lack canonical target labels and absolute metric scale. No gaze-target
accuracy, accommodation or calibrated uncertainty claim follows from them.

The blocking replay shares live ingestion decisions, source admission,
registration and contact consumers, but uses optimistic post-SAM focus and does
not reproduce asynchronous live cadence, AF, UI or mouse calibration. Synthetic
source-order/in-flight tests complement it rather than replace a live smoke test.
GPU arms ran sequentially for resource/latency comparability, alongside CPU
validation; cross-day latency differences are not performance evidence.

Separate 16-exposure fixed/step-XY branches preserve the same first nine inputs
and produce exactly the same first nine fits within each policy. Each branch
uses 396x264 native pixels; the experimental branch adds `(12,8)` to the inset
offset at source 927. All seven later fitted geometries differ. In the existing
default control pair, mean same-exposure center/major-radius discrepancy is
6.65/3.30 px (seven jointly admitted sources). Photometry v2 yields 6.56/6.45 px
(seven); crop memory plus arbitration yields 5.78/4.29 px (six), with one
fixed-branch dropout but none in the moved branch. These are **disagreements**,
not errors against ground truth. They do not establish equivalent predictions
after a move, and the incomplete-label nudge cannot identify the correct branch.

Final validation: 762 passing no-default-feature viewer tests and 775 passing
SAM-enabled viewer tests, each with the same 41 pre-existing failure names and
17 ignored tests as the previous team pass. All nine new regressions pass.
The full suites are therefore not green. The SAM-enabled live build,
all-targets check, `git diff --check` and source-tree audit pass. No live viewer
or camera session was started in this resumed pass.

The next comparisons should test motion-aligned lighting evidence, uncertain
prior/current-competitor arbitration, and historical RAW re-encoding/model-grid
alignment. Do not relax contact jumps, RAW support, sign or visibility gates to
make these experiments appear to preserve tracking.

## Enabled change: spend the sparse motion budget in the overlap

`NativeGlobalSimilarityTracker` already converted both frames' local patch
coordinates to native sensor coordinates. Its feature selection nevertheless
spent part of its 10-by-8 budget on the previous crop's outgoing pixels.
An overlapping reframe could consequently lose independent motion/scale
support even when plenty of common texture remained.

The feature grid now covers the intersection of the two sensor rectangles,
with the existing patch margin. It still selects at most 80 corners, searches
the same bounded native-resolution neighborhoods, and applies the same RAW
matching, forward/backward, uniqueness, robust-fit and admission tests. The
final support must still span the original ROI: a narrow overlap does not get
to claim whole-eye precision simply because its local fit looks good.

Crop translation is not eye translation. The similarity remains centered at
the **previous sensor-space ROI center**, and consumers must use that center
when transporting positions. Its rotation field is an affine matrix
coefficient, not an angle in radians. Independent linear scale is
`hypot(1 + scale_delta, rotation)`.

## Experimental SAM identity continuity: disabled by default

SAM's spatial features and mask memories carry crop-dependent context and
positions. Retaining those tensors unchanged after a crop move is unsafe.
The reset also discards object-pointer history and the RAW-validated
detector-query identity, forcing the next frame through cold candidate
selection; these dependencies need to be separated rather than assuming every
stored tensor is an image grid.

The opt-in `BUTTERCUP_SAM31_REFRAME_IDENTITY=1` separates the two:

- Clear all crop-addressed SAM memory on relocation.
- Retain the query embedding only for compatible epoch, prompt, dimensions,
  increasing source sequence/time, at most 900 ms since a RAW-confirmed mask,
  and at least 80% of that foreground still visible in the new crop.
- Compare at most four current query alternatives with embedding cosine at
  least 0.70 and sensor-aligned mask IoU at least 0.50.
- Recondition only from a **fresh current-frame mask** that passes the existing
  shape and untouched-RAW ring gates. An old footprint is association evidence,
  never a fresh observation or a synthesized fit.
- Keep the existing three-processed-miss release. Repeated moves cannot renew
  that budget or the last RAW observation's source clock.

These are engineering association bounds, not calibrated probabilities.
The experiment remains off because its labeled low-light coverage regressed.
Ordinary/default SAM spatial-memory crop resets retain their established
behavior, while host contact/session state now survives compatible moves.
This work does **not** claim to have solved learned memory transport across crops.
Repeated buffers with an identical source timestamp are now ignored without
renewing or resetting temporal memory, even if the sequence differs.

`SAM31_REFRAME`, `SAM31_REFRAME_RECOVERY` and `SAM31_REFRAME_QUERY` traces explain
identity retention and failures when `BUTTERCUP_SAM31_VIDEO_TRACE=1` is set.
The replay report records the active policy explicitly.

## Historical findings: initial pre-team replay

All artifacts below are runtime data under
`outputs/roi-reframe-20260905/`. No captures, models or compiled artifacts were
added to the source tree. Evaluations use `mild-blur` preprocessing and the
production SAM video worker; CPU motion replays ran alongside GPU evaluation.

**Historical scope correction:** these initial replays did not exercise the live caller's
sensor-origin-triggered session reset described in TODO item 1. They therefore
did not establish end-to-end continuity, and the opt-in identity preservation
could not survive that live epoch change. The later shared-caller fix and
evaluations above supersede that integration limitation. Identical baseline/candidate ellipses
below mean agreement between software versions on the same moving-crop inputs,
not equivalence between moved and unmoved crops.

| Test | Baseline | Candidate / finding |
| --- | --- | --- |
| Recorded right-eye nudge, 35 exposures, sequences 918–952 | Independent motion valid on 33 frames | 34 frames valid; all 34 transitions supported |
| Nudge at sequence 927, sensor shift `(+32,+24)` | 7 robust inliers, scale rejected, stability reset to 0 | 11 inliers, residual 1.19 px, stability reaches 9 |
| Same nudge, SAM outer limbus | 35/35 RAW admitted | 35/35, identical fitted ellipses with either identity prototype |
| Every third exposure of the nudge, 12 frames | 12/12 RAW admitted | Four-query prototype: 12/12, identical fits |
| Fixed left-eye control, same 35 exposures | 14/35 RAW admitted | Single-query prototype: identical admission and geometry |
| Recorded large-jump subset, 40 right-eye frames | 29/40 SAM admitted; 27 frames with motion support | Same coverage; neither large move claims reliable global motion |
| Labeled RAW inset-cycle stress, 37 frames / 14 targets | 35/37 SAM admitted, 13/14 targets admitted | Single-query prototype: 28/37 and 11/14; four-query prototype: 30/37 and 12/14 |
| Labeled fixed-inset control, same frames and crop size | — | 34/37 admitted, 13/14 targets admitted |

The large-jump source is
`outputs/labeling/roi-jump-1788101004/extracted/`: sequence 309 shifts
`(+144,-144)` and 317 shifts `(-176,+128)`. At the first move only about 33.6%
of the last RAW-confirmed foreground remains in the new crop. SAM produces no
current mask on seven consecutive available exposures through sequence 316.
This is a severe clipping case, not evidence that a tiny follower nudge needs
global reacquisition. The small nudge is from
`outputs/iris-arc-compatibility-20260905/capture-1788594056/`.

With the final defaults, the recorded nudge still has a signed SN-FEIDA log
step of -0.09585 (about a 9.1% area decrease). Reliable motion transport does
not fix that fitted-limbus inconsistency, and absent canonical labels we cannot
attribute it specifically to reframing or claim improved limbus localization.

The labeled subset is the existing prediction-free
`outputs/flat-tire-area-20260905/canonical-triplets-v2/` index: 37 unique RAW
frames, with 14 target annotations from the canonical `annotator/labels`
locations. The target sequences are 222, 223, 224, 226, 247, 10091, 10124,
10163, 10181, 10193, 10206, 10217, 10235, and 195. Labels are read only after
the entire inference sequence has finished, never for seeding or selection.

The four-query experiment loses baseline admissions at 10125, 10163, 10164,
10194 and 10207. The lost labeled fit at 10163 was itself poor (31.13 px
visible-point RMS). That does **not** establish that the other four missing
fits are improvements: those frames lack target labels. All 12 commonly
admitted labeled targets have exactly the same localization errors, including
the still-poor 30.54 px result at 10193. There is no localization win to justify
enabling the experiment.

All 17 commonly supported consecutive
[SN-FEIDA](flat-tire-area-and-motion.md#canonical-diagnostic-sn-feida) log steps
are also unchanged by the four-query prototype. Their median absolute log
step is 0.0264 and maximum 0.5315. The independent scale comes from RAW texture,
not the candidate radius. Unsupported motion and missing current observations
are not bridged. These are relative area-consistency diagnostics, not metric
anatomy, gaze accuracy or calibrated uncertainty. The physical nudge and large
jump sequences have no canonical target-label scoring in this evaluation.

## Reproduce and extend

After the normal SAM-enabled build/runtime environment is configured:

```sh
data/target/live/buttercup-eye-viewer --offline-sam-sequence-eval \
  outputs/new-report.json CAPTURE_DIR subject-right 0 37 1 inset-cycle
```

The optional final argument is `native` (default), `inset-fixed`, `inset-cycle`,
or `inset-step-x`, `inset-step-y`, `inset-step-xy`. All step modes use the same
first nine exposures and fixed crops, then diverge in independent runs. For a
native 384x256 ROI, all inset modes use 360x240 native
pixels. The fixed offset is `(12,8)`; the cycle uses `(12,8)`, `(24,16)`, `(0,0)`,
`(24,0)`, `(0,16)`. There is no resizing, padding or CFA phase change. Derived
views retain the original sensor-read clock and are **not independent new
exposures**. The `frame` field remains the original RAW manifest record;
`processed_roi` describes the actual input crop. Reported outer/pupil ellipses
are translated back into original-frame coordinates for label comparisons.
Post-SAM pupil diagnostics remain explicitly processed-ROI-local.

Use separate output files for baseline and experimental runs. Clear all four
experimental flags explicitly for the baseline; the current matrix uses the
three flags listed above with `BUTTERCUP_SAM31_REFRAME_IDENTITY=0` throughout.
Match original source records and crop policies;
do not compare area-only aggregates from different admitted subsets as an
accuracy gain. The source stride can simulate slower SAM query cadence without
changing exposure timestamps. Gaps beyond 900 ms reset the replay motion chain.

Synthetic regressions cover repeated positive/negative X/Y crop shifts, static
texture, simultaneous real translation/scale, tiny/no overlap abstention,
exact RAW sample preservation, incompatible clocks/epochs/sizes, immutable
identity age, the miss budget, and rejection of sensor-displaced lookalikes.
The historical pre-team no-default-feature viewer suite had 712 passes, the same 41 known
failures as the checkpoint, and 17 ignored tests; no failure names changed.
The historical SAM-enabled suite had 718 passes, the same 41 failures, and 17
ignored. Final-default replays reproduce all 35 nudge ellipses and admissions,
and all 37 inset-cycle outer/pupil results, admissions and label errors from
their respective baselines. The improvement enabled here is continuity of the
independent motion/scale evidence, not a newly accurate SAM segmentation model.

## Live pivot and sensor-band transactions (September 6)

The viewer always negotiates
`REGION_CAPS 1` with the external camera service. The camera implementation
belongs to Podbay, not this repository. Old cameras retain the paired stream.
Successful transactions retain the TCP stream and source epoch; source-keyed
`ORG1` metadata binds the applied band, ROI coordinates, and active-eye mask.
Queue acknowledgement is not application. Only applied metadata grants
residency/hysteresis credit. Discarded transition buffers are not observations.

`pivot_region_scheduler` plans sparse sensor-coordinate support around the
projected, movable 3D pivot. SAM uses the source-matched implied globe center
only when the same ellipse is RAW-admitted and its sign resolved; the smoothed
iris center and presentation-only contact are not steering evidence. The
visible iris cap has an explicit offset from that pivot: the invisible whole
globe need not fit inside a fine ROI. Source-keyed coherent pivot translation
supplies prediction, while current gaze offset supplies containment, not head
velocity. Default engineering margins are 60 ms prediction, a 12 px cap guard,
64 px sensor-band reserve, and bounded age/displacement. They are defeasible
support, not calibrated probabilities or metric depth measurements.

An otherwise contained crop survives missing 3D evidence. Actual packing
conflicts may temporarily evict an eye; its absent packets are neither failed
eye observations nor invented positive observations. Reentry requires fresh
source evidence for SAM publication. Manual ROI hold and checkerboard mode
disable automatic steering. Ordinary crop moves retain the 250 ms command
cadence; a verified discarded band transition has a bounded exposure-dependent
wait and a one-second throughput recovery interval. Missing active packets and
unannounced stalls remain errors. Single-eye recording termination and actual
membership across reconnects are explicitly covered.

Validation for this patch: 30 focused host region tests, one source-time pivot
prediction test, one G-label test, 24 camera-service tests, ARM build, and the
SAM-enabled viewer build passed. Camera-side protocol smoke tests passed with
one connection through upward/downward moves, both eviction directions, and
pair restoration. At frame length 700, measured command-to-first-complete-set
latencies for the three band moves were 116–127 ms; at frame length 1730 they
were 300–312 ms. Each reported two discarded transition buffers. Focus target
and generation, exposure, and final sensor geometry were unchanged by each
smoke test. These are short examples, not a calibrated latency distribution.

Runtime evidence is in `outputs/region-live-smoke-20260906/` and
`outputs/region-live-smoke-long-20260906/`; build/test logs are
`outputs/region-final-tests.log`, `outputs/region-sam-build.log`, and
`outputs/camera-region-deploy.log`. Deployment was RAM-only. The restarted
viewer enables the extension by default whenever the camera supports it.
There is no environment/config switch. The session-only `W` hotkey pauses or
resumes automatic following without switching the camera protocol. Following
starts on at each launch; already queued transactions finish normally.

Remaining validation: independently verify first-delivered RAW pixels against
old/new optical origin references, then matched baseline/candidate motion
corpus replay with labels, coverage, source alignment, independent scale, and
SN-FEIDA. This patch has not run that corpus comparison and establishes no
fitter/localization improvement. Register readback and fixed buffer discards
do not independently prove a same-sized DMA buffer's physical origin. Camera
timestamps remain post-extraction wall-clock samples, not hardware exposure
timestamps. Continuous transport is therefore not a zero-frame-loss claim.
