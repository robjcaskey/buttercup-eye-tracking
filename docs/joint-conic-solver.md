# Source-aligned joint conic gaze solving

Current integration update (2026-09-12): the live distribution uses discarded
pilot fitting, conditional nuisance proposals, analytical integration of
unobserved inner radii, and four sampling batches with a numerical admission
margin. The matched comparison below records substantial recent-SAM coverage
gains, remaining SAM/Student losses, and increased work. This improves numerical
integration; it does not establish native gaze accuracy or calibrated model
probabilities. The older experimental verdicts below describe their dated
controls; `IntegrationConfig::live()` specifies the current production recipe.

The matched native calibration audit below confirms more qualifying sources
with identical fitted conics and source handling. Both brighter clips retain
their final fits, but the dark clip still fails with the required 500 ms source
settle. Its within-target scatter is comparable to neighboring target spacing;
having enough numerically supported samples does not establish useful accuracy.
The source-motion follow-up also finds an admitted 85-degree iris-plane switch
inside the middle clip. Matched million-draw references still support that
frame under the current observation model; monitor-fit acceptance and numerical
integration precision do not certify every native gaze sample.

Broader replay status (2026-09-08): implemented core and live adapter; the matched
replay covers all **387,519 surviving RAW exposures**, with mixed results.
Calibration source/phase handling and the off-axis distance singularity are
corrected. Paired host admission now anticipates work, and recording preserves
native ingress before analysis shedding. The current SAM viewer builds, as
does the no-SAM feature check. The full viewer test run has 1,036 passes,
39 existing failures and 29 explicitly ignored diagnostics; it is not fully
green. **Boundary selection and low-light gaze accuracy remain unresolved**;
the darkest recent calibration still fails its qualified, original-target-window
replay. Do not
equate synthetic proofs, build success, or monitor-fit acceptance with gaze
accuracy. Detailed results and negative controls follow below.

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
  Residuals are integrated along observed polylines. Information mass uses a
  provisional 32-native-pixel correlation length (wider for large blur bands),
  capped per arc. Eight tiny pupil fragments no longer receive eight times
  the weight of a long limbus arc. Shorter alternatives cannot inherit a longer
  alternative's coverage. These weights are heuristics, not sample counts or
  calibrated probabilities; their lengths/weights are exported for inspection.
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

Conic decomposition now also yields the circle-center/radius ratio. It starts
the nuisance depth/radius inside the **existing** scene support. An arbitrary
350 mm starting range must not put a large observed limbus outside every arc's
robust-loss basin. The nominal prior, hard bounds and independent SN-FEIDA scale
remain unchanged. If the two initialized depths violate the shared IPD bound,
the initial geometry is moved toward the nominal scene until feasible; the
constraint itself is never widened. No-feasible-initialization and
all-evidence-rejected outcomes have separate diagnostics.

Each alternative outer conic now initializes its **own** circle center/range
and metric radius, not just a new target direction around the first conic's
geometry. Starts interleave ROIs before advancing to their next conic hints.
The regression test for secondary-conic geometry failed before this change.

An explicitly unlocalized-ROI association can compete in the **same objective**.
It pays every omitted correlation group's full capped cost, exports no center,
normal or ellipse for that ROI, and removes its fitted pair-position constraint.
This is not an unpenalized choice between two independent gaze estimates.
`modeled_eyes` distinguishes an unlocalized ROI from a modeled ROI whose arcs
all became outliers; `unlocalized_eye_cost` retains the omission penalty.
The full request's clocks, timing and scene are validated before any omission.
The frozen coarse scene support remains conditional on its acquisition priors.

Initially reserving four starts for each unlocalized association needlessly
halved a 16-start joint search. The corrected scheduler first tries a joint
prefix, then prunes an association if its omitted-evidence cost **alone** exceeds
the best current total cost. All other residual costs are nonnegative, making
this a genuine objective lower bound rather than a confidence gate. Unused
starts return to the joint search; the total cap does not increase. Diagnostics
count starts by `[both, right-only, left-only]`, including infeasible starts.

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

Successful solves have a source-sorted, 32-entry target-seed history, separate
from presentation slots. A read with only one ROI remains useful. A later
partner supersedes its provisional same-read seed; a failed combined solve
retires that provisional result without erasing earlier successful reads.
Only strictly earlier sensor times within 500 ms can seed a new solve. Up to
two recent reads can supply separate competing starts within the fixed budget;
their targets are never averaged or added as residuals.
Clock/configuration changes clear history; crop translation and unrelated
display invalidation do not. No retained image, old arc, or previous gaze
becomes a new observation. These are bounded search starts, not a physical
temporal-motion constraint or gaze-point averaging.

Rejections retain a per-ROI **newest-observation timestamp** independently of
the optional latest successful fit. Otherwise clearing a new empty observation
could let an old, uniquely arriving second-eye result resurrect old geometry.
Recorded source 1766 reproduced that failure under a synthetic 100 ms left-eye
arrival delay. Both that case and invalid coarse-scene support now clear/retain
state according to the observation timestamp, not whether a fit pointer exists.
An old pair may still be returned as historical output without replacing a
newer ROI's live state. A source/provider generation change resets these floors.

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
the two full jobs were **still running at this initial checkpoint**. They and
the later source-ordered full-corpus comparison are complete in the subsequent
sections; do not read this historical pilot description as current run status.

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
All ten are now evaluated, using 76 fresh native SAM exposures around their
source positions (49 reads). Labels entered scoring only after fitting.
At this pre-partial-outline checkpoint, eight targets have a fit and two have
no boundary evidence. That candidate is frozen as `eval-pair-init`; the baseline is
`eval-projected` (before conic depth initialization and geometric arc weights).
Visible and guessed landmarks are separate; limbus localization is not gaze
ground truth.

| Reviewed source index | Baseline joint visible RMS, px | Candidate, px | Original SAM, px |
| --- | ---: | ---: | ---: |
| 105182 | 52.51 | 9.80 | 8.63 |
| 105191 | 44.13 | 35.92 | 46.14 |
| 105933 | 33.57 | 3.33 | 3.28 |
| 105935 | 3.11 | 3.33 | 3.40 |
| 105937 | 3.62 | 3.29 | 3.23 |
| 106295 | 9.26 | 8.92 | 8.17 |
| 105983 | 4.61 | 4.48 | 4.51 |
| 107262 | 2.59 | 2.40 | 2.95 |

Indices 106191 and 106217 remain absent in both algorithms; index 105191 remains
bad, not a success merely because its error decreased. Index 105935 regressed
by 0.21 px. Six reviewed labels have no exact match to this stereo RAW inventory.

### Expanded matched evaluation

Both frozen algorithms have now evaluated 50,000 exposures: indices 100–25099
and 193759–218758, with 30,973 reads, 19,027 RAW pairs, 10,483 reads with both
current extractor packets, and 61 capture entries. The candidate supplies
21,808 shared solves, including 10,444 with both eyes contributing. There are
9,161 no-boundary reads and four all-evidence-rejected reads, with no remaining
initialization failure on this subset. This is still **not all 387,519 exposures**.

The separate development-excluded comparison uses indices 10100–25099 and
203759–218758: 30,000 exposures in 17,979 reads. On exactly matched, admitted
joint outputs, right-eye withheld RMS median/p95 changes from 1.918/7.537 to
1.839/5.894 px (12,059 comparisons); left-eye from 1.928/7.218 to 1.810/5.640 px
(8,193). Right-eye errors improve by over one pixel on 808 reads and regress
on 366; left improves on 639 and regresses on 264. Admission gains are 130/26
right/left, with two/zero losses. Common accepted-arc comparisons are reported
separately, so rejecting a troublesome arc does not conceal the change.

These are **optimizer-only** trials: the cache, extraction and withheld points
are unchanged. The older frozen exports predate coordinate fingerprints; their
source identity, arc IDs/kinds/counts and unchanged extractor code establish
the comparison contract. New exports add withheld-coordinate fingerprints,
and the reporter rejects mismatches even when point counts agree.

On all 30,973 solver requests, including early abstentions, joint solve median/p95/max
is 2.41/10.30/18.14 ms while sharing CPU resources with the SAM exports. This is
not an uncontended benchmark or end-to-end live latency measurement.

### Independently scale-normalized motion check

The 50,000-exposure subset still has no recovered independent scale. A separate
post-fit audit joins the labeled-neighborhood exposures to the existing
`manual35` and `low0`–`low7` native RAW similarity reports. Exact RAW SHA256,
ROI geometry, sequence, timestamp and source lineage must agree. Of 109 motion
links, 75 meet the inherited motion-support heuristic; 61 lack an unambiguous
exact exposure join and two lack a fresh accepted outer boundary, leaving
**12 matched SN-FEIDA transitions** and **zero ROI reframes**.

For each link the previous exposure supplies a fresh local pixel-scale reference;
the independent scale ratio is `hypot(1 + scale_delta, rotation_coefficient)`.
The absolute log SN-FEIDA step median/mean/p95 changes from
0.05575/0.19379/0.52878 to 0.03280/0.03457/0.05892. This small subset supports the
initialization/weighting fixes alongside the labels; it does not establish
anatomical constancy or ROI-reframe performance. The emitted scale-only bounds
omit conic uncertainty and are not confidence intervals. Unnormalized area
tails on the larger subset still regress and cannot be relabeled SN-FEIDA.

The full viewer suite currently has 985 passing tests, the same 41 failures as
the pre-change baseline, and 24 ignored tests. Live-adapter tests additionally
check source deduplication, different ROI sequences on one clock, crop transport,
missing-eye behavior, provider changes, radius units and shared gaze mapping.
No live user calibration or desktop-pointer trial has been performed for this
new path.
The standalone evaluator has 96 passing tests, including its source-replay,
shared live-tracker, partial-extraction and local-linearization tests. All 22 independent report tests pass and
exercise source matching, probe changes, missing-read accounting, chronological
area transitions, rejected geometry and determinant-based scale normalization.

### Current bounded-search comparison

The frozen `eval-search-bound-v3` comparison uses the same 50,000 exposures and
unchanged training/withheld points. Against `eval-partial-control` with the
ordinary extractor, there are 19,780 right / 12,244 left matched withheld-error
comparisons and 107,874 verified coordinate fingerprints. Right p95 changes
6.382→6.356 px; left 6.940→6.879 px. Right/left improvements over one pixel are
48/35, versus 52/30 regressions. Admission gains are 2/6 and losses 8/11. This
is a modest conditional optimizer change, not a uniformly better localization.

The separate comparison against `eval-unlocalized-pose-v2` isolates the
cost-bound search scheduler: 19,782/12,246 matched right/left results, with
94/96 improvements and 37/49 regressions over one pixel. The earlier scheduler's
halved joint search is therefore not retained merely because its tests passed.
There are 10,338 available requests using all 16 starts on the paired model;
the total observed hypothesis maximum remains 16. Shared-CPU joint request
median/p95/max is 2.46/10.45/18.10 ms, not an end-to-end or exclusive benchmark.

With the still-experimental partial extractor, the same solver improves
829/635 right/left withheld errors by over one pixel and regresses 554/442,
relative to `eval-partial-v1`; admissions gain 159/92 and lose 158/192. The
right/left p95 remains very large at 74.15/72.79 px, including rejected probes.
The reviewed target 105933 regresses 3.33→3.49 px. Targets 106191/106217 still
have roughly 26.65/45.23 px visible-label RMS. These are not successful fits.
Native inspection of remaining large-error sources 2799, 5332, 15368 and
19322 shows skin/lid or heavily clipped off-target regions, not established
ground-truth limbuses. Boundary identity remains a separate unsolved problem.

The corresponding byte-matched independent-motion audit has 14 SN-FEIDA links,
zero reframes, median absolute log change 0.03976 in both arms and mean
0.16176→0.16235. The 1.66914 maximum remains. This comparison uses the former
partial candidate as its baseline: the extra 106213→106215 link was already
admitted there and is **not** a new solver recovery. Neither this admission
intersection change nor the small area differences demonstrate better anatomy.
No independent absolute scale exists on the 50k optimizer subset.

The expanded 100,000-exposure control/candidate evaluation has completed on
cache prefixes 100–50099 and 193759–243758: 58,253 reads, 41,747 RAW pairs,
26,598 both-evidence reads and 124 capture entries. There are 46,733 available
candidate solutions, including 26,487 with both eyes contributing. The
independent output auditor verifies shared-target rays and the 16-start cap.
There are still **287,519 available exposures not evaluated in this comparison**.

The separately inspected new half (25100–50099 and 218759–243758) contains
27,280 reads and 151,752 matched coordinate fingerprints. Its 22,771 right /
18,036 left matched withheld-error comparisons have p95 7.310→7.312 and
7.380→7.312 px, respectively; 61/53 improve over one pixel and 71/42 regress.
Admissions gain 3/7 and lose 15/11. Common accepted-arc comparisons are retained
separately. Native source 31883 contains a real eye, but the upstream contour
extends beyond the limbus: its higher contour residual does not by itself prove
worse anatomical localization. Source 40991 is a clipped skin/lid region. No
human label or gaze ground truth is available for either inspection.

Unlike the first 50k subset, this expansion has 1,391 right / 859 left
exposure-attached **coarse acquisition scale priors**. Source-code provenance
is `coarse_centimeter_scales`: MediaPipe apparent radius with an assumed 12 mm
limbus, updated at semantic reacquisition and held between updates. They are
independent of the tested SAM/joint candidate radius, but are **not 2,250 fresh
scale measurements**. Their measurement timestamps are not recovered, and
their wide bounds are heuristic. Normalizing with these priors yields 1,261/753
matched right/left log-area steps: unchanged medians 0.014742/0.013958 and
maxima 1.97919/1.41501. This is conditional coarse-prior-normalized SN-FEIDA,
not independently measured frame-to-frame physical-area constancy. Keep it
separate from the 14 exact-RAW motion links above, and do not treat held scale
values or stable wrong ellipses as fresh corroborating evidence.

### Recorded-source live-tracker replay

`buttercup_stereo_conic_eval --source-order-replay` disk-indexes the immutable
caches and passes the actual sparse native packets through `JointTracker`.
It retains only the live 32-packet/ROI, 1.5-second evidence window; the offline
disk-offset index is separate. Optional `--arrival-delay-ns ROI NS` changes
arrival scheduling, never source time, RAW bytes, origin or sample coordinates.
It is not a replay of SAM video memory or measured GPU completion times.

The pre-fix tracker failed the native replay at source 1766 after a newer empty
right-eye observation. Two unit tests independently reproduced stale revival
and failure to clear an invalid scene. After the timestamp-floor correction,
all three frozen `eval-source-watermark-v4` replays completed and passed the
independent Python source/geometry audit:

| Arrival scenario | Unique RAW exposures | Same-read RAW pairs | Native crop moves | Duplicate checks | Shared publications using both eyes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source order | 50,000 | 19,027 | 879 | 50,000 | 10,438 |
| Right ROI delayed 100 ms | 50,000 | 19,027 | 879 | 50,000 | 10,438 |
| Left ROI delayed 100 ms | 50,000 | 19,027 | 879 | 50,000 | 10,438 |

Each scenario covers the same 30,973 reads, 61 capture entries and 108 source
lineages. The auditor checks exact publication-source identity, co-clocked
pairing, the single shared target/rays, source age rather than arrival age,
per-eye rejection floors and duplicate suppression. It counts unique arriving
RAW exposures, not repeated same-read publications. Right-first and left-first
monocular interim availability differs; that is not an extra stereo dropout.
Sorted paired source identities plus contribution flags have the same SHA256
in all three scenarios (`81bf877c7698a7d2c57d8a38fc902fc8247b460be463f5366c6dda17c955f0c4`),
so this comparison is not relying merely on equal aggregate admission counts.
There are 757/756/757 fresh fitted crop-move arrivals in the displayed order.
These checks establish source-state correctness, **not** geometric accuracy or
SN-FEIDA stability across those 879 moves. Their independent scale/label support
still needs evaluation. No viewer restart or live user calibration was performed;
the updated viewer builds successfully.

## Known failures and remaining work

### Partial-outline experiment (not enabled in the live solver)

`outline_conic_segments/partial_outline.rs` now shares the observation-only
flat-tire chord censor with the legacy complete-ellipse fitter. It accepts up
to four ranked mandatory-outer-iris mask contours, checks outward native RAW
contrast, preserves gaps, and emits at most eight correlated sector groups
with four alternatives and 16 points per arc. Different SAM queries share a
fixed sector budget. Optional conic fits are starts only; a synthetic mixed-eye
test also succeeds after discarding every complete-ellipse seed for the
partial eye. Flat/saturated RAW, reversed contour winding, chord exclusion and
duplicate-query budgets have independent tests.

The current experimental extractor additionally excludes deep inward-curving
notches using the convex hull of the same 128 measured samples. The 3 px
allowance is in 384-wide model coordinates and is an engineering raster/jitter
tolerance, not calibrated anatomical uncertainty. The hull only censors points:
neither its closing chords nor its vertices become replacement observations.
The existing two-sample tangent padding also applies around the excluded bites.
This catches curved bright occluders that pass outward RAW polarity and the
old straight-chord test. Rotation, translation, winding, raster jitter and
exact sampled-point provenance have tests; the bright-bite test failed before
the change with 44.77 px of false inner-boundary support.

This rule assumes a mask is a subset of a convex projected disk. A convex but
wrong reflection frontier, an outward semantic extension, or a skin/lid mask
can still pass. It is not a general boundary-identity solution.

The frozen `eval-partial-v1` trial is deliberately offline-only, selected by
`--partial-outlines`. Its matched control is `eval-partial-control`. Both ran
the same 50,000 exposures described above, and 107,868 withheld-coordinate
fingerprints matched for the existing admitted outputs. Admissions increase
by 5,859 right / 4,775 left, but 13 / 15 existing admissions are lost. Among
unchanged probes, 191 / 162 regress by more than one pixel versus 44 / 36
improvements. More accepted ellipses are **not** proven better localizations.

The two formerly absent reviewed targets, 106191 and 106217, now fit with
26.65 and 45.23 px visible-label RMS, respectively: still unacceptable. The
other eight reviewed target results are unchanged. Native overlays show
contamination by lid/reflection boundaries. Source 212420 reveals a second
problem: even when every new arc from its partner is rejected, that partner's
initialized geometry and pair constraints can damage the previously supported
fit. Native inspection of 212419/212420 and 15631 shows off-target or heavily
clipped ROIs, not established ground-truth irises; those large mask-residual
regressions must not be described as measured gaze/localization errors.
Thus widening localization bands is not a complete treatment of uncertain
boundary identity or a genuinely unlocalized second eye. The subsequent
penalized unlocalized-eye model, per-conic geometry starts and search-budget
correction are described and evaluated above. They do not yet repair the
remaining boundary-identity and partial-outline localization failures.
No independent absolute scale exists on this 50k subset, so the area summaries
cannot establish an SN-FEIDA improvement.

The separate byte-matched RAW motion audit has 13 usable local-reference
SN-FEIDA transitions for this newer control/candidate pair, with no ROI
reframes. Both arms are identical on those links: absolute log-step median
0.03572, maximum 1.66914. The additional link, 106193→106195, was absent from
the earlier 12-link comparison because its older baseline lacked accepted
support. Its large area jump is a remaining failure of the current baseline,
not an improvement or regression attributable to partial-outline extraction.
Changing a comparison's admission intersection must not conceal that tail.

`--export-sparse-evidence` adds exact training samples and optional seeds to
small diagnostic replays. The native RAW preview shows all alternatives in
magenta and selected samples in cyan, separately from reconstructed ellipses.
These diagnostic coordinates and recordings remain outside the source tree.

### Convex-notch and rejected-conic comparisons

`eval-convex-outline-v5` isolates the notch exclusion against frozen
`eval-search-bound-v3` with partial extraction enabled in both. The matched
50,000 exposures are indices 100–25099 and 193759–218758: 30,973 reads,
19,027 RAW pairs and 61 capture entries. They are **not** the entire available
387,519-exposure corpus; 337,519 exposures are outside this particular trial.

Changing extraction changes some withheld probes. The report now requires
explicit `--allow-extractor-changes` for such a comparison. RAW/source identity
must still match exactly. A per-eye residual comparison is skipped unless
every probe's structure and coordinate fingerprint match; unchanged probes in
the other eye remain comparable. Probe equality does not assert equality of
all training samples. Missing hashes cannot certify an extractor comparison.
Admission and matched source-time area diagnostics retain changed-extraction
eyes rather than silently shrinking the entire comparison.

For v5, 3,626 right / 3,217 left changed probe sets are explicitly skipped,
and 116,223 coordinate fingerprints match. On the 21,520/12,886 fixed-probe
comparisons, 148/146 improve over one pixel and 95/83 regress. Admissions gain
168/196 and lose 443/717. There are 27,001 available shared solutions, 14,962
using both eyes. The 16-start budget is preserved. No candidate-independent
scale is available on this subset, so its pixel-area changes are not SN-FEIDA.

The separate 76-exposure reviewed-label replay changes target 106191 from
26.65→20.29 px visible-label RMS and 106217 from 45.23→14.28 px. Both remain
poor fits; the other eight matching reviewed targets are unchanged. Labels
are post-fit only; six of 16 reviewed labels still have no matching available
RAW. All 14 exact-RAW independent-motion links remain comparable, with no ROI
reframes. Mean absolute SN-FEIDA log change is 0.16235→0.15328, median 0.03976
unchanged, and maximum 1.66914 unchanged. One modestly improved transition does
not establish physical constancy. Native regression inspection at 216130,
5022, 212573 and 201151 shows lid/skin or clipped-eye false geometry; neither
arm establishes correct limbus localization on those images.

`eval-partial-policy-v6` also tries the partial extractor when a complete
upstream ellipse exists but fails its RAW admission gate. Previously that
rejected fit bypassed the RAW partial-boundary checks entirely. In the explicit
partial experiment, those old sections and their center prior are replaced,
not double-counted alongside new arcs. The old ellipse remains diagnostic
output. The accepted control's extraction is unchanged. A flat-RAW fixture
reproduced the bypass before this policy change.

Against v5 on the same 50k exposures, v6 has 26,771 available solutions and
14,608 using both eyes. Admission gains are 35/24 and losses 228/415. There
are 150,238 matched coordinate fingerprints, with 680/438 changed probe sets
skipped. Fixed-probe improvements over one pixel number 109/68 and regressions
106/59; corresponding p95 values slightly worsen. This is not a general
accuracy/coverage win and the experiment remains disabled in the live bridge.
Native inspection of the largest v6 fixed-probe regressions, 206209 and
213207, likewise shows clipped-eye/lid or off-target skin geometry rather
than a labeled true limbus. These remain failures, not interchangeable
accurate solutions simply because their robust objectives are small.

The reviewed target 105191 loses its erroneous 35.92 px fit rather than being
recovered: only the other eye contributes. Five measured partial arcs were
offered, but their penalized unlocalized-eye alternative won. The remaining
nine reviewed target results are unchanged from v5. No zero error is imputed
for that dropout. The same 14 independent-motion links remain; their maximum
absolute SN-FEIDA log step drops from 1.66914 to 0.67411, but native source
106195 now has a too-small reflection-bounded slice in place of the old
oversized lid ellipse. Both are wrong. This is an explicit example of a better
area statistic **not** proving a better anatomical reconstruction. There are
still no independently validated crop moves in this small motion subset.

The SAM viewer build succeeds, with exactly the same 41 full-suite failure
names as the pre-change baseline. No viewer restart, live calibration, SAM
memory change or partial-extractor enablement accompanies these experiments.

### Measured image-boundary directions

`BoundaryNormalObservation` preserves a measured outward **2D image-boundary
normal** and an engineering angular sigma alongside its actual contour sample.
It is not the 3D iris normal or gaze ray. Missing observations remain missing;
the accepted-complete-SAM and RAW-pupil adapters do not manufacture directions
from fitted ellipses. The partial adapter retains the measured ordered-contour
tangent already used for its RAW polarity check. Winding, point/normal alignment,
training decimation and malformed-input behavior have tests.

Direction uncertainty combines a 15-degree floor with 1.5-model-pixel endpoint
jitter over the measured tangent span. These are engineering assumptions, not
empirical confidence intervals. Position and direction share one arc's length
weight, correlation group and complete outlier cost; directions do not create
independent votes. Two mirror-related 3D circles with the same projected conic
also have identical image normals. A separate test explicitly prevents calling
that observation new 3D sign evidence.

Frozen `eval-boundary-normal-v7` penalized angular error directly. Against v6
on the same 50k partial-extraction exposure subset above, it loses 1,231/2,406
right/left admissions and gains 32/20. Fixed-probe regressions over one pixel
outnumber improvements, 2,091/931 versus 1,610/741. Its 143,577 matching probe
fingerprints verify unchanged coordinates, not anatomical correctness. The
reviewed bad target 105191 returns with 43.81 px error, while 106191 and 106217
worsen to 21.42 and 14.76 px. The 14 independent-motion links also worsen:
mean absolute SN-FEIDA log change 0.08220→0.10574, maximum 0.67411→0.99068.
This is a rejected direct-penalty formulation, not a claimed improvement.

`eval-normal-band-v8` instead uses direction as a compatibility allowance:
there is no angular pulling force within two supplied engineering sigmas;
only excess disagreement is penalized. This is not a statistical two-sigma CI.
Against the point-only v6 control, the same 50k comparison verifies 148,986
probe fingerprints. Admissions still lose 616/1,567 and gain only 27/30.
Right/left fixed-probe improvements number 1,266/738 and regressions 1,340/956.
The common accepted-arc subset improves more often, but selecting that subset
alone would hide dropouts and incompatible groups. Unnormalized area tails
still regress, and this large subset has no independent scale support.

The reviewed-label bad revival at 105191 is gone in v8; that eye is rejected,
not recovered. Target 106191 is 20.31 px versus v6's 20.29, and 106217 is
14.28 px in both. The other seven accepted reviewed targets are unchanged.
Mean absolute SN-FEIDA log change on the same 14 links is 0.08220→0.08210,
with unchanged median and no ROI reframes. This negligible area difference
does not establish an accuracy win. Native inspection confirms that source
106195 still fits a reflection-bounded slice rather than the whole iris.
The largest 50k residual regressions at 15361 and 19316 show clipped-eye/lid
or off-target structure in both arms, not established correct limbuses.

The separate ordinary 50k comparison, where no measured directions are supplied,
has exactly unchanged admission and residuals relative to the frozen v3
ordinary solver: 107,888 probe fingerprints match. Thus the new optional
observation contract does not silently alter ordinary extraction. Neither
partial-boundary formulation has been enabled in the live bridge.

### Rejection activity during numerical linearization

Frozen `eval-active-arc-v9` fixes a reproduced optimizer defect independently
of the boundary identity problem. At a capped arc's rejection boundary, the
old finite-difference calculation could cross from a constant position-only
cost vector to an uncapped mixed position/direction vector. Their squared
costs are nearly equal, but their residual components jump. Differencing that
jump gives a rejected arc a fictitious force in the shared solve.

The selected alternatives and rejection activity are now fixed only while
estimating one iteration's local derivatives. Every actual proposed step
reselects alternatives and recomputes the original capped objective, and the
next iteration refreshes activity. This is not temporal exclusion memory,
weaker rejection, extra anatomical bounds, or averaged monocular gaze.
The regression test failed before the fix, exercises a real finite-difference
gate crossing, and verifies both zero rejected-arc force and re-admission.
A second test reproduces the same discontinuity with negative position
residuals and no measured directions, covering ordinary live evidence too.
This residual-vector sign issue is not itself a physical gaze-sign correction.

All 85 standalone tests pass. The ten matching reviewed-label outcomes are
unchanged from v8, including the rejected eye; six of 16 reviewed labels still
lack matching RAW. The SAM viewer builds, with 974 passing tests, the same
41 pre-existing failure names, and 24 ignored tests. The 14 independently
supported RAW-motion/SN-FEIDA links are exactly unchanged, with no ROI reframes.

The completed strict 50k partial-arc comparison verifies 149,155 fixed-probe
fingerprints. Right/left improvements over one pixel number 4/0 and regressions
1/6. There is one right-eye admission gain and one loss in each eye. Most
outputs are identical. Native inspection of 19023 and 18520 shows the same
clipped-eye or off-target failure class in both arms. Source pair [15823,15824]
is different: these are actual visible eyes, not disposable negative examples.
The old solution omits the right eye. The new shared solution uses its 139 px
arc and two left-eye arcs while rejecting a 75 px left arc; the left fixed-probe
RMS changes 1.57→10.51 px. Both eyes' measured arcs and reconstructed ellipses
were inspected separately. Without human labels for this pair, the anatomical
winner is unresolved; a smaller coupled objective does not settle it.

The expanded ordinary comparison uses indices 100–80099 and 193759–273758:
160,000 exposures, 91,564 reads, 68,436 RAW pairs and 137 capture entries.
This leaves **227,519 available exposures outside this comparison**. Both arms
use the identical extraction and frozen priors; 338,819 probe fingerprints
match. There are 67,244 available shared solutions, 36,245 using both eyes,
and no initialization failures. Compared with v8, one right eye is newly
admitted and none are lost. Right/left improvements over one pixel number 3/5
and regressions 1/4; p95 fixed-probe errors change only slightly, from
6.241→6.236 and 6.814→6.804 px. This is not a broad accuracy breakthrough.

Separately reporting the 60k exposures beyond the earlier 100k comparison
avoids hiding new failures in a mostly unchanged earlier prefix. That subset
has 33,311 matched reads, 79,157 verified probe fingerprints and unchanged
admission. It has no improvements over one pixel and 1/2 right/left regressions.
All three sources were inspected natively: 54454 contains a real eye whose
ellipse changes, 53310 is heavily clipped, and 244852 is partially clipped
with an occluded upper boundary. They are unlabeled localization questions,
not evidence of improved anatomy simply because a different hypothesis wins.

The 2,399/1,205 normalized-area transitions in the larger report are unchanged.
Those use held coarse MediaPipe scale under the existing assumed 12 mm limbus,
not that many fresh independent physical-scale measurements. They are distinct
from the 14 exact-RAW-motion links above. The 16-start cap remains verified;
shared-host joint-solve time, including unavailable reads but excluding RAW
preparation and SAM, is 2.44 ms median / 8.45 ms p95 / 14.81 ms maximum. These
are workload diagnostics, not isolated end-to-end real-time benchmarks.

Expanded v9 source-order replays with native, right-delayed and left-delayed
arrival schedules completed on the same 160k exposures. All three passed
source-state checks, including 2,738 native ROI reframes and 160,000 duplicate
checks each. This was insufficient: a separate comparison of the actual paired
gaze geometry exposed the arrival-dependent initialization defect below.

Read-only metadata inspection also found recorded nine-point calibration
episodes, including original capture entries 104, 107–109 and 116–118. These
can support a separate post-fit target-response study. Their target events
are host submissions, not measured scanout or eye fixation, and the archives
do not supply a measured camera-to-monitor transform or bounded sensor/host
clock mapping. Recorded gaze estimates have not been fed to this solver or
substituted for true gaze labels.

### Paired initialization must not depend on display-slot survival

The original tracker selected its previous target from the two latest display
slots. Arrival order, a new invalid observation, or a newer fast-eye result
could erase a usable past joint seed or leave a different monocular seed in one
slot. This changes an optimization start even for the same completed RAW pair.
On v9's 68,436 matched pairs, a synthetic 100 ms right-ROI delay changed 493
targets by over `1e-6 mm` and 63 individual eye rays by over one degree;
40 exceeded five degrees. The maximum ray change was 120.66 degrees. Pair
[44082,44083] changed by 90.77 degrees while both eyes contributed and both
reported cost margins exceeded the existing heuristic threshold of two.
Correct clocks and equal aggregate admission counts did not detect this.

`eval-source-seed-v10` separates bounded paired seed history from presentation.
Both exact exposure inputs must be present to record a paired seed; either eye
may still be rejected or unlocalized by the joint objective. Source-sorted
lookup never selects equal/future times, never revives rejected geometry and
does not rewind current displays when an old pair completes. Duplicate seeds
cannot refresh history. The paired-history fix does not change conic residuals,
anatomical bounds, extraction, or the 16-start/12-refinement budget.

The two new regression tests failed before this change and now pass. Tests also
cover capacity, out-of-order insertion, expiry, lineage/configuration reset,
native crop translation and useful monocular continuity. There are 90 passing
standalone tests. The SAM viewer builds; its 979 passing, 41 failing and 24
ignored tests have exactly the same failure names as the pre-change baseline.
No GUI restart, user calibration or live partial-extractor enablement occurred.

All three v10 160k replays and their independent source/geometry reports are
complete. Each contains the same 91,564 reads, 68,436 RAW pairs, 137 capture
entries and 175 clock lineages; **227,519 available exposures remain outside
this comparison**. Each retains 2,738 native crop moves and passes 160,000
duplicate checks. Native, right-delayed and left-delayed schedules produce
exactly the same 51,936 available paired targets, same-eye rays and contribution
flags, including 36,263 using both eyes. There are no paired target differences
over `1e-6 mm`, and the measured same-eye ray differences are exactly zero.
This proves invariance for these constant-delay schedules and their available
past paired history, not arbitrary packet loss or every possible reordering.

Genuine singleton reads are reported separately, not hidden or counted twice.
Their available past evidence and monocular initialization can still differ
with scheduling; the largest measured same-eye change is 62.90 degrees on
73579. Its cost margins are below the existing sign heuristic in both arms.
The final-read join retains failed pairs and unpaired dropouts, so the area
diagnostic cannot bridge them merely because interim publications existed.
The report now compares geometry as well as provenance, with tests detecting
changed gaze despite unchanged RAW and rejecting missing/altered source reads.

Against v9's **native-order** output, v10 is not geometrically identical: 352
paired targets change over `1e-6 mm`, one formerly unavailable pair fits,
and right/left admissions gain 4/1 and lose 0/1. The 338,857 fixed-probe hashes
match. Right/left improvements over one pixel number 13/17 and regressions
15/4. All 2,399/1,205 coarse-prior-normalized area steps are unchanged, but this
cannot certify the changed gaze directions or unlabeled localization.

Native RAW inspection of both versions shows off-target skin/structure or
clipped-boundary fits at 235187, 63461, 261967 and 257086. In contrast, 70259
contains a visible iris: v10 narrows its ellipse and its fixed-probe RMS worsens
3.23→38.65 px. Pair [213838,213839] shows closed lids; the newly admitted right
ellipse and the much larger hypothesis margin are not evidence of an iris.
These remain boundary-identity/search failures, not successes conferred by
repeatability, a smaller objective, or valid camera-facing surfaces.

A separate **ordinary, source-timed** replay of the 76 reviewed-label-neighborhood
exposures gives 49 reads. All ten label-matched outcomes are unchanged: eight
accepted and two missing, with the persistent 35.92 px visible-edge RMS failure
at 105191. Six of the 16 reviewed labels lack matching RAW. The post-fit label
scorer now omits unexported SAM/monocular reference methods rather than
inventing failed detections. The ordinary replay has 13 matched independent
RAW-motion/SN-FEIDA links, no ROI reframes, and one missing fresh outer-boundary
link. Its mean/maximum absolute log steps are unchanged at 0.16097/1.66914.
Do not confuse this with the earlier 14-link **partial-extraction** experiment.
All 22 report/scoring tests pass. Shared-host native tracker timing is 2.31 ms
median / 7.71 ms p95 / 16.33 ms maximum, excluding SAM and RAW preparation;
this is not an isolated end-to-end benchmark. Full-corpus export remains in
progress, and neither the remaining RAW nor gaze ground truth is substituted
with these subset results.

### Retaining useful missing-partner initialization

Tracing 70259 through the exact source replay exposed a limitation of v10's
paired-only history. Its preceding successful pair was 182.83 ms old, while
several newer single-ROI reads existed. v9 happened to retain a 102.70 ms-old
left-only result in the display slot. v10 fixed arrival dependence by excluding
all such singleton starts, not by retaining their useful information reliably.
That is unnecessarily lossy when a partner is absent or unavailable.

`eval-source-history-v11` therefore remembers the latest successful solve of
each physical read, with the same 32-entry/500 ms bound. Source time, not
display-slot survival, chooses initialization for both single and paired
observations. A second ROI supersedes the provisional record for its read;
its successful joint result replaces that record, and a failed combined solve
removes it. No same-time result initializes itself, no duplicate refreshes a
seed, and no independent target average is formed. Three added tests cover
missing-partner history after display rejection, exact provisional replacement,
and retirement on both scene and conic failures. A further independently
forward-projected synthetic sequence crosses zero vertical gaze repeatedly;
both eyes stay within the specified four-degree test tolerance and paired
results match exactly for either eye's arrival order. That frozen v11 code
has 94 passing standalone tests. The SAM viewer builds with 983 passes, the
same 41 baseline failure names and 24 ignored tests.

All four v11 reports are complete: native versus v10 and v9, and each delayed
schedule versus v11 native, on the same 160k scope above. Delaying either eye
preserves paired availability and contribution flags for all 68,436 RAW pairs.
Each delay has one paired target change above `1e-6 mm`: maximum target changes
are 0.0000151/0.00000937 mm for right/left delay, with maximum same-eye ray
changes of 0.0000000345/0.000000214 degrees. This is strong numerical agreement,
not literal bitwise identity or a claim of such physical gaze precision.
Singleton schedules still have differing available past information; their
geometry is reported separately, not claimed invariant.

Against v10 native, right/left admission each gains one and loses one. There
are 338,861 matching probe fingerprints. Improvements over one pixel number
10/8 and regressions 5/6. The 2,399/1,205 common coarse-prior-normalized area
steps retain their medians; p95 right steps change 0.19652→0.19608 and maximum
left steps 1.41425→1.40126. This remains conditional held-scale support, not
anatomical constancy. Native tracker time is 2.37 ms median / 7.83 ms p95 /
16.52 ms maximum on the shared host, excluding SAM and RAW preparation.
The documented 70259 regression does **not** recover: retaining useful
singleton history is necessary but the newest start is not always the useful
one. New regressions include 46299 and 44083; smaller aggregate residual
changes cannot conceal those frames.

The separate 76-exposure ordinary label-neighborhood replay is complete. All
ten matched label outcomes and all 13 supported RAW-motion/SN-FEIDA links are
unchanged from v10, including its eight accepted/two missing labels, persistent
105191 localization error and missing fresh-boundary motion link. This small
unchanged subset is not evidence that the 160k regressions have recovered.

### Two historical starts, not an average

`eval-two-starts-v12` is testing the two most recent distinct successful read
initializations. For 70259, the newest prior read is a right-only observation
79.17 ms earlier; the useful v9 start came from the preceding left-only read
102.70 ms earlier. Both can be offered to the same current segment objective
without averaging their targets or keeping either as a temporal measurement.
The 16-start/12-refinement limits stay fixed, so these starts can displace other
starts and must be evaluated for regressions, not presumed beneficial.

Tests verify distinct targets remain distinct starts, residuals are exactly
unchanged by their presence, duplicate/nonfinite starts do not consume slots,
the source-age bound applies independently to the older read, and missing
partner pixels are not revived. All 96 standalone tests pass; the SAM viewer
test suite has 985 passes and the same 41 baseline failures. The matched
native/delayed 160k runs have now completed. The paired targets stay within
0.000134 mm across the tested 100 ms single-eye scheduling delays, but the
native optimizer comparison still has both improvements and large withheld
localization regressions. This does not validate the full 387,519-exposure
corpus or remove the boundary-selection failures described below.

The video worker now publishes fresh rejected/empty proposal packets even
when no complete single-eye ellipse exists, retaining exact source RAW,
sequence, timestamp, ROI and prompt/stream generations. It does not condition
SAM memory or admit an ellipse from that failure, and unrelated prompts are
rejected. This prevents an older proposal from concealing a new missing-eye
observation; it does **not** enable the unvalidated partial-outline fallback.

Native RAW inspection of pair [7358,7359] exposes a serious failure: a few
compatible upper-eyelid arcs can leave a fitted ellipse above the iris, despite
other groups being rejected. New weighting preserves the good right-eye limbus
in that pair but does not establish a correct left-eye localization. Native
inspection of 6185, 196553 and 6531 confirms eyelid/skin or clipped-eye detections
whose small surviving sections still receive influence. Pair [5460,5461] also contains rejected upstream
fits and large contour disagreements. These are not successful localizations
merely because an accepted subset has a small residual.

Still required: stronger boundary identity and ambiguity/fidelity accounting;
useful partial evidence when a full upstream conic cannot be fitted; broader
human-label and independent source-motion/scale/SN-FEIDA checks, including actual
ROI reframes; inspection
of all completed corpus results and regressions; and live/source-replay checks
of every presentation/calibration consumer. Full implementation validation is
not complete until those checks are actually run and inspected.

Reproduce with the repository runtime environment:

```text
viewer --offline-stereo-sam-export frames.jsonl cache.jsonl START COUNT
buttercup_stereo_conic_eval output.jsonl cache.jsonl...
report-stereo-conics.py output.jsonl report.json --expected-manifest manifest.json
score-stereo-labels.py inventory.json frames.jsonl labels.json output.jsonl...
python3 scripts/report-stereo-motion.py baseline.jsonl candidate.jsonl motion.json SCALE_REPORT.json...
buttercup_stereo_conic_eval source.jsonl cache.jsonl... --source-order-replay --arrival-delay-ns 2 100000000
report-stereo-conics.py source.jsonl source-report.json --source-order-replay --expected-manifest manifest.json
report-stereo-conics.py delayed.jsonl arrival-comparison.json --source-order-replay --baseline-evaluation source.jsonl --expected-manifest manifest.json
report-stereo-conics.py candidate.jsonl comparison.json --baseline-evaluation baseline.jsonl --allow-extractor-changes
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test-stereo-conics-report.py
buttercup_raw10_preview --source-index frames.jsonl INDEX comparison.png output.jsonl
```

`--max-frames-per-cache N` permits bounded prefix comparisons of caches whose
writers have already completed that many records. It never counts the rest of
the corpus as evaluated. The coverage checker rejects missing, duplicate and
out-of-range source indices. RAW comparison panels are custom-decoded native
RAW / magenta SAM / green joint / yellow monocular; no ImageMagick is involved.

## Last-calibration acquisition regression (2026-09-07)

Scope: `sam31-mouse-3d-1788821385-696201968.tar`, approximately 20 seconds,
318 native RAW10 exposures (198 right, 120 left), 120 exact same-clock pairs
and 78 singletons. The original session ended with `SURFACE DIRECTION NOT
ACQUIRED`, before stationary target one. No target coordinates, recorded gaze
predictions or human labels were used as detector/solver inputs. This clip has
no human localization labels and cannot establish monitor-calibration accuracy.
Artifacts are under `outputs/latest-calibration-joint.NY4ncJ`.

Four live-path problems were corrected:

- Joint SAM now requests iris plus pupil evidence independently of the selected
  rough-center provider. The pupil still has its independent RAW admission.
- Every source-attested SAM completion reaches the joint bridge immediately;
  the next display callback no longer decides which proposals get consumed.
- Dual-eye pending input is replaced atomically by source read. Once either
  worker claims a pair, its partner cannot be replaced independently. Input
  clocks are attested through the source ledger, not matched by ROI sequence.
  An incomplete nominal stereo read is dropped, not paired with old pixels;
  an actually evicted ROI still permits immediate single-eye processing.
- Calibration's safe-frame check uses the joint source/presentation ellipses,
  not the older SAM review ellipse. Missing/provisional UI updates do not erase
  confirmation votes. Four distinct signed sources and the 300 ms span remain
  required. The maximum confirmation gap is now 1.5 seconds, allowing one
  missed approximately 500 ms stereo confirmation. Source freshness remains
  900 ms; held output cannot add a vote or refresh the confirmation deadline.
  Tracking-generation changes and long gaps still clear progress.

Matched completion-paced video-worker replays exercised all 318 RAW exposures,
with independent eye/video state starting at the clip boundary. Outer-only
versus iris-plus-pupil produced 35 versus 69 pupil fits; both retained 294 outer
review fits. Through the corrected bridge and acquisition gate, outer-only
acquired after 8.51 seconds of source footage and the combined request after
1.72 seconds. These are availability-isolation tests, not inference latency.

Two additional runs offered the clip at its native camera cadence through the
actual asynchronous workers and atomic pair mailboxes. Blank-frame warm-up
used a separate tracking epoch; no future eye frames were used to warm memory.
Every completion retains actual ready time and the latest received ROI crop
metadata, then passes through the production bridge and acquisition code:

| Native-cadence run | Completed exact pairs | Dropped waiting/busy pairs | Ready latency median / p95 | Acquisition |
| --- | ---: | ---: | ---: | ---: |
| First | 87 | 33 | 492 / 692 ms | 4.11 s |
| Repeat | 88 | 32 | 469 / 663 ms | 3.92 s |

All 78 incomplete nominal stereo reads are accounted for separately, not
counted as detector rejections or successful pairs. Both timing runs briefly
paused our two background GPU corpus exporters, then automatically resumed
them. Their resource leases remained held: soft exclusivity was **not honored**.
A competing-load run completed only 48 pairs, with 754 / 1019 ms median / p95
latency, and failed acquisition with the earlier 750 ms confirmation gap.
Do not quote the uncontended timings as performance under that background load.

The matched source-order area check uses the recorded coarse/held independent
scale hints, not each candidate's own fitted radius. Fresh contributing outer
observations were unchanged: right 185, left 108. SN-FEIDA absolute log-area
step p95 worsened on the right (0.0941 to 0.1272, 179 steps) and improved on the
left (0.1726 to 0.1560, 101 steps). Missing fits break continuity; held results
are not counted as new measurements. This is a mixed area result, not a blanket
geometry improvement. Pupil extraction changes the held-out boundary probes,
so an optimizer-only localization A/B is invalid; the report tool correctly
refuses it. Native RAW inspection of exposures 28/29 was qualitative only.

Remaining limits: global-thumbnail motion arbitration is not replayed here;
these tests cover the real SAM workers, bridge and acquisition state machine,
not the complete camera/UI loop. The inherited uncalibrated intrinsics, coarse
scale, heuristic sign margin and boundary-selection failures remain defeasible.
The original failed clip has no stationary target sequence to validate the final
monitor fit. A user-led live calibration is still required.

Validation: 992 viewer tests pass, with the identical 41 pre-existing failure
names and 25 ignored tests. New source pairing, geometry authority and bounded
acquisition tests pass; the ignored recorded-evidence test was explicitly run
and passed on both native-cadence caches. Source-tree and diff checks pass.
No commit or push was made for this work.

Reproduce the two replay modes using `BUTTERCUP_STEREO_LIVE_REPLAY=combined`
or `offered-combined` with `--offline-stereo-sam-export`. The ignored test
`recorded_calibration_acquires_from_source_timed_live_worker_evidence` consumes
`BUTTERCUP_JOINT_CALIBRATION_CACHE`, writes a new
`BUTTERCUP_JOINT_CALIBRATION_REPORT`, and asserts acquisition when
`BUTTERCUP_JOINT_CALIBRATION_REQUIRE_READY=1`.

### Live follow-up: acquisition was being re-entered during target collection

The subsequent live session exposed a missing outer-state-machine test.
`sam31-mouse-3d-1788825599-460642422` had 16 acquisition episodes and 15 sign
restarts without an authority change. The later `1788825748-313200645` session
had 27 episodes/26 restarts, also without an authority change. The initial
replay above exercised the acquisition helper, but did not call the enclosing
`VirtualMouseMode` through its later stationary-target updates.

That enclosing mode incorrectly treated any unsigned SAM update as grounds to
clear an already acquired basis and all collected targets. Acquisition now
starts only when no calibration sign epoch has been established. Unsigned,
missing or provisional observations pause collection; genuine source-generation
or signed-epoch changes still discard incompatible partial samples.

The regression test failed before the fix and passes afterward. The recorded
native-cadence cache is now also replayed through the actual enclosing phase
machine: 176 completions retain one acquisition episode, zero sign restarts,
and advance to target index 1. This remains a state-transition check, not a
claim of valid monitor calibration from the original moving-stimulus footage.
Artifacts: `outputs/calibration-phase-fix.OEW4Ur`.

The live viewer separately exited when a tracking packet exceeded its 350 ms
queue-age guard by 5 ms. The next user-led run uses the existing session-only
`BUTTERCUP_CALIBRATION_CAPTURE=1` overrun behavior: discard those stale tracking
packets instead of exiting. This does not admit stale gaze, change camera
settings, or disable the other source/transport integrity checks.

### Further calibration failures: eye binding, joint cache lineage, and gaze jumps

The subsequent RAW/OIM1 recordings distinguish three different failures:

- `1788827060-509774734`: after collecting the first target, automatic viewer
  focus switched from ROI 2 / authority 135 to ROI 1 / authority 142. This
  aborted calibration as a provider change. The calibration now keeps the eye
  of its first admitted sample for collection, source clocks, thumbnail and
  scene reference, even when automatic viewer focus changes. An absent bound
  eye pauses collection; it cannot be silently replaced by the other eye.
- `1788826811-109153184`: ROI 2 authority stayed 84 while joint cache generations
  15, 16 and 17 restarted calibration twice. A sibling ROI's readmission can
  clear the joint optimizer's history, but that counter is not a correction to
  the selected eye's sign coordinates. Joint surface samples now expose the
  selected eye's authority lineage as their calibration epoch. The independent
  joint evidence generation remains in the recorded diagnostic publication.
  This does **not** prove branch continuity or promote unsigned hypotheses.
- `1788826709-274018364`: all nine targets completed, without restarts, but
  neither a 2D affine nor a metric display plane fit the recorded centroids.
  Replaying those centroids with the recorded EDID dimensions reproduces this
  rejection. The middle top/bottom pair has the opposite vertical response
  from the two corner pairs. The later `1788827178-325761427` also completed
  all nine targets but mixed positive and strongly negative vertical directions.
  These are underlying gaze failures, not evidence to loosen the monitor gate.

Terminal failed sessions no longer reset their in-memory target sequence on a
later source-generation change after the RAW writer has already stopped.
Four added regression tests cover binding, terminal state, cache-generation
semantics and collection-through-final-fit rejection. The live-feature suite
has 997 passing tests, the same 41 named pre-existing failures, and 25 ignored;
the viewer builds. No geometry extractor/objective was changed by these state
fixes, and no claim of improved SN-FEIDA or localization is made for them.
Artifacts: `outputs/calibration-provider-fit.cVlx1Z`.

Importantly, the final live session `1788827261-537156697` **was accepted**, with
all nine targets, no authority/sign restarts and a completed RAW archive. This
provides a successful comparison session, not gaze ground truth or independent
verification of the fitted monitor's physical pose. The user then closed the
viewer and is unavailable for live guidance for approximately twelve hours.

The independent follow-up uses all 1,337 byte-verified RAW frames across the two
rejected nine-target sessions and this accepted session: 573 exact same-read
pairs plus 191 singleton reads, no missing payloads. Fresh completion-paced
SAM video-worker export starts empty at each capture lineage. It preserves
native source clocks, crops and coarse scale support; it does not replay the
live pre-capture video memory or independently measured thumbnail motion.
There are no human limbus labels or independently measured gaze positions for
this subset. Monitor fit and sign stability must not be mistaken for accuracy.

### Off-axis target coordinates and reflection-censored pupil arcs

The independent RAW replay found two geometry/photometry defects beyond the
calibration phase-machine problems:

1. The old joint target bounds were camera-optical-axis `X/Z`, `Y/Z` slopes,
   although the eye can be far from that axis. A valid, convex, camera-facing
   solution then hit the arbitrary 1.5 slope limit and competed unfairly with
   its mirrored orientation. In `1788827178-325761427`, 476 of 582 measured
   outer-circle negative-normal hypotheses exceed that old global limit;
   none of the 260 hypotheses in the successful comparison session do.
   The first correction uses the tangent plane about the scene reference's
   direction toward the camera. That intermediate version retained optical-Z
   depth; the complete-corpus range audit below exposed why the depth axis
   also needs correction. Individual camera-facing, positive-distance, radius,
   nesting and IPD checks remain. This chart limit is an engineering envelope,
   not a measured head orientation or anatomical confidence interval.
2. The SAM pupil fitter already censored specular light using a robust iris
   annulus brightness ceiling. The joint adapter discarded that policy and
   searched new pupil arcs with only a near-saturation ceiling of 990 RAW10.
   Unsaturated screen reflections could therefore re-enter as pupil evidence.
   Both stages now share the existing annulus/MAD policy. Native pupil
   profiles touching brighter pixels, or with a censored neighbor that
   prevents locating an actual maximum, stay absent. Missing iris photometry
   does not authorize a substitute pupil perimeter. The outer SAM runs are
   unchanged; neither a display contrast setting nor a completed ellipse is
   converted into observed edge points.

Also, a pupil arc's measured edge width no longer changes the full ROI's
optical reliability and silently reweights already appended limbus arcs.
Its width remains in its own normal band; independent full-ROI focus evidence,
when supplied, is preserved. Unknown focus remains unknown.

The off-axis synthetic test independently forward-projects the source points
at positive and negative vertical offsets. The old coordinates select a
normal about 77 degrees wrong in one case. The corrected solve passes both
offsets, with both eyes and either eye alone. Additional tests cover chart
round trips, invalid/backward targets, missing photometry, unsaturated glints,
retained visible pupil support and RAW exposure scaling of the tissue ceiling.

With the coordinate correction held fixed, adding reflection censoring changes
large adjacent-source orientation jumps (>30 degrees, <=500 ms gaps) from
68 to 44 per eye in the difficult session, and from 6 to 0 per eye in the
successful session. The first failed session is unchanged (13 right-eye and
0 left-eye jumps). These are continuity diagnostics, not known wrong-sign
counts: the recordings have no independent gaze labels. The 1,337-exposure
comparison retains identical coverage. Its fixed outer-limbus probe sets have
no >1 px regression; changed pupil probe sets are explicitly not compared as
if they were identical measurements.

This does **not** establish corpus-wide robustness. The 160,000-exposure,
91,564-physical-read comparison covers 137 capture entries and 175 clock
lineages, not the full 387,519-frame inventory. Compared with the old solver,
the optical/coordinate changes reduce pooled withheld p95 but worsen the
common-accepted probe p95 and include large individual regressions. Native
inspection of source 16688 confirms a downward-shifted ellipse despite a
clearly visible iris; source 60577 has very weak/blurred support and poor
extrapolation in both versions. There are no human labels for those frames.

The subsequent **glare-only** comparison on that same 160k scope gains one
left-eye admission and loses none. Fixed outer-probe p95 is nearly unchanged
(right 2.996 -> 2.991 px, left 3.850 -> 3.848 px), but 28 eye-frames regress
by >1 px and 53 improve; the worst right-eye change is about +34 px. Thus
an improved aggregate or the successful calibration clip is not a substitute
for inspecting the outliers. Independent-scale SN-FEIDA step p95 is unchanged
in this glare-only comparison, with only 2,399 right / 1,205 left supported
adjacent steps; most of the wide corpus has no independent scale support.

Ten native human-label matches are available in a separate 76-exposure replay:
eight accepted fits and two unavailable in both versions. Glare censoring
changes one visible-label RMS from 3.455 to 4.030 px and leaves the other
matched scores unchanged, including an existing roughly 37 px bad fit. These
labels are localization evidence only, never optimizer inputs or gaze truth.

Artifacts remain under `outputs/calibration-provider-fit.cVlx1Z`: frozen
baseline/candidate evaluators, `viewpoint-*`, `glint-*`, exact-source native
outlier previews, and explicit `legacy-chart-glint-*` coordinate-only ablation
outputs. The legacy-chart binary is an offline control, not a live option;
its slope diagnostics are legacy optical-axis coordinates. The production
source was restored byte-for-byte after building that control.

### Fixed target-window validation and monitor consensus refinement

`calibration_replay_joint_target_windows` scores already-solved gaze against
the recorded target windows with the actual Rust fixation cluster reducer,
coverage checks, robust affine and metric monitor fitters. It evaluates both
eyes and both the complete recorded sample window and an additional 500 ms
trim. Only the final publication of each native sensor read is counted.
Target locations never enter conic fitting. This is **not** a closed-loop
replay of live admission, detector latency, or changed target presentation.
The earlier Python median/least-squares sweep is only an approximate
diagnostic; this Rust test is authoritative for the application's fit checks.

Before the coordinate/photometry changes, the fixed-window test fails shared
monitor support for the selected eye in each failed session, while preserving
the successful session. With the changes, it passes the selected eye in all
three sessions without relaxing any gates. The unselected left eye of the
first failed session still fails. A separate live-worker offered-cadence cache
passes the actual bridge/acquisition/phase replay: 174 completions, first
acquisition at 4,110 ms, one acquisition episode, no sign restart. That proves
the exercised phase transition, not final live accuracy.

An existing synthetic calibration test exposed a further estimation defect:
Huber's linear tail kept pulling the metric monitor even after its consensus
rejected a sign-flipped target. The eight good targets then had about 4.94%
screen RMS. The fitter now performs at most three consensus-only refinements,
re-scoring all original observations each time and refusing lost coverage or
a worse equal-size consensus. Rejected targets no longer exert a refinement
force. Width/height, coverage, conditioning and acceptance thresholds are
unchanged. The existing test improves to roughly 0.02% inlier RMS; additional
tests exercise clean and one-outlier rotated rectangles, including reversed
screen horizontal axes. No monitor preset is automatically overwritten.

Source-native motion experiments additionally retain at most 80 diagnostic
patch correspondences per interval, with strict from/to ROI clocks and an
optional exterior-of-iris mask. Missing motion is not an identity transform.
Default tracker numerical outputs match the earlier 1,337-frame export
exactly; correspondence storage and iris exclusion are opt-in. Causal
projected-pivot hypothesis-ranking trials reduce some remaining direction
jumps, but stronger weights regress other clips and a whole-ROI transform
is not independently measured head motion. Those temporal ranking trials
remain offline and are not enabled as live sign authority.

### Early acceptance, optional contour directions, and search-budget controls

The fixed-window success above is **not sufficient to predict an early live
success**. An additional source-causal prefix diagnostic uses the first stable
recent cluster after the 1.5 s source-time hold, separately from the final
window's cluster. With a 500 ms settling trim, the corrected solver passes
both eyes in the second and third recordings, but fails both eyes in the
first. That first recording's selected right eye has roughly 3.1--6.9 degree
early-cluster RMS and several 3--4.4 degree differences from its final cluster.
Its final-window monitor success used only seven inliers, not nine accurate
targets. The current 12 degree cluster radius / 7.5 degree RMS threshold can
accept signal this noisy. Tightening it would withhold those samples, not
demonstrate that the iris fit or focus was repaired; no threshold was changed.

These prefix results use actual Rust cluster/monitor functions and no future
samples for prefix selection. They still lack attested presentation/scanout,
live detector completion, and admission timing for the new predictions.
The artifacts are `prefix-{baseline,glint,hypothesis24}-monitor-fit-*.log`.
The offered-cadence bridge harness now honors `selected_query`, rather than
assuming the first semantic candidate was selected, and can explicitly test
either physical eye with `BUTTERCUP_JOINT_CALIBRATION_EYE=0|1`. Neither harness
can turn an unselected eye's acquisition into evidence for the selected eye.
The 174-completion offered cache passes separately for eye 0 (4,110 ms) and
eye 1 (3,974 ms), with one acquisition episode and no UI sign restart in each.
Its selected query is already the first query, so correcting mask selection
does not explain the old real-world failure in this particular cache.

An opt-in `--retained-outline-directions` evaluator experiment derives tangent
directions only from neighbors in the same retained SAM run, confirms the
outward polarity with current RAW, and never borrows a normal from a fitted
ellipse or bridges an occluded gap. Position and direction share one capped
arc information budget. Flat/saturated/missing RAW abstains. The default live
adapter remains positional; a 1,337-exposure numerical control, including all
returned current hypotheses, is identical with the experiment disabled.

The direction experiment does not improve the recent calibration jumps. On
160k exposures it has 97/162 right/left fixed-outer-probe improvements over one
pixel but 102/167 regressions, with worst changes of about +67/+40 px.
Common-accepted outer probes improve more often than they regress (76/116
versus 44/56), but that cannot hide the lost/poorly extrapolated support.
The right SN-FEIDA step p95 improves from 0.1863 to 0.1829 while the left worsens
from 0.2215 to 0.2241 on the same limited independent-scale subset. Human-label
coverage is unchanged (eight accepted/two missing); one RMS changes from
9.799 to 9.792 px and the others, including the 37 px failure, are unchanged.
Consequently this experiment remains **disabled in the viewer**.

A separate coordinate-only control, with glare policy and optical weighting
held identical, gains 13 and loses 21 eye admissions on the 160k scope. Pooled
withheld p95 improves, but individual local-search regressions remain. Inspect
the **final same-read publication**, not just the row carrying the first ROI:
for pair `[60577,60578]` the right-only provisional result is almost unchanged,
whereas the final pair gives the large regression. Pair `[21425,21426]` also
has a worse final objective, showing that a better feasible basin was missed;
this is not just a changed confidence score. Offline 16-refinement and
24-start/16-refinement controls test bounded search effort separately. The
live default remains 16 starts and 12 refinements; more search is not itself
evidence of improved localization or direction.
The completed 16-refinement-only control preserves admission, improves 35
fixed outer-probe eye-frames by >1 px and regresses 37; its worst regression
is +15.34 px and both SN-FEIDA step p95 values worsen slightly. The 24-start,
16-refinement control gains 29 eye admissions and loses none, but still has
228 fixed outer-probe regressions versus 273 improvements and worsens both
normalized-area step p95 values (0.1863 -> 0.1961, 0.2215 -> 0.2287). Its worst
fixed outer-probe regression is about +71 px. Neither trial justifies a live
budget change on its own. Timings were shared-host diagnostics, not an
isolated real-time capacity test.
A new monocular regression independently projects outer/pupil 3D points
through camera-relative meridian crossings and turnarounds, with small 3D
head translations and native ROI reframes. Both physical eyes pass separately
at the live search budget. Its observer ray is off-axis: a camera-global
vertical zero is not substituted for the actual circle-pose degeneracy.
This is clean synthetic geometry, not a claim that blurred pupil boundaries
have become observable in the real recordings.

Exterior-motion diagnostic v3 masks excluded iris patches **before** selecting
the strongest corner in each budget cell, and also respects that mask during
subpixel refinement and backward checking. Otherwise an excluded iris corner
starved the usable exterior corner in its cell. This improves exterior-link
availability, but those links are not pure 3D head-motion measurements. All
1,337 default whole-ROI results and retained diagnostic correspondences are
bit-identical to the prior export, excluding elapsed time. Exterior-only
ranking still trades improvements in one recording for regressions in another
and remains offline.

### Low-light labeled replay and representation limits

Keep three inference regimes distinct. The wide historical caches
`sam-pilot.jsonl`, `sam-all-a.jsonl`, and `sam-all-b.jsonl` are **stateless
single-frame semantic outline exports**, including unselected/rejected mask
candidates for component evaluation. They are not a measurement of live video
memory or its offered-frame scheduling. The three recent calibration clips
use the actual video workers, completion-paced from an empty capture lineage.
The separate offered-cadence cache tests real completion/presentation age for
one recorded schedule. A result from one regime is not silently evidence for
another.

Replaying the same 76 human-label-neighborhood exposures through the current
combined-query video worker retains eight accepted and two absent human-label
matches. The largest labeled error is still 29.76 px (versus 37.39 px in the
stateless semantic-export component evaluation). These are the same native
images and post-fit labels, but different inference evidence and temporal
contexts, not a matched conic-only algorithm improvement.

Existing preprocessing options were retested with that video-worker baseline.
`strong-low-pass` improves the 29.76 px case to 11.18 px, preserves the two
dropouts, and regresses three other accepted labels by more than one pixel.
`gentle-shadow-lift` recovers the two missing labels at 8.62/5.30 px but worsens
the largest error to 39.37 px and another from 9.34 to 15.51 px. Neither is a
universal repair; the live mild-blur default is unchanged. Likewise, the
current partial-outline experiment worsens the stateless 37.39 px case to
57.58 px; its two recovered labels remain poor at 20.34/14.28 px. It remains
disabled.

Byte-verified native intensity statistics support treating the first failed
calibration as a different optical condition, not simply another sign epoch.
Its median 4x4 CFA-cell intensity is about 164/179 RAW10 units for right/left,
versus 809/655 in the second and 910/856 in the successful recording. Median
near-upper-rail site fractions (`RAW >= 990`) are 0/0%, 35/21%, and 49/40%.
Whole-ROI composition and reflections affect these statistics: they do not
measure iris illumination, lens focus, camera exposure settings or causality.
In particular, the brighter successful recording does not justify saturating
the iris or automatically increasing camera exposure. Native bytes remain
untouched; the diagnostic only decodes RAW10_LE40_1X1 and summarizes samples.

The longer production-worker trial also rejects a blanket stronger-blur
default: `strong-low-pass` raises the second recording's >30 degree source
steps from 44 to 53 per eye. Its fixed-window monitor result is unchanged,
including the first recording's failed early selected-eye fit. The opt-in
`adaptive-denoise` experiment uses spatial local-variance shrinkage and a robust
mixed-difference noise proxy, not a calibrated sensor-noise model. It leaves
native evidence intact, has no temporal image history, and retains `mild-blur`
as default. Its Gaussian first/second moments use f64 to avoid brightness-
dependent cancellation in `E[x²] - E[x]²`; constant-input, noise/step-edge,
and linear-exposure-equivariance tests pass.

Adaptive denoising improves labeled seq-308 from 9.34 to 4.08 px, but loses
seq-317 entirely and regresses seq-247 from 3.71 to 5.79 px. It recovers one
previously missing label at 10.85 px: overall coverage is still eight of ten,
not ten improved fits. Across the recent recordings it raises >30 degree
steps from 13/0, 44/44, 0/0 to 18/2, 52/52, 2/2 (right/left). The first
recording's early selected-right-eye monitor fit still fails; an early left-
eye fit passes, but its final-window fit fails. This does not authorize an
eye switch during calibration. The experiment is not enabled by default.

### Angular continuity is not sign evidence by itself

The runtime `analyze-angular-kinematics.py` experiment uses a bounded forward
beam (12 distinct histories, three source observations per eye, 750 ms maximum
gap), observer-relative surface normals, and source-timed angular velocity or
acceleration prediction. It only chooses an already optimized **current**
camera-facing conic system; it never holds/blends a gaze output, reads future
frames, or supplies calibration targets to geometry. Its scales and decaying
costs are engineering choices, not measured physiological distributions.

On the recent default-input videos, modest history weight reduces the second
recording's 44 jumps to two, and higher weight removes them. That visually
appealing result is **not safe to enable**. A new ignored Rust diagnostic
forward-projects 864 monocular observations: both eyes separately, crossings,
turnarounds and abrupt steps, 40/100/300 ms source intervals, four pupil-
boundary uncertainty bands, small 3D head translation and native crop moves.
Truth is emitted only after fitting. These are exact geometric samples with
uncertainty bands, not simulated optical blur or noisy real camera images.

The unranked current solver has no >4 degree errors on that constructed set.
The angular-velocity trial at weight 0.25 introduces 18 crossing errors,
36 turnaround errors and 71 step errors; some wrong-branch errors are about
47 degrees. Adding angular acceleration does not cure this. Even a genuine
near-meridian turnaround can be mistaken for continuation through the
meridian. Lower jump counts cannot justify a prior that overrules the current
image evidence or delays real saccades. This ranker remains runtime-only;
no angular smoothing/sign lock has been wired into the viewer.

Artifacts: `angular-kinematics-grid-v2.json`,
`synthetic-meridian-uncertainty.log`, and
`synthetic-angular-kinematics-scored.json` under
`outputs/calibration-provider-fit.cVlx1Z`. The ordinary clean moving-meridian
regression remains a passing unit test. Independently supported scene/pivot
transport still needs separate validation; missing exterior motion must never
be treated as a stationary head or proof of one sign.

### Complete semantic-export receipt audit

All 387,519 surviving exposures in `replay-inputs-v2` are now joined to the
completed pilot/a/b caches without missing, extra, reordered or substituted
images. They cover 186 capture entries and 224 clock lineages; 50,222 receipts
have attested clocks. This is not all historical capture entries: the original
inventory separately identifies missing RAW and duplicate receipts. There
are 250,604 exposures without a selected semantic query; rejected alternatives
remain component diagnostics, not live accepted coverage.

The audit records 1,163 scale-metadata serialization differences, each at most
one representable f64 step, in `pixels_per_10mm` or its lower bound. For example,
107.33479960362239 became 107.3347996036224. All image hashes, native dimensions,
sensor origins, integer source times and other receipt fields must match
exactly. The immutable inputs/caches were not rewritten to hide the rounding.
`full-sam-cache-audit.json` records counts, examples, byte lengths and complete
cache SHA256 hashes. This receipt check establishes provenance, not accuracy,
production video-memory behavior, or real-time throughput.

The completed baseline/intermediate comparison contains 220,893 physical reads
(166,626 paired, 54,267 singletons), with no missing/unexpected source indices.
It gains 18/37 right/left admissions and loses 30/30. All-withheld RMS p95
improves from 4.73/6.77 to 4.28/6.01 px, but the unchanged common-accepted
outer-probe p95 changes from 2.925/3.796 to 2.952/3.784 px. The latter is a
small right-eye regression, not universal improvement. There are 707/901
fixed-outer >1 px improvements and 700/718 regressions; the worst increase is
117.45 px on source 60577, with additional >60 px failures outside the earlier
160k subset. Human-label coverage is still eight of ten exact native matches;
seq-317 worsens from 35.92 to 37.39 px and seq-222 from 3.49 to 4.03 px.

SN-FEIDA adjacent-source log-step p95 improves from 0.1965/0.2368 to
0.1863/0.2215, but still uses only 2,399/1,205 matched independent-scale steps.
The expanded corpus adds no scale support to that subset. Most captures lack
independent scale and all lack independent gaze-angle truth. These are the
**intermediate optical-Z-depth** results in `full-matched-report.json`, not
validation of the subsequent axial-distance correction.

### The depth prior must share the viewpoint chart's axis

Full-corpus inspection exposed an additional defect in that intermediate
coordinate conversion. Multiplying a viewpoint ray by `optical_z / ray.z`
turns a finite 150--2,000 mm optical-Z prior into an effectively unbounded
Euclidean target range near the optical horizon. The intermediate replay has
186 publication-eye results beyond 5 m, 74 beyond 10 m, and five beyond 1 km;
one provisional source-100550 result is approximately 486 km away. These are
publication counts, not independent physical reads. The optical-axis baseline
has no such >5 m results. Camera-facing normals alone do not prevent this
coordinate singularity.

The candidate correction uses two tangent slopes and **log axial distance on
the same reference-to-camera axis**. `fixation_axial_distance_mm` explicitly
replaces the misleading optical-Z field; this changes the interpretation of
the broad off-axis depth prior, not merely its spelling. It remains an
engineering working-volume prior, not measured accommodation or an anatomical
limit. For axial distance `d` and slopes `u,v`, the reference-to-target range is
`d * sqrt(1 + u² + v²)`: finite configured bounds now bound the metric range.
The conversion no longer divides by the ray's optical-Z component. Exact
on-axis coordinates retain their prior meaning. No SN-FEIDA scale, iris-radius
bound, monitor preset, or image evidence is inferred from this distance prior.

Tests cover the near-horizon counterexample, round-trip coordinates and the
constructed off-axis positive/negative signs. The latter fixture now declares
the actual ~1 m axial distance of its ~1.28 m target, rather than an optical-Z
distance of 600 mm. The ordinary moving-monocular-meridian test still passes.
The debug/evaluator chart now exports the reference, axial distance and named
axis; the independent Python reporter reconstructs the target and rejects a
mixed-axis or incomplete declaration. Corpus and recent-video comparisons
are inspected separately below.

The completed matched replay contains 387,519 surviving exposures, 220,893
physical reads (166,626 pairs), 186 capture entries and 224 clock lineages.
There are no missing or unexpected indices against the surviving manifest.
It still uses stateless semantic exports, not the production video-memory
path, and does not restore the missing historical RAW listed in the inventory.
The independent range audit includes provisional **and** final publications:
all 169,352 available candidate charts reconstruct correctly and obey the
configured axial/ray bounds. Its 221,554 modeled publication-eye ranges have
no >5 m results; the maximum is 2,866 mm (maximum reference range 2,837 mm).
These are consistency bounds, not measured fixation distance or accommodation.

Compared with the frozen original baseline, final-read admission gains are
17 right / 41 left, with 30 / 42 losses. On matched unchanged outer-limbus
probes, p95 residual is 3.0805 -> 3.0808 px right and 4.9399 -> 4.7598 px left;
there are 760 / 1,022 improvements over 1 px and 752 / 806 regressions.
Restricting to probes accepted by both algorithms gives 2.9253 -> 2.9526 px
right (worse) and 3.7945 -> 3.7729 px left. The broader unchanged all-boundary
intersection improves p95 from 4.7336 -> 4.2751 and 6.7554 -> 5.9517 px, but
is not a replacement for the outer-boundary and human-label comparisons.

SN-FEIDA still has only 2,399 right / 1,205 left independently scale-supported
adjacent steps. Its p95 absolute log step is 0.19652 -> 0.19561 and
0.23679 -> 0.23461 against the original baseline. This is **worse** than the
intermediate mixed-axis version's 0.18630 / 0.22147; constant area cannot
justify retaining a singular coordinate map. Coarse MediaPipe scale is held
between reacquisitions under the documented size prior, not freshly measured
at every exposure. No candidate radius supplies its own normalization.

The full-history human-label comparison has only ten exact native RAW matches:
eight accepted, two missing, as before. The difficult sequence-317 label
regresses from 35.924 to 37.401 px visible-boundary RMS; sequence 222 goes from
3.491 to 4.035 px. The remaining accepted labels change little. The missing
10124/10163 fits remain missing. These numbers must not be conflated with the
separate short-neighborhood production-worker preprocessing trial.

The worst new fixed-probe regressions were reviewed with the Rust native RAW
decoder, not a generated reconstruction. Source 200931 has an eye clipped at
the right edge; 125890 has a partial iris at the lower/left boundary and strong
glare; 173067 is mostly a blurred clipped lower strip; 329725 has no clear eye
and nevertheless receives an admitted ellipse. These are qualitative reviews,
not newly drawn human labels. Their +85.08, +69.85 (against the intermediate
version), +76.74 and +67.71 px regressions remain failures. Three have weak
sign margins and/or clipping, but the non-eye crop has a margin over six:
objective separation is not proof of an eye or correct boundary identity.

For the three recent production-worker clips, final-read >30 degree jumps are
13/0, 24/24 and 0/0 (right/left), compared with 13/0, 44/44 and 0/0 before the
axial correction. Actual Rust monitor/cluster replay still fails the darkest
clip's source-causal prefix and passes both eyes of the other two. All 864
constructed single-eye meridian/turnaround/step observations remain within
4 degrees of their known targets; maximum error is about 1.2e-6 degrees for
these noiseless points with declared uncertainty bands. This does not validate
sign under real blur, occlusion, reflection, or anatomical-model error.

Artifacts in `outputs/calibration-provider-fit.cVlx1Z`:
`full-axial-vs-{baseline,intermediate}-report.json`,
`full-target-range-audit.json`, `full-axial-depth-human-labels.json`,
`axial-depth-target-summary.json` and `prefix-axial-depth-monitor-fit-*.log`.
CPU comparisons were shared-host runs, not isolated capacity benchmarks.

### Calibration must not consume a provisional paired-source vote

The offered-load replay exposed a separate consumer defect. SAM's atomic
two-ROI submission still produces two completion callbacks. Calibration used
the first usable gaze at a source timestamp and rejected the later paired
refinement as a duplicate. For eye 1 in the 174-completion recorded-cadence
cache, all six usable exposures first arrived as monocular publications;
paired refinements changed gaze by up to 1.279 degrees, 18--202 ms later.
Such a change is material to the small calibration grid even without a sign
reset. The physical exposure is still only one observation.

`ProposalMasks::source_group_roi_count` now preserves whether the exact SAM
request was single-ROI or an atomically submitted pair. It does not derive
from temporal history length or the UI's active-eye count. While collecting
calibration samples, an otherwise usable explicitly paired result waits for
both source completions with matching exposure time and clock. A completed
but unhelpful partner need not contribute geometry; an evicted or genuinely
single-ROI request does not wait. Missing and legacy metadata is explicit.
No source/sign/ROI validity threshold is relaxed, and the final pair casts
one vote rather than two. Diagnostics distinguish `WaitingForSourcePartner`
from an unresolved sign or missing surface.

Rendering and the completed absolute cursor do not acquire that wait. The
cursor may replace a provisional placement once with its same-source,
same-sign-epoch completed pair, without interpolation or incrementing the
fresh-exposure counter. Older sources, repeated completed results, late
provisional results and sign-epoch-only changes cannot use this exception to
rewind the cursor. Training provenance remains unchanged.

Matched controls through the actual bridge/acquisition/UI use identical cached
production results and recorded completion times. Both physical eyes acquire
at 4,110 ms with the paired rule; the old control acquires at 4,110 ms for eye
0 and 3,974 ms for eye 1. All six distinct usable exposures survive, with no
provisional monocular votes, one acquisition episode and zero sign restarts.
The cache predates the new field; only its attested
`live-video-worker-offered-load` atomic-ingress contract permits inferring a
pair from active-mask=3. Reports explicitly name this legacy inference; generic
two-image/completion-paced caches cannot make it. These results validate
acquisition timing, not final calibration accuracy or the user's eye position.

Tests cover single requests versus temporal history length, metadata surviving
RAW rejection, same-read completion, missing/evicted partners, clock mismatch,
duplicate votes and completed-cursor refinement provenance. The full viewer
suite has 1,022 passes, the same 40 named pre-existing failures, and 27 ignored
tests; the independent reporter has 25 passing tests. The suite is not fully
green. Artifacts: `paired-bridge-tests.log`, `paired-full-tests-v2.log` and
`paired-offered-eye-{0,1}{,-control}.jsonl` in the same work directory.

### The host queue was selectively discarding the second eye

A later OIM1/source audit establishes a separate upstream loss, not a camera
delivery failure. In the dark `1788826709-274018364` recording, 191 surviving
reads have only one archived ROI although native region metadata declares
active-mask=3. **Every missing partner has an exact matching host
`tracking-warmup-or-backlog` drop receipt.** The archive has 340 right and 155
left RAW frames; recorded eye-drop counts are four right versus 189 left.
The left drop-event minus socket-arrival age has median 218.57 ms and p95
246.62 ms, just beyond the 200 ms consumption limit. These dropped native
payloads are not present in the archive and cannot be recreated by replay.
"No missing indexed payloads" never meant that every camera packet was saved.

The original policy tested packet age separately. Beginning expensive work on
a right-eye packet close to the limit could make its same-source left packet
stale before reaching it. Downstream atomic SAM ingress cannot repair that:
it correctly refuses to manufacture a pair from another exposure. The two
brighter comparison clips have no eye-drop events or incomplete reads. They
also have approximately 201.5 ms native source periods, versus 96.8 ms in the
dark clip, so illumination is confounded with a substantially easier host
work rate. These records do not establish which camera setting caused the
rate change; brightness alone is not a demonstrated explanation.

`source_queue_budget.rs` now reserves anticipated first-ROI work **before**
admitting an already-aging pair. It retains only eight timing samples and one
pending source key, not images or gaze. A deliberately skipped head also
skips its matching tail. Read identity includes source time/sequence and
region session/generation; single-eye inputs remain independent. The existing
200 ms consume limit, 350 ms hard limit and separately bounded camera-control
policy still decide actual freshness. This budget can only shed extra work,
never override a rejection. Fresh starts within a bounded 20 ms window remain
eligible, and work samples expire after one second. Both are needed: recurring
context work may leave every packet outside the fresh-start window, so a
count-only timing history could otherwise prevent its own recovery forever.

This is anticipatory work admission, **not a guarantee of atomic processing
under arbitrary CPU stalls**: an unexpectedly slow first ROI may still make
its tail exceed the unchanged hard freshness rule. A sustained-load synthetic
control reproduces one-sided starvation; the new rule completes more pairs
without consuming any >200 ms packet. Tests also cover missing/mismatched
partners, independent ROIs, and recovery from an over-budget measurement.
Future drop receipts include separately measured `queue_age_ns` and the
`tracking-paired-source-headroom` reason. The old log implying skipped packets
remained in a complete RAW recording was corrected. Actual improved pair
retention still needs a new live recording: unavailable historical bytes cannot
prove it. The suite now has 1,026 passes, the same 40 old failures and 27 ignored.

The second ROI should not simply be ignored to avoid this scheduling problem.
Using identical recorded SAM results with only one eye's evidence increases
>30 degree jumps markedly:

| Clip | Stereo right / left | Each eye solved alone |
| --- | --- | --- |
| Dark failed calibration | 13 / 0 | 26 / 19 |
| Brighter failed calibration | 24 / 24 | 72 / 62 |
| Successful calibration | 0 / 0 | 8 / 13 |

Coverage in this ablation is unchanged. Both monocular versions fail final and
early-prefix monitor fits in the two failed clips; both pass in the successful
clip despite their additional jumps. This illustrates why either a fitting
score or a monitor pass alone is insufficient. Stereo remains off by default;
no live eye-selection policy was changed by the experiment.

New production-worker offered-load replays, one clip at a time, return 282,
494 and 222 proposals. For the dark clip, 191 unpaired archive frames cannot be
offered atomically; another 22 are shed by the worker. The other two have 88
and 38 worker drops. All returned proposals explicitly retain group-size=2.
Through the actual live bridge, all six per-eye acquisition-helper runs reach
Ready (1,423/1,423; 5,036/5,036; 1,544/1,317 ms), with no UI sign restarts.
Some UI runs begin with an already-qualified surface and need no moving
acquisition episode. There are respectively 126/129, 143/137 and 104/105
distinct qualified sources. Repeated Ready callbacks are not additional votes.

These are **new offline measured completion schedules**, not the original
session's detector timings or a closed-loop user test. The blank warmup does
not reproduce pre-capture eye memory; global thumbnail motion is unavailable.
Target collection under the new schedule does not establish that the recorded
person looked at newly simulated target times. The diagnostic monitor-fit
failure after simulated collection must not be mistaken for a matched-target
accuracy measurement. All geometry/window comparisons above remain separate.

Artifacts: `recent-source-drop-audit.json`, `source-queue-full-tests.log`,
`monocular-eye-*-target-summary.json`, `monocular-eye-*-monitor-fit-*.log`,
`recent-offered-*-sam.{jsonl,log}` and `recent-offered-*-eye-*-bridge.{jsonl,log}`.
The worker runs used announced shared CPU/GPU claims, not exclusive capacity
benchmarks. No camera command, live capture, autofocus, exposure, monitor
preset or default preprocessing was changed by these offline evaluations.

### Recording admission is independent of analysis admission

The live ingress path now handles active RAW writers and start/stop requests
before its analysis-age gate. Native sensor-band thumbnails likewise reach
their bounded archive/publisher path before context analysis can be skipped.
An old packet is still forbidden from updating SAM, focus, motion, presence or
calibration; saving its actual bytes is not permission to analyze it. This also
prevents an overloaded analysis loop from waiting indefinitely for an eligible
first ROI before acknowledging a recording request. Timed/hotkey complete-set
counts now describe recorded source sets, not only the subset admitted to
analysis.

`source_dropped` events distinguish `drop_scope=analysis` from an actual source
discard. They retain the exact source clock and measured host `queue_age_ns`.
For ROI analysis skips, `archived=true` means the native writer accepted that
payload; bundle finalization remains separately reported. For non-ROI packets,
`archived=null` tells consumers to resolve native membership in the thumbnail
index. An inactive recording has `archived=false`. The validator checks every
positive archived-ROI claim against the exact frame index. No compression,
re-encoding, resampling, fabricated missing RAW or new camera-side option was
introduced. The archive can be larger under overload because previously lost
native evidence is now retained.

The regression fixture archives all six native ROI packets across three
physical reads while the actual paired policy admits analysis for four. It
round-trips both RAW byte streams and exact source-clock keys; the two skipped
sources have explicit saved-but-unanalyzed receipts. No image analysis runs in
that fixture, and all six prediction rows correctly say unavailable.

The timing stress fixture adds context work, separate per-packet arrival times,
variable per-eye CPU work, and five isolated 180 ms stalls. Over 400 offered
reads at a 97 ms cadence with 80–105 ms of work per eye, complete pairs rise
from 16 to 191 and one-sided reads fall from 356 to one. With the stalls,
complete pairs rise from 16 to 181, while one-sided reads fall from 345 to five.
Under capacity with no stalls, both policies admit all 400 pairs. The
conservative recent-maximum estimate has a real recovery cost: under-capacity
stall scenarios lose an additional five or ten complete pairs (one or two per
stall). These measured synthetic tradeoffs are explicit, not an assertion of
free throughput or immunity to arbitrary CPU pauses. No packet older than
200 ms is consumed. Tests additionally cover source/session/generation changes,
temporary eviction/re-admission, and every original warmup/control/hard-failure
decision.

Those figures include the one-second timing-history expiry. A separate
counterexample test supplies a 400 ms overrun followed by only 30 ms-old heads:
the old fresh-start exception alone would reject forever, but expiry admits
new bounded work and learns the recovered 50 ms cost. Expiry never overrides
the actual packet-age gate. See `queue-expiry-stress.log`.

The post-ingress full suite reports 1,032 passes, the same 40 named pre-existing
failures, and 27 ignored tests, before adding the explicitly ignored CPU/I/O
profile. `ingress-full-tests.log` contains the run. The no-SAM-feature check
also passes after removing an inappropriate feature guard from the shared
RAW-support predicate. Runtime RAW/support semantics did not change in that
build fix. The viewer still needs a future live subject capture to establish
actual pair retention and calibration accuracy.

### Native ingest cost and exact sparse-patch reuse

The explicitly ignored `native_calibration_ingress_cost_diagnostic` measures
selected production CPU stages on the same 1,337 native recent calibration
ROIs. It does not emulate the whole host loop, camera controls, SAM or display.
Per-ROI archive writes had approximately 0.06 ms median, 9.0 ms maximum and
57.3 ms total archive finalization on this run. These are page-cache/file
operations, not `fsync` or worst-case-storage guarantees. All 1,337 benchmark
copies independently matched the input SHA256, source clocks and ROI geometry;
the benchmark archive explicitly omits applied-region transaction metadata
and must not be treated as a new live capture.

Native border-focus medians were approximately 14.2–14.4 ms in the dark clip,
11.6–12.1 ms in the brighter failed clip and 9.6–10.8 ms in the successful clip.
Eyelid extraction was 1.7–2.7 ms; whole-ROI native motion was 13.9–14.6 ms.
The image-quality-associated difference in these measured CPU stages is
modest relative to the approximately twofold difference in offered source
cadence. This does not measure every source of the original host backlog.

The native matcher now computes each sparse reference patch's 25 neutral RAW
samples and scalar moments once per bounded candidate search. Its original
RAW backing remains shared; no full neutral image, pyramid, interpolation,
threshold change or new motion prior is introduced. Forward/backward searches
retain their exact candidate ordering, costs, rejection rules and subpixel
refinement. Unit tests compare every f32 cost bit across translated, clipped,
fractional-center, odd-size and flat/rail cases. The scalar reference path
exists only in tests.

The matched recent-corpus test compares both full motion evidence and every
sparse correspondence for all 1,337 sources: all are exactly equal. Alternating
which implementation runs first gives overall median matcher times of
14.15 ms reference and 8.76 ms cached, with a 1.61x total-time ratio. The run used
an announced **shared** CPU/memory/block claim; these are not exclusive capacity
results or exposure-to-display latency. Equivalent motion cannot demonstrate
better iris localization or gaze accuracy, but it avoids changing the scale
and reframe evidence to obtain this measured CPU saving.

Artifacts: `native-ingress-profile.{jsonl,log,tar}`, `patch-cache-unit.log` and
`patch-cache-recent-parity.{jsonl,log}`. The frozen full-corpus test binary has
SHA256 `7a7c499dfe44a5586e2727fdc3171f80b336f9b02620cd11378dc9a29ce86948`.
Its independent input audit rechecked all 387,519 native payload hashes
(51,055,724,160 bytes), across 186 captures and 224 clock lineages, with no
mismatches. The separate full-corpus motion comparison and source-receipt
audit are now complete, as detailed next; the hash audit alone was not that
validation.

### Full native-motion equivalence result

All **387,519 native sources** produce identical motion evidence and every
sparse correspondence with scalar versus cached reference-patch costs. The
final merged receipt audit matches index, native SHA256, capture, clock lineage,
physical ROI, sequence and sensor timestamp for every source in order, without
missing or duplicate records. There are 3,430 native ROI-origin changes within
continuous histories. Dimension changes and explicit exclusion/reset transitions
are additionally covered by the nontrivial synthetic matcher test; no dimension
changes occur within these particular corpus histories.

This is not equality achieved only through abstention: 386,974 sources have
candidate matches, 386,973 have at least eight, and the unchanged reliable
motion counts are 145,660 right / 139,493 left. Reliability remains the existing
heuristic, not proof of eye identity or pure head motion. The test never counts
a held iris prediction as a fresh observed boundary.

The matcher owns independent per-ROI histories. For execution, 93,061 sources
from 206 **fully completed** sequential-prefix groups were reused; the remaining
294,458 sources ran in four CPU workers. Partitions follow contiguous clock-run
and physical ROI, including resets when a clock label recurs noncontiguously:
230 clock runs, 447 populated ROI groups. No individual matcher history was
split mid-run, and the joint conic solver was not partitioned this way.

On the announced shared host, overall matcher median time is **14.52 → 8.94 ms**;
p95 is **15.97 → 10.31 ms**, with a 1.609x ratio of summed times. This is about
5.6 ms lower median cost for this stage, not end-to-end latency or an exclusive
GPU/CPU capacity result. The scalar test path constructs a small unused cached
patch before invoking the old scalar cost, an explicit small baseline overhead.
There are 144 individually slower candidate timings, seven by over 1 ms
(largest approximately 2.08 ms). The only capture with a higher candidate median
or total consists of two initialization-only frames: 60 versus 160 ns total.
These reversals are retained in the report; exact outputs do not guarantee
every individual wall-clock timing improves under shared execution.

Artifacts: `patch-cache-full-parity.jsonl`,
`patch-cache-full-parity-completion.json`, `patch-cache-full-parity-summary.json`
and `patch-cache-parallel-plan.json` under the same work directory. These results
do not overturn the separate mixed human-label/SN-FEIDA geometry results or
the dark calibration's failure. A faster ingest path also changes the offered
SAM schedule; a new live recording must establish that downstream outcome.

### Original target windows with actual qualified live-bridge samples

The fixed-window diagnostic now also accepts the measured production-worker
bridge output. It retains only Ready, post-acquisition, fully completed paired
source samples, verifies the selected physical eye's exact clock, and counts
only the first qualified publication of each exposure. It uses the original
recorded target windows—not the unrelated target schedule generated by the
offline UI exercise. Targets still never enter geometry solving.

With an additional 500 ms source-window settling exclusion, both eyes in the
two brighter clips pass the affine, physical-plane and shared-support checks
for both final clusters and earliest eligible source-prefix clusters. The
brighter failed clip has no qualified samples in its original first target
window; its other eight windows provide sufficient existing coverage. A live
calibrator would hold that first target, so this is not a prediction that the
same nine points complete at the recorded times.

The dark clip **fails shared calibration support for both eyes**, despite
having qualified sources in all nine windows. The right-eye plane and affine
fits both fail; the left-eye plane passes but affine fails. Thus successful
sign acquisition and completed pairs are necessary but not sufficient. In
this clip the remaining problem cannot be described only as a missing-point
or timeout problem. The windows lack independent gaze/scanout truth, and this
run does not establish that the person followed every target correctly.

Results are in `qualified-target-windows-summary.json` and
`qualified-target-windows-{0,1,2}-eye-{0,1}.log`. These differ deliberately from
the earlier completion-paced/all-geometries evaluation: they use a different
measured worker schedule and actual live qualification, and must not be
presented as the same population of observations.

### Current build and failure accounting

After the paired timing-history expiry and native archive consumer fixture,
the SAM viewer build and no-default-features check pass. The full viewer
suite has **1,036 passes, 39 failures and 29 ignored diagnostics**. All 39
remaining failure names were present in the initial `full-tests.log`: 28 need
unavailable legacy RAW fixtures, eight are render/boundary assertions and three
are ROI-steering assertions. Missing images were not replaced, and tests were
not ignored to improve this count. The monitor consensus correction removed
`robust_metric_plane_rejects_one_antipodal_target`; the SAM rough-center fixture
now supplies its required source dimensions, with missing dimensions still
explicitly rejected by the unchanged production registration rule.

The independent source/conic reporter has 25 passing tests. The RAW-ingress
fixture also runs the actual OIM1 validator: both intentionally unanalyzed
packets resolve to their exact saved native bytes, without claiming a
prediction or verified fixation. Tree allowlisting and whitespace checks pass.
See `final-validation-results.json`, `final-queue-expiry-*.log` and
`final-test-failure-accounting.json` in the work directory.
The built SAM viewer's SHA256 is
`400fc529183f6e93b8c497e71d389da5ec2e7fef3d7e0c7e24bd3b898cde751e`.

Additional native spot checks at recent-corpus indices 14, 125, 282 and 492
retain visible eyes in the dark clip, with noisy boundaries and upper-lid
occlusion. The side-by-side PNGs are native Rust RAW previews with the
completion-paced evaluator's final same-source ellipse; they are **not** the
production-worker qualification replay, new human labels, or independent
gaze truth. They cannot turn a failed calibration into an accepted one.
Artifacts: `dark-review-{14,125,282,492}.png`.

### Next live check (requires the subject)

The viewer is rebuilt but is not recording an empty room while the subject is
away. No automatic focus, exposure, LightBox, monitor-default, stereo-default,
mouse-output or gaze-focus setting was changed by the ingest/performance work.
Nothing from this investigation has been committed or pushed.

For a future retest, preserve the currently usable SAM preprocessing/focus
setup and record the complete attempt, including acquisition and final fit
feedback. Stereo still requires the existing `3` toggle. Verify exact paired
source retention and the new analysis-drop receipts first; do not infer sensor
delivery from how many SAM results were displayed. Check that partner arrival
does not consume a second calibration vote, a timeout remains terminal, and a
completed calibration remains completed through ordinary tracking gaps.

Compare original target windows, native localization and dropouts before
judging the cursor. A separate visible-target accuracy run is still needed:
fitting the calibration targets is not an independent accuracy measurement.
If comparing LightBox states, record actual focus/exposure and sensor cadence;
the historical dark/bright clips do not hold those conditions fixed. New
human localization evidence must use the canonical native-RAW labeler with
predictions hidden until `SAVE + DONE`, not these preview overlays.

## Conditional probabilistic integration (September 12 stereo viewer)

SAM3.1 and Eye Student now share a source-aligned probabilistic joint solver
in the Stereo viewer. Enabling the existing second-ROI solver also enables this
integration. The joint MAP target, contributing groups and selected association are
preserved. In probabilistic mode, direction admission now requires a sampled
90% angular radius of at most 15 degrees, plus the existing two-eye or pupil
support condition. Unresolved sampling withholds direction authority. This is
an engineering support gate, not calibrated screen accuracy; exact-source,
age, camera-facing and paired-calibration completion checks still apply. A strong single eye can
contribute when its partner has no usable arcs; two independent gaze estimates
are never averaged. The newer bright clips retain their calibration support;
the dark clip remains a documented failure below.

The undamped full parameter information matrix yields a local angular and target
covariance after marginalizing eye-center/range, radii, pupil offset/depth and
axis alignment. Freezing those quantities at their fitted values would understate
uncertainty. Active constraints and rank deficiency keep this symmetric local
covariance unavailable. See [Ceres covariance documentation](https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/docs/source/nnls_covariance.rst)
for the distinction between observation covariance and optimizer damping.

The local matrix also proposes heavy-tailed Student-t samples around up to four
current direction basins. At a hard constraint, a feasible one-sided derivative
can propose samples without being published as Gaussian confidence. Every
sample must satisfy the original bounds and is rescored against the original
robust arc objective and priors, reconsidering arc alternatives and group
rejection. Density correction uses the complete mixture proposal; repeated
optimizer starts do not acquire extra mass. The axial-distance/log-distance
Jacobian is included. Gaussian engineering priors and uniform bounded viewpoint
slopes define the integration measure. The density is conditional on the selected
ROI association; an unlocalized-eye alternative is not integrated as another
probability model with a different dimensionality.

Integration uses deterministic common random numbers in batches of roughly 512,
up to 8,192 samples. It may stop with at least 48 effective samples and no sample
carrying over 10% of the weight. At the limit it reports an estimate only with
at least 24 effective samples and maximum weight at most 20%; otherwise covariance,
direction masses and the 90% angular radius remain unavailable. Those thresholds
are engineering diagnostics, not a guarantee of integration accuracy. Large
ESS does not certify that a region was visited; see [Owen, Importance sampling,
section 9.3](https://artowen.su.domains/mc/Ch-var-is.pdf). The reported target mean
is a distribution summary, never a replacement cursor target. No historical
sample is counted as a fresh observation.

The live recording JSON and `VIEW STATUS` expose `local_uncertainty` and
`posterior`. The latter contains conditional direction-region masses, the
sampled target covariance, and a per-eye angular radius enclosing 90% of the
sampled model mass about the selected gaze. A radius is not a pixel error or a
screen accuracy guarantee. Source/prompt/crop/clock checks and the 900 ms
presentation age limit also apply to these diagnostics. Unknown sampling support
is shown as unresolved, not zero spread.

### Matched validation and remaining failures

Frozen baseline: commit `eda3aa0` standalone evaluator, built before these changes.
Artifacts are under `outputs/probabilistic-stereo-20260912`; `baseline-eval.sha256`
identifies the executable. Candidate `posterior-eval-v4` uses `--probabilistic`.
No labels or screen targets are solver inputs.

- **Development:** first 2,500 exposures of each immutable `sam-all-a.jsonl` and
  `sam-all-b.jsonl` cache: 5,000 exposures, 3,226 reads, 18 capture entries,
  indices 100–2,599 and 193,759–196,258. All lack independent scale; 4,173 have
  unattested source clocks. Geometry and admission match the frozen baseline
  exactly. Of 2,384 available solves, 1,928 have an integration estimate and 456
  have insufficient sampling. Another 842 reads have no boundary solve. Sampled
  90% radii are often broad (per-eye medians approximately 60° for either eye), so this
  corpus does not demonstrate dependable gaze direction. Descriptive joint time
  median/p95 was 3.47/17.66 ms versus 2.15/8.39 ms baseline.
- **Reviewed subset:** 76 exposures / 49 reads / six capture entries, also without
  independent scale or attested clocks. All geometry/admission is unchanged;
  37 solves have estimates, four insufficient sampling, eight no solve. Human
  localization scores therefore remain unchanged. Ten source-matched reviewed
  labels exist, eight with jointly accepted visible-point scores; this is a
  small localization check, not gaze truth.
- **Source ordering:** first 1,000 exposures from each full cache: 2,000 sources,
  1,434 physical reads, six lineages, four native ROI reframes. Native and an
  added 100 ms partner-arrival delay preserve exact pairing, duplicate suppression
  and current-source checks. Repeated provisional/paired publications are not
  additional exposures. See the `posterior-source-*-report.json` artifacts for
  the exact candidate revision and source-count accounting.
- **Synthetic/contract tests:** 123 standalone component tests, 23 UI tests and
  ten live-adapter tests pass. Strong same-eye outer+pupil support narrows the
  sampled radius compared with the outer-only mirror pair (approximately 5.9°
  versus 34.9° in the fixture). The full two-eye fixture gives about 3.7° versus
  6.1° for one eye. A sharp pupil with a quarter outer arc recovers the exact
  synthetic target but retains broad uncertainty (about 33°); recovery of a
  noiseless optimum is not proof of identifiability. Two separated eighth-circle outer arcs near the pupil-offset direction
  support a sharp pupil with about a 12° radius and do authorize a direction;
  support placement matters, not just point count. Existing glare/blur,
  complementary-arc, crop, clock and duplicate tests remain included.

All timing runs announced shared low-priority CPU/block claims; some overlapped
concurrent student live activity. They are not exclusive latency benchmarks.
The older subsets provide no independent SN-FEIDA evaluation: candidate radius
was never substituted for missing external scale. The recent clips below provide
a separate comparison with their recorded, candidate-independent coarse scale. The camera still uses engineering
intrinsics, and refraction, subject visual-axis calibration, likelihood/coverage
calibration and missed posterior modes remain unvalidated. This phase does not
establish new-user readiness or resolve the darker clip's calibration failure.

### Recent recordings and production direction admission

Three cached production-worker clips (`recent-offered-{0,1,2}-sam.jsonl` under
`outputs/calibration-provider-fit.cVlx1Z`) add **998 exposures / 499 paired reads**.
They carry recorded MediaPipe-derived coarse scale, held between independent
acquisition updates under a nominal limbus-size assumption. This is not a fresh
physical ruler or candidate-radius normalization. Across 474 right-eye and 485
left-eye matched adjacent transitions, SN-FEIDA log steps remain exactly equal
to baseline (median 0.01332 and 0.01473, respectively). Geometry and geometric
admission are unchanged. The 8,192-sample budget provides 357 estimated
posteriors, 141 insufficient integrations and one unavailable solve. The older
2,048-sample budget estimated 168 of those reads. Descriptive joint time is
14.55 ms median / 22.44 ms p95. In the clearest clip, 103/111 reads yield estimates
with approximately 6.2° median angular radius; the dark clip still contains
broad modes and failed integration. These radii do not validate accuracy.

The frozen native test executables `viewer-calibration-{baseline,candidate}-tests`
and `calibration_compare.py` replay the same actual bridge, source/prompt gates,
paired completion barrier, acquisition state machine and recorded original
target windows for both eyes. Targets are post-fit scoring inputs only. No new
GPU inference or subject session was run. The cached worker arrival schedule is
preserved; this does not simulate extra backend latency or a subject waiting
longer when a target lacks support.

With a 500 ms source settle, **both eyes in both brighter clips retain passing
plane, affine and shared-support fits** under the new posterior direction gate.
The first target of the middle clip has no qualifying sources in either arm;
its other windows provide the existing fitter's sufficient coverage. This is
an original-window comparison, not a prediction of live completion times or
independent gaze accuracy.

The darker clip **still fails for both eyes**. Its source coverage regresses
from sufficient to insufficient: right-eye source counts per target change
from `[8,12,10,12,13,9,10,9,24]` to `[1,5,6,7,2,4,4,7,14]`; the left eye similarly
loses eligible support. Its formerly passing left-eye plane-only fit also
becomes unavailable/failing; the joint plane+affine criterion already failed.
The new gate withholds ambiguous directions rather than fixing those images.
This failure is not hidden by counting held frames, lowering the acquisition
requirements or changing the monitor fitter. Logs and full per-window reports
are `admission-*.jsonl`, `windows-*-summary.json` and
`recent-sampling-budget-comparison.json` in the work directory.

### RAW optical weighting experiment — disabled in live operation

`--raw-boundary-uncertainty` is an explicit offline experiment. It measures
per-arc native RAW edge widths, position mismatch and photometric variation
without moving points or using the candidate's own fit residual. Its optional
`localization_sigma_px` replaces the ROI optical fallback only for that arc;
contour and timing allowances remain separate. SAM and Student training and
preprocessing are not changed.

Do not enable this policy by default based on its synthetic tests. In the
reviewed subset, 169/216 arcs hit the conservative unknown allowance. It worsened
unchanged held-out outer-point RMS on many reads and produced a 3.50 px regression
on one visible-label case. Eight accepted label cases were mixed (median RMS
4.26→3.81 px; one severe case improved 37.40→29.90 px), which does not compensate
for the broader regressions or establish accuracy. The 5,000-exposure development
comparison also worsened many common accepted contour fits. Coarse priors can
pull weakly weighted geometry away from useful segmentation evidence. The frozen
`raw-uncertainty-eval-v1`, matched reports and `raw-v1-rejection-summary.json`
preserve this negative result. Live operation retains the established arc
allowances while the new posterior describes their conditional uncertainty.

### Real temporal pupil ablation: promising area case, not validated geometry

The September 12 pupil ablation now contains a reproducible **real-source area
diagnostic**, but not a human-validated example establishing that pupil support
is necessary for a correct, stable outer ellipse. The result includes a clear
counterexample in the same dark recording.

`buttercup_stereo_conic_eval --without-pupil` removes both eyes' pupil arcs and
pupil initialization hints after the normal shared extraction. It preserves
outer observations, RAW admission, camera/pose priors, independent scale and
source identities. Surviving conic support indices are remapped. The matched
998-exposure audit verifies 2,084 identical non-pupil arc records and 985
identical outer hints; it removes 5,469 pupil arc records and 764 pupil hints.
This tests the joint effect of pupil observations and initialization, not the
separate contributions of each eye or each mechanism.

All three recent SAM3.1 worker caches were replayed with and without pupils:
**499 physical paired reads, 998 native ROI exposures, three clips, one person**.
This is the recorded worker's offered subset, not every camera frame and not an
Eye Student evaluation. There are 474 comparable right-eye and 485 left-eye
adjacent area transitions. Neither arm loses extra fits: 489/499 right and
496/499 left. Source-order replay uses the live tracker, then counts each final
physical read once; held or provisional publications are not extra samples.
No transition bridges a dropout, changed clock/dimensions/scale reference or
gap over 500 ms. History supplies optimization seeds, not area smoothing.

The first pass retained the evaluator's ordinary training/withheld split.
`--all-boundary-samples` then repeated the entire comparison using every
retained native point, as the live geometry adapter does. This mode emits no
withheld probes: its outer-contour residuals are explicitly in-sample. Both
passes preserve the following area effect and regression. Figures and tables
use the full-sample pass unless marked `v1`.

The metric is **SN-FEIDA**, `pi * major_radius² / independent_scale²`, under the
frontal-equivalent planar disk model. It is neither screen-calibration affine
area nor a measured curved surface. The recorded independent MediaPipe coarse
scale is held throughout each clip, with broad engineering bounds; it does not
correct fresh within-clip depth changes. No candidate radius supplies scale.

| Case | Fresh right-eye frames / duration | Mean absolute log area step, without pupil | With pupil | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Dark clip 0, sequences 313–333 | 9 / 1.940 s | 0.06582 | 0.02142 | 67% smaller mean step; an isolated area spike at 329 disappears |
| Same clip, sequences 602–620 | 8 / 1.744 s | 0.08205 | 0.13627 | 66% larger mean step; pupil support introduces a downward excursion at 611 |
| Bright clip 2, sequences 89–98 | 9 / 1.814 s | 0.00803 | 0.00681 | Small improvement, insufficient to call pupil support instrumental |

These values are dimensionless absolute natural-log steps, not percentages of
anatomical area. All fixed two-second windows are retained in the report; these
examples were selected after examining the results and are exploratory.
The dark recording is `sam31-mouse-3d-1788826709-274018364.tar`; the bright one is
`sam31-mouse-3d-1788827261-537156697.tar`, under `outputs/calibration-corpus`.
The case report retains exact RAW offsets, SHA256, native origins, timestamps
and sequences for repeat review.

The apparent win at 329 rejects some observed outer fragments. Its all-fragment
in-sample outer RMS gets worse, and its posterior integration is insufficient;
it is not an admitted, accurate gaze result. Current accepted contour support
and sensor-coordinate motion are checked by the diagnostic test, but neither
certifies the discarded or hidden rim. The later regression also worsens
outer-contour localization. No human labels match these recent intervals.
On the separate reviewed subset, eight accepted visible-label comparisons
were available: seven are unchanged and sequence 222 improves from 4.035 to
3.378 px RMS **when the pupil is removed** in the ordinary withheld-split
evaluator. Those older examples lack independent scale. This evidence does
not justify claiming a uniform pupil benefit or tuning for constant area.

Artifacts under `outputs/probabilistic-stereo-20260912`:

- `pupil-temporal-report-v2.json`: full timelines, every fixed window, exact case
  sources, uncertainty status and coverage; `pupil-full-comparison-v2.json`
  contains independent source/duplicate audits for both arms.
- `pupil-area-cases-v2.png`, `pupil-area-all-clips-v2.png`, and
  `pupil-raw-cases-v2.png`: case comparisons, complete scope and native RAW.
- `pupil-ablation-isolation-v1.json`, `pupil-reviewed-labels-v1.json`, and
  `pupil_temporal_report.py`: unchanged-evidence audit, post-fit labels and
  reproducible analysis. The script accepts `v1` or `v2` and creates new output.

The real-source Rust test is opt-in because captures stay outside the source
tree. It uses the existing immutable cache by default, overridable with
`BUTTERCUP_PUPIL_ABLATION_CACHE`, and replays 18 source exposures through both
arms with all retained samples. It passed, alongside 125 component tests.
This guards a causal area diagnostic, not a full geometry quality claim:

```sh
cargo test --no-default-features --bin buttercup_stereo_conic_eval \
  recorded_dark_clip_pupil_support_suppresses_one_area_spike -- --ignored --nocapture
```

`--without-pupil` and `--all-boundary-samples` are offline evaluator controls.
They do not change live detector/training defaults. Independent temporal
scale/motion and canonical human labels for before/target/after exposures are
still needed before calling the area-only case a validated localization win.

### Pupil-edge continuity experiment: retained offline, not enabled live

The competing RAW peaks are ranked by contrast independently at each profile.
When their strengths exchange, following rank zero can connect different
physical edges. `--coherent-pupil-arcs` tests matching the two original peaks by
radial offset within each sector, with a 4 px adjacent-profile gate. Missing
profiles and larger jumps terminate a path. It preserves observed coordinates,
photometry and widths, and gives competing paths the same correlation group;
it does not fill gaps, borrow temporal points or turn alternatives into
independent votes. The gate is an engineering assumption, not a learned or
calibrated uncertainty model.

This remains an **offline evaluator option**. The live shared extractor calls
the original ranked-path behavior. A matched 258-exposure source replay with
the option omitted is exactly equal to the frozen baseline after excluding
elapsed runtime and the new, false option metadata. The extraction refactor
therefore has no demonstrated default-behavior change on that fixture.

The matched evaluation under `outputs/stereo-conflict-20260912` includes:

- The same three recent SAM clips as the pupil ablation: 499 paired reads,
  998 native ROI exposures, all retained samples in source-order replay.
  Geometric coverage stays at 489 right / 496 left observations, but the
  95th-percentile adjacent absolute log SN-FEIDA step worsens from 0.12029 to
  0.13260 on the right and from 0.07167 to 0.07858 on the left. There are
  474 / 485 common, fresh, short-interval transitions; absent fits break the
  chain. Scale remains the same recorded acquisition estimate, held within
  each clip, rather than a fresh physical scale measurement.
- Native SAM and Student caches for the **same** 129 paired reads / 258 ROI
  exposures from `both-eyes-1789218305-631052561.tar`, sequences 4395–4523.
  These are frozen completion-paced contour exports in
  `outputs/student-shadow.FkzvzE/{sam,student}-live.jsonl`; this evaluation does
  not rerun inference or change training. Eighty-three input exposures lack
  scale hints; 175 have acquisition hints, which are not independently
  validated rulers. Neither full-clip physical area nor gaze accuracy is
  established by these caches.
- The recent clips with the ordinary withheld-point split, plus the existing
  reviewed-label subset. Changed pupil probe sets are skipped; unchanged
  outer-limbus probes retain their exact coordinate fingerprints.

| Same-read diagnostic | Ranked baseline | Continuity experiment |
| --- | ---: | ---: |
| SAM geometrically contributing eyes, right / left | 121 / 121 | 121 / 121 |
| SAM estimated conditional posteriors | 112 | 113 |
| SAM supported directions, right / left | 67 / 66 | 56 / 55 |
| Student geometrically contributing eyes, right / left | 123 / 121 | 123 / 121 |
| Student estimated conditional posteriors | 89 | 85 |
| Student supported directions, right / left | 42 / 41 | 25 / 25 |

Counts use final physical reads, not provisional or repeated publications.
Supported directions require the existing conditional sampling and angular
width checks; these are model-conditional estimates, not measured gaze
successes. The candidate also changes some fitted ray branches by over 70
degrees, so a lower contour cost alone cannot select it for live use.

Unchanged withheld outer-contour RMS improves slightly: right median/p95
1.386/2.717 to 1.379/2.631 px, left 1.534/2.354 to 1.534/2.311 px. Three
eye-frame comparisons improve by more than 1 px and none regress by that
threshold. Of eight accepted visible human-label comparisons, seven are
unchanged; sequence 222 improves from 4.035 to 3.648 px. Those older labels
lack independent scale and do not cover the recent pupil-conflict intervals.
The small localization benefit does not outweigh the temporal and supported
coverage regressions, so the experiment is not promoted.

A separate opt-in Rust diagnostic,
`recorded_pupil_conflict_search_diagnostic`, compares 16 starts / 12 refinements
with 24 / 16 on real dark-clip sequences 325, 329 and 611. It completed; the
larger search barely changes the first two fits and reproduces the same cost,
area and rejected outer fragment at 611. This bounded result argues against
optimizer budget alone as the cause of that conflict; it is not an accuracy
test. The component suite passes 128 tests, with three corpus diagnostics
opt-in. `source-audits-v1.json` verifies source clocks, exact native identities,
reframes and duplicate suppression across both matched caches and the recent
candidate replay. `flag-off-identity-v1.json` records default equivalence.

For independent review, eight exact native targets are prepared beneath the
dark capture's `annotator/archive`: right sequences 327, 329, 331, 609, 611,
614 and left 329, 611. Each includes the immediately preceding and following
actual same-eye exposure. Left-eye sequence 328 does not exist; the context
for left 329 is 327/329/331, preserving actual times rather than fabricating
consecutive frames. Twenty-one unique RAW pieces have verified SHA256 and
native origins, dimensions and stride. `annotation-preparation.json` and
`annotation-verification.json` retain the exact provenance and checks.
The canonical labeler specified in [AGENTS.md](../AGENTS.md) loads all eight
with predictions hidden; no human labels were authored by this
evaluation. Paired, triplet and possibly-occluded annotations belong in this
capture's `annotator/labels` directory. Human review remains outstanding.

### Independent numerical references: useful joint evidence and remaining failures

The September 12 numerical audit finds real cases where combining the two
eyes reduces conditional ambiguity, but also finds that the current sampling
admission is not consistently reliable near a competing-mode threshold.
An estimated posterior and a good MAP are insufficient evidence of numerical
convergence. None of the experimental proposals below is enabled in the live
viewer, and this audit does not establish measured gaze accuracy.

`IntegrationConfig` makes the seed, integration budget and early stopping
explicit for numerical comparisons. The production configuration retains the
original seed, Student-t mixture, 8,192-draw ceiling and admission rules.
Experimental adaptation, a broad scene proposal and conditional nuisance
proposals are compiled only under `cfg(test)`. They keep the original target
loss, priors, arc alternatives, optimizer and selected MAP unchanged.

The pilot-adaptive experiment discards its pilot draws, freezes the fitted
proposal mixture while retaining the original components, and uses only fresh
draws with the full mixture density for estimation. The scene proposal adds
proposal coverage, not extra observations or a new target prior. The third
experiment mixes full Student-t components with Student-t target coordinates
and conditional Gaussian nuisances, including their relative normalization.
Density correction and limitations follow the importance-sampling framework
in [Owen, chapter 9](https://artowen.su.domains/mc/Ch-var-is.pdf). Analytic
Gaussian and bounded uniform fixtures check moments and relative mixture
normalization; these checks do not establish coverage on the eye model.

Seven deliberately difficult paired reads use every retained native point:
SAM sequences 4396/4430, Student 4397/4411/4418 from the same hotkey recording,
and recent dark SAM 329/611. Each variant has three independent seeds per
budget. The 8,192-draw arm permits normal early stopping; the 65,536-draw
reference consumes its full budget. Both eyes, sources, priors and MAP remain
identical between proposal variants.

| Proposal | Estimated / 21, up to 8,192 | Median effective samples | Estimated / 21, 65,536 | Median effective samples |
| --- | ---: | ---: | ---: | ---: |
| Original mixture | 7 | 15.20 | 12 | 41.33 |
| Discarded-pilot adaptation | 8 | 20.69 | 15 | 71.25 |
| Additional scene proposal | 11 | 24.97 | 11 | 48.50 |
| Adaptation and scene proposal | 11 | 25.07 | 12 | 37.90 |
| Conditional nuisance proposal | 10 | 22.46 | 16 | 80.32 |

These are integration-status counts, not correct-gaze counts. Improvements
are mixed across cases, budgets and seeds; more work can discover a dominant
weight and reduce effective sample size. Dark sequence 611 remains difficult.
The consolidated report retains every improvement and regression. Shared,
low-priority CPU timings are diagnostic and do not constitute an isolated
performance comparison. No candidate is promoted on these results.

The full-sample **static** comparison separately covers the same 129 paired
SAM reads, 129 paired Student reads and 499 reads from the three recent SAM
clips. Joint support for both directions while neither monocular arm admits
its own direction occurs on 58/129 SAM, 38/129 Student and 121/499 recent reads.
All three posteriors are numerically estimated on 58, 35 and 116 of those
reads respectively. These are per-read optimizations without history seeds;
they must not be substituted for the preceding source-order replay counts.

The common ten-read interval **4487–4496** initially appears fully supported
with both providers at the default seed. It spans 0.872 seconds and exactly
20 native RAW10 exposures from `both-eyes-1789218305-631052561.tar`. The audit
verifies their SHA256, source clocks, timestamps, native origins and dimensions.
Within each provider, all arms share the same coarse scene and priors; only
the presence of an eye's conic observations changes. Retained-point
fingerprints, source identities and each arm's MAP remain identical across
budgets and seeds. Monocular results are separately optimized, never averaged.

| Joint reference on those ten reads | SAM: both directions admitted / 30 frame-seed estimates | Student: both admitted / 30 |
| --- | ---: | ---: |
| Up to 8,192 draws, three seeds | 29 | 27 |
| 65,536 draws, three seeds | 30 | 27 |
| 1,048,576 draws, three seeds | 29 | 23 |

Every entry in this table has status `estimated-conditional`. At 65,536 draws,
each monocular arm remains broad on every frame and seed: its angular radius
ranges from approximately 55 to 75 degrees. At 1,048,576 draws, nine SAM frames
and seven Student frames admit both directions for all three seeds. This
preserves a conditional cross-eye benefit on much of the interval, while
refuting the initial interpretation of a consistently supported ten-frame run.

Student sequence 4488 is the clearest failure of the default numerical
interpretation: its default maximum radius is about 11.3 degrees, but the
million-draw references consistently reveal about 60 degrees of spread and
17–18% mass in the other explored basin. Student 4487/4490 also disagree across
large reference seeds. SAM 4492 straddles the 15-degree gate. Finite importance
sampling does not certify all unvisited modes; even the larger budget is a
reference diagnostic, not truth. Effective sample size and largest-weight
checks alone do not establish precision for a direction-admission quantile.

The same concern appears in synthetic support cases with the extra artificial
inner-limbus boundary removed. Complete same-eye outer/pupil evidence stays
near a 5-degree radius across three million-draw references; an outer-only
mirror case remains near 35 degrees and correctly withholds direction. Two
separated partial outer arcs plus a full pupil initially appeared supported,
but its reference radii span 13.94–15.98 degrees. A weak complementary stereo
case remains broad at about 33–35 degrees. An initial assertion that all such
partial cases were supported failed at 65,536 draws. That failure is preserved
in the logs; the explicit reference diagnostic reports the unresolved cases
instead of turning a good MAP into an unsupported convergence assertion.

There are no matching human localization labels or known screen targets for
the ten-read interval. Fourteen of its twenty exposures have coarse acquisition
scale hints; six lack them. SN-FEIDA therefore excludes three paired reads,
and the available hints are not independently validated physical rulers. The
numerical variants do not change MAP area or localization, so this audit
claims neither an area-stability improvement nor anatomical correctness.
The recordings cover one person and the workers' offered subset of exposures.
The pending native before/target/after review described above remains necessary
for the separate pupil/outer-boundary conflict.

Artifacts under `outputs/posterior-proposal-20260912`:

- `numerical-reference-report-v1.json` and `report_numerical_references.py`:
  every proposal outcome, reference seed, cohort count, exact source identity
  and recorded limitation; all 20 selected RAW hashes are checked.
- `cross-eye-reference-v2.png` / `.svg`: the full interval, including the
  higher-budget failures; angular spread is explicitly not measured gaze error.
- `real-cases-v4.log`: the consolidated five-proposal comparison;
  `cross-eye-reference-v2.log` and `cross-eye-large-reference-v1.log` preserve
  the native 8,192/65,536 and full-interval million-draw references.
- `supported-reference-v2.log`, `supported-cases-v1.log` and
  `supported-cases-tail-v1.log`: synthetic references and the original failed
  partial-case assumption, including the competing draw's actual arc support.
- `default-identity-v1.json`: 258 source-order events match the frozen prior
  default exactly after excluding elapsed time and added diagnostic metadata.
  This is separate from the static comparisons above.

The component suite passes 130 tests; eight diagnostics requiring native
runtime data or large sampling budgets are opt-in. Six native stereo UI tests,
twelve live joint-publication tests, the native SAM viewer build and the
source-tree audit also pass. No running viewer was restarted. For example:

```sh
cargo test --profile live --no-default-features --bin buttercup_stereo_conic_eval \
  recorded_posterior_proposal_diagnostic -- --ignored --nocapture
cargo test --profile live --no-default-features --bin buttercup_stereo_conic_eval \
  recorded_cross_eye_support_reference_diagnostic -- --ignored --nocapture
cargo test --profile live --no-default-features --bin buttercup_stereo_conic_eval \
  recorded_cross_eye_support_large_reference_diagnostic -- --ignored --nocapture
```

The next numerical work must address integration and admission precision near
competing-mode thresholds, while retaining honest abstention on weak geometry.
Relaxing the uncertainty gate or promoting a proposal merely because it
increases estimated-count coverage would not resolve the demonstrated failures.

### Numerical precision is observable; the stricter admission experiment is not promoted

The posterior now exports `direction_numerics` for each modeled eye: sampled
mass within 15 degrees of the selected ray, its approximate numerical standard
error, and the estimate plus/minus two standard errors. The native stereo VIEW
inspector displays the mass and numerical error in percentage points only for
the same fresh publication as its other uncertainty fields. This telemetry
does not change the selected gaze, posterior proposal, early stopping or live
admission policy. It is not a calibrated gaze-confidence interval, and it
cannot rule out an unvisited tail or mode.

The variance is a delta-method estimate for a self-normalized importance ratio,
stratified by the proposal that **generated** each sample, rather than by its
nearest fitted mode. For normalized weights `w`, event indicator `I`, estimated
mass `m`, and influence `z = w*(I-m)`, each proposal stratum contributes
`n/(n-1) * (sum(z²) - sum(z)²/n)`. Its `n` includes infeasible zero-weight
draws. Any adaptive pilot is excluded. This applies the ratio-variance and
multiple-importance-sampling ideas in
[Owen, chapter 9](https://artowen.su.domains/mc/Ch-var-is.pdf); the finite-sample
and unexplored-mode limitations remain. Eight hundred independent trials of
an analytically known stratified probability check its variance, alongside
the existing density/moment tests and a zero-weight-draw test.

A test-only candidate uses this error to continue sampling when the admission
decision is unsettled, up to the existing 8,192-draw ceiling. It admits only
when `m - 2*SE >= 0.9`, in addition to the existing integration and radius gates.
This is an engineering decision margin, not a certified coverage guarantee.
The short ten-read SAM/Student comparison removes both admitted-eye outcomes
on the three reference-broad Student-4488 seeds and retains 47 of 48 frame-seed
outcomes that stay narrow in the prior million-draw references. One consistently
narrow Student-4491 outcome is lost, and several unsettled cases remain admitted.
Strong complete same-eye outer/pupil evidence still passes; the outer-only
mirror case still withholds. Partial same-eye evidence retains numerical tail
failures, so these results do not establish all requested geometry cases.

The broader **source-order** comparison exposes a material coverage cost:

| Same native cohort | Original admitted directions R / L | Precision margin | Conditional nuisance + margin | Adaptive conditional nuisance + margin |
| --- | ---: | ---: | ---: | ---: |
| SAM, 129 paired reads | 67 / 66 | 56 / 56 | 49 / 49 | 46 / 45 |
| Student, same 129 reads | 42 / 41 | 37 / 35 | 33 / 32 | 35 / 34 |
| Three recent SAM clips, 499 reads | 299 / 302 | 281 / 283 | 302 / 303 | 270 / 271 |

All four arms preserve every MAP target, eye direction, ellipse, arc support,
SN-FEIDA value, source identity and replay scheduling field exactly. The
comparison covers 1,514 native ROI events per arm and counts final physical
reads once. Geometric contribution is unchanged: SAM 121/121 eyes, Student
123/121, recent 489/496. More admitted directions are not assumed to be better.
The conditional proposal increases estimated posteriors in the recent clips
from 364 to 382; adaptation instead reduces them to 325. Shared, low-priority
timings do not establish a live-pipeline speed improvement.

Every changed physical read from the initial precision-margin experiment is
then checked with **three independent 1,048,576-draw references**, along with
an equal-sized retained control sample. Controls are chosen by native RAW
SHA256 order, independently of their fitted confidence. That is 11+11 SAM,
7+7 Student and 19+19 recent reads, 74 selected physical reads in total.
The replay still solves the entire source history; it spends the large sampling
budget only on the selected final-arrival events. Every event's geometry and
source context is verified against the baseline. Missing or provisional
publications never become extra reference observations.

| Directions newly withheld by the initial margin | Reference stays narrow in all three seeds | Reference stays broad | Straddles threshold | Integration remains unresolved |
| --- | ---: | ---: | ---: | ---: |
| SAM | 2 | 16 | 1 | 2 |
| Student | 0 | 6 | 4 | 2 |
| Recent clips | 28 | 0 | 9 | 0 |

These classify numerical model support, **not signed gaze truth**. They show
why the blanket margin is not a sufficient repair: it removes many broad
SAM/Student directions but also rejects 28 recent-clip directions whose larger
references stay narrow. One newly admitted Student direction is reference-broad.
The retained controls themselves include four reference-broad and four
numerically unresolved directions, so the original policy is not certified.

The conditional nuisance proposal restores 16 of those 28 recent narrow
directions while preserving all 38 recent retained-control directions. It
still loses twelve narrow recent directions and has mixed outcomes elsewhere.
Adding pilot adaptation worsens recent coverage and introduces additional
reference-broad Student admissions. The fixed 74-read reference sample was
chosen for the original precision-margin comparison; newly changed cases
outside that sample are not independently certified for the later variants.
Neither proposal nor the stricter admission rule is enabled live.

No human localization or screen-target truth is added by this numerical audit.
Scale availability and held acquisition-scale limitations are unchanged from
the preceding corpus descriptions; no candidate radius supplies normalization.
The exact geometry identity also means there is no claimed localization or
area-stability improvement. The canonical native temporal label review remains
outstanding. Live subject/lighting work by the other agent retains priority;
the later comparisons use one low-priority CPU with idle I/O and no GPU.

Artifacts under `outputs/posterior-admission-20260912`:

- `source-comparison-v1.json` and the `conditional` / `adaptive-conditional`
  variants: complete source audits, unchanged geometry and every admission change.
- `reference-selection-v1.json`, `selected-reference-comparison-v1.json` and
  its proposal variants: exact selected sources, all three large-reference
  posteriors and outcomes, including regressions and unresolved sampling.
- `short-reference-comparison-v1.json`, `synthetic-admission-v1.log` and
  `report_admission.py`: initial controlled cases and reproducible analysis.
- `numerical-admission-tests-v1`, `selected-reference-tests-v1` and
  `proposal-admission-tests-v1`: frozen native test runners for the compared
  stages. Their source lives in the existing evaluator and joint-solver tests.
- `telemetry-production-identity-v1.json`: all 1,514 production events match
  the frozen default, including sampled numerical error, after excluding
  elapsed time, diagnostic contract wording and the absent test-only flag.
  The older SAM/Student baseline also matches after excluding explicitly
  checked inactive proposal metadata added during the preceding experiments.
- `inspector-render-v1`: inspected SAM and Student native VIEW fixtures, with
  both numerical-error rows visible. The exact-publication expiry test passes.

The component suite passes 132 tests, with twelve corpus/large-budget
diagnostics opt-in. Six native stereo UI tests and twelve live joint-publication
tests pass. The production evaluator and native SAM viewer builds pass; the
latter was completed with the concurrent lighting update after these telemetry
source changes. The source-tree audit passes. These are software and replay
checks; no new live gaze-accuracy measurement or human label is claimed.

The native source-order diagnostics require a fresh output directory beneath
`outputs`; they are opt-in and write new files instead of replacing prior runs:

```sh
BUTTERCUP_POSTERIOR_REPLAY_DIR=outputs/NEW_RUN \
  cargo test --profile live --no-default-features --bin buttercup_stereo_conic_eval \
  recorded_numerical_admission_source_replay -- --ignored --nocapture
```

Optional `BUTTERCUP_POSTERIOR_REPLAY_COHORT` selects `sam`, `student` or `recent`.
`BUTTERCUP_POSTERIOR_REPLAY_PROPOSAL` selects `conditional`, `adaptive` or
`adaptive-conditional` for a separate candidate output. Selected large
references additionally require `reference-selection-v1.json` in that run
directory and use `recorded_selected_posterior_reference_replay`.

### Integrating an unobserved inner radius helps coverage but does not settle competing modes

The native SAM/Student cohorts above provide outer-limbus and pupil arcs but
no observed inner-limbus arcs. A test-only integration path removes that
unobserved radius from numerical integration while preserving its Gaussian
prior and `outer >= inner > pupil` constraint. It does not remove observed
pupil evidence, change an ellipse, or turn an initializer into an observation.
Eligibility checks **all** alternative arcs, including those rejected at the
selected fit; any observed inner-limbus alternative keeps its radius explicit.

For an omitted inner radius, the admissible interval is
`[max(prior_min, pupil_radius), min(prior_max, outer_radius)]`. Its integrated
factor is `sigma*sqrt(2*pi)*(Phi(upper_z)-Phi(lower_z))`. A representative
interior radius lets the remaining unchanged conic calculation run, and its
Gaussian residual is removed from the loss before adding that factor. This
representative radius is not a fitted surface or a measurement. The proposal
uses the retained **covariance** submatrix, rather than conditioning on omitted
coordinates through an information submatrix. Student-t marginals retain the
same degrees of freedom. Tail probabilities use
[libm's complementary error function](https://docs.rs/libm/0.2.16/libm/fn.erfc.html);
very narrow intervals use direct Gaussian quadrature to avoid cancellation.
`libm` is a development dependency because this path remains test-only.

Four focused checks cover direct Gaussian integration, integration of a
correlated Student-t density, the original full conic loss integrated over
nested radius bounds, and exact output identity when inner evidence is present.
A second draw implementation retains the original full proposal's random
stream and every retained gaze candidate, while weighting with the same
marginal density. Its transformed draws are checked bit-for-bit against the
original sampler. This separates radius integration from changing which
finite set of gaze candidates happens to be sampled.

The default-seed, full-source comparison is:

| Native cohort | Original estimated posteriors | Integrated radius | Integrated radius, original draws |
| --- | ---: | ---: | ---: |
| SAM, 129 physical reads / 124 solves | 112 | 116 | 118 |
| Student, same 129 reads / 124 solves | 89 | 89 | 89 |
| Three recent SAM clips, 499 reads / 498 solves | 364 | 448 | 439 |

In the first implementation, usable draws rise from roughly 9–13% to 33–35%.
Mean draws per solved read change from 3,955/6,097/6,317 to
2,556/5,387/3,560 for SAM/Student/recent, respectively. The implementation using
the original draws uses 3,010/5,325/4,244. These are integration-work counts;
shared low-priority execution does not establish a measured live speedup.
Every native source, MAP target, eye direction, ellipse, support group,
SN-FEIDA value and scheduling field stays exactly equal in all comparisons.

The gains do not justify live promotion. Across three numerical seeds on the
previously fixed 74-read subset, the original-draw variant retains 56/66,
20/24 and 170/198 reference-narrow direction outcomes for SAM/Student/recent;
the unchanged sampler retains 58/66, 24/24 and 180/198. It also admits 32/57
reference-broad SAM outcomes versus 29/57 originally. The first marginal-draw
implementation has better aggregate counts on that subset, but the expanded
comparison reveals additional losses. A stricter two-standard-error margin
and a matched full-8,192-draw control do not eliminate the tradeoffs.

The expanded audit covers **every status/admission change** in the first
implementation: 35 SAM, 39 Student and 124 recent physical reads. Including
the earlier controls gives 259 provider/read cases, representing 240 unique
physical reads, each checked with three independent
65,536-draw references through the complete source history. Of the recent
directions newly admitted by this implementation, 144 stay narrow in all
three references, three stay broad, eighteen straddle the threshold, and
thirty have unresolved integration. It also newly withholds twelve recent
reference-narrow directions, nineteen SAM directions and twelve Student
directions. Those categories describe conditional numerical support, not gaze
truth. Several reference masses themselves vary substantially with seed;
comparison against the older million-draw runs remains recorded rather than
treating 65,536 draws as an automatic certificate. The fixed 259-read reference
set does not cover every new change introduced by the original-draw variant.

The strong complete same-eye outer/pupil synthetic case remains supported;
an outer-only mirror remains ambiguous. Partial same-eye support still lies
near the admission threshold, and the weak complementary-eye fixture remains
broad. No extra anatomical, scale or temporal observation was supplied to
force those outcomes. No new localization labels or measured gaze accuracy
were obtained, and no area-stability improvement is claimed from unchanged
geometry. The existing canonical eight-triplet native review was restored
after its server was verified stopped; all predictions remain hidden.

Both radius integration and its alternative draw policy remain `cfg(test)`
and disabled by default. The remaining task is to improve integration of
competing directions without losing supported observations, then verify
localization and gaze behavior against independent evidence. Neither more
estimated posteriors nor fewer draws alone establishes that outcome.

Artifacts under `outputs/posterior-integrated-inner-20260912` include
`report-integrated-inner-v2.json`, its `seed-*` / `fixed-budget-v1` /
`paired-draws-v1` variants, `expanded-reference-v1/expanded-report-v1.json`,
the old-million-reference comparison, and the inspected
`integrated-inner-comparison-v2.png` / `.svg`. Frozen test runners v1/v2 use
the first draw implementation; v3 uses the original full draws. The current
source selects either explicitly and records `marginal_draw_policy`.

To reproduce with a fresh directory beneath `outputs`:

```sh
BUTTERCUP_POSTERIOR_REPLAY_DIR=outputs/NEW_RUN \
BUTTERCUP_POSTERIOR_REPLAY_PROPOSAL=integrated-inner \
  cargo test --profile live --no-default-features --bin buttercup_stereo_conic_eval \
  recorded_numerical_admission_source_replay -- --ignored --nocapture
```

Use `integrated-inner-paired` for original draws; either name accepts
`-precision` for the separate admission-margin experiment. Optional
`BUTTERCUP_POSTERIOR_REPLAY_SEED`, `BUTTERCUP_POSTERIOR_REPLAY_BUDGET` and
`BUTTERCUP_POSTERIOR_REPLAY_FULL_BUDGET=1` select independent randomness and
fixed work. The selected-reference diagnostic also accepts a budget, a seed
and either draw policy, with the default three seeds preserved when omitted.
`BUTTERCUP_POSTERIOR_PAIRED_DRAWS=1` selects the paired version of the synthetic
integrated-inner diagnostic. Output files are created fresh, not overwritten.

Final verification passes 136 component tests, six native stereo UI tests,
twelve native joint-publication tests, both production builds and the source
audit. Thirteen data/large-budget component diagnostics remain opt-in. The
production replay matches all 1,514 earlier production events exactly after
excluding elapsed time, including the full posterior and numerical-error
fields; see `production-identity-v1.json`. No live viewer was restarted.

### Broader proposals in the reduced model: gains remain seed- and source-dependent

The next offline comparison combines analytic inner-radius integration with
global scene proposals, conditional Gaussian nuisance draws, and discarded-pilot
adaptation. Five initial recipes run on all three frozen cohorts. Two then run
with three independent numerical seeds and a matched full-8,192-draw control.
A sixth recipe adds a second global component: its target coordinates retain
Student-t tails while its conditional nuisance innovations are Gaussian. The
original global Student-t component remains in the mixture, and every final
sample is corrected using the entire generating mixture. These changes alter
the integration proposal, not the conic loss, scale support or anatomical priors.

The conditional sampler also now counts the actual free target coordinates.
Previously, fixing a target coordinate could cause a nuisance coordinate among
the first three active parameters to receive the target's Student-t scaling.
A regression checks both the draw moments and density ratios against the
known one-dimensional Student-t plus two-dimensional Gaussian distribution.
This correction and all new proposal switches remain test-only.

The default-seed full-source counts are:

| Native cohort | Original estimates | Global + conditional local proposal | Also conditional global scene proposal |
| --- | ---: | ---: | ---: |
| SAM, 129 reads / 124 solves | 112 | 112 | 115 |
| Student, same 129 reads / 124 solves | 89 | 94 | 88 |
| Recent SAM, 499 reads / 498 solves | 364 | 453 | 432 |

Across three seeds on the earlier fixed 74-case million-draw reference subset,
the global/conditional-local recipe admits 62/66, 24/24 and 184/198 directions
that stayed narrow in those references, versus 58/66, 24/24 and 180/198 for the
original sampler. It admits 14/57 reference-broad SAM outcomes instead of 29/57,
but 11/30 Student outcomes remain, versus 12/30 originally. Adding the conditional
global component reduces that Student count to 5/30 but returns SAM to 28/57,
and retains only 178/198 of the recent reference-narrow outcomes. These repeated
numerical seeds are not additional physical exposures or measured gaze truth.

The larger, previously fixed 259-case/240-physical-read comparison exposes
further regressions. Across the same three seeds, the global/conditional-local
recipe admits 562/654 recent reference-narrow directions versus 367/654
originally, but also admits two of 63 reference-broad outcomes and 78 of 138
numerically unresolved outcomes. The additional conditional scene component
loses ten Student reference-narrow outcomes relative to the original sampler
(60/81 versus 70/81). Each run reports every changed read outside this fixed
reference subset; the subset does not certify all newly admitted cases.

Simply forcing the full budget does not resolve the tradeoff. For the
global/conditional-local recipe, the fixed-budget SAM control admits eleven
of nineteen earlier reference-broad directions; the matched original sampler
admits three. Tail diagnostics show that some dominating samples still fit
many observed arcs, so discarding them as if all evidence were rejected would
hide model ambiguity. The updated synthetic diagnostic keeps complete same-eye
support near five degrees and an outer-only mirror near thirty-five degrees
at a million draws. The partial same-eye case remains near the admission
threshold; the weak complementary-eye case remains broad at about 33–34 degrees.

Artifacts are under `outputs/posterior-reduced-proposals-20260912`:
`proposal-comparison-v1.json`, `report_proposals.py`, the inspected
`proposal-comparison-v1.png`/`.svg`, per-source outputs and tail logs, and
`scene-synthetic-v1.log`. All fourteen complete candidate/run combinations
preserve the exact geometry and source fields in their 1,514 events. The
137 component tests and source-tree audit pass. No new localization labels,
independent scale measurements, empirical gaze accuracy or SN-FEIDA improvement
are claimed. No experimental proposal is enabled in the live solver.

The existing source-replay diagnostic accepts `integrated-inner-global`,
`integrated-inner-conditional`, `integrated-inner-adaptive`,
`integrated-inner-global-conditional`, `integrated-inner-global-adaptive`, and
`integrated-inner-global-conditional-scene`. Its existing seed and fixed-budget
environment variables still apply. The selected-reference diagnostic also
accepts `integrated-inner-global-conditional-scene`; the new
`same_eye_and_complementary_stereo_conditional_scene_diagnostic` uses this
mixture for all four known-geometry cases and three seeds at each budget.

The conditional scene mixture was additionally run with **three independent
1,048,576-draw references on all fourteen previously selected Student reads**.
This repeats the complete source history, preserves every original selection
and changes no priors or fitted geometry. Across the 28 eye directions, median
seed-to-seed mass range falls from 3.28 to 1.42 percentage points, and the
maximum from 56.41 to 6.66 points. Four directions still span more than five
points, versus twelve previously. All eight earlier consistently narrow
directions remain narrow in these runs. Of ten earlier consistently broad
directions, six stay broad, two straddle admission, and two now have unresolved
integration. One read (sequence 4415) remains insufficient in one new reference;
sequence 4439 is only just estimated, with one run's ESS around 25. The narrower
seed range is evidence of better numerical behavior on this subset, not a
finite-sample certificate or calibrated direction probability.

`student-reference-comparison-v1.json` and its inspected figure retain all
three posterior estimates for both methods, including those that disagree.
Sequence 4488 remains broad in both reference methods, despite the original
small-budget solver's confident direction. The new reference's right-eye mass
within fifteen degrees is about 86.7–88.5%, versus 81.7–83.2% earlier; that
remaining proposal dependence is not hidden by the aggregate improvement.
The production build passes and all 1,514 production replay events match the
earlier production telemetry exactly except elapsed time, including every
posterior field. No live viewer was restarted.

### Independent batches: isolate numerical diagnostics from changed draws

Four separately seeded importance-sampling streams exposed instability, but
also lost useful observations. Each stream samples the same frozen mixture,
and samples retain their original joint importance weights. Averaging the
four normalized posterior ratios would suppress a stream carrying substantial
competing probability mass and is deliberately forbidden. At an 8,192-draw
total ceiling, the separate-stream precision gate admitted one of three seeds
on the synthetic pupil-plus-partial-outer fixture. The original stream with the
existing precision gate admitted all three. That fixture's million-draw
references straddle the admission threshold, so admitting all three seeds is
not itself evidence of better integration. Increasing the ceiling to 65,536 did not
repair the corresponding Student losses: the missed reads actually consumed
the larger budget, rather than stopping early.

The new test-only `preserve_replica_draws` control partitions the **unchanged
original draw stream** into four batches, assigning complete proposal cycles
to each. Every batch balances the same proposal components. Original budget
rounding and stopping cadence remain unchanged, so batch sizes may differ by
one proposal cycle. Infeasible zero-weight draws count toward batch work; a
batch with no posterior mass cannot establish numerical agreement.

For batch indicator sums `A_r`, normalizers `Z_r`, draw counts `n_r`, total work
`N`, and `R` batches, the pooled estimate remains `m = sum(A_r) / sum(Z_r)`.
Linearizing that ratio gives the work-weighted between-batch variance estimate
`N/(R-1) * sum(((A_r-m*Z_r)/sum(Z_r))^2 / n_r)`. It reduces to the equal-work
formula when every `n_r` is equal. The self-normalized importance ratio and its
delta-method interpretation follow [Owen, chapter 9](https://artowen.su.domains/mc/Ch-var-is.pdf);
the unequal-batch expression is the implementation's corresponding derivation.
Tests compare its average estimated variance with independent simulations of
a known probability for both equal and unequal work allocations. Four batches
provide only three degrees of freedom; neither this diagnostic nor the optional
two-error margin has calibrated coverage, especially at adaptive stopping times.

With numerical admission disabled, the paired control preserves every original
posterior field, weight-derived estimate and stopping decision in all 1,514
native source events, excluding only the newly added diagnostic fields and
elapsed time. A dedicated synthetic regression and all 36 ungated synthetic
case/seed/budget comparisons also verify this identity. The core suite now has
140 passing tests. With the precision gate enabled, all three partial same-eye
synthetic solves survive; complete support remains admitted, and the weak
complementary-eye and outer-only mirror fixtures remain broad and withheld.

The default-seed comparison on the fixed earlier 74-case reference subset is:

| Outcome | Original stream + precision | Separate streams + precision | Same draws + batch precision |
| --- | ---: | ---: | ---: |
| Reference-narrow SAM retained | 22/22 | 16/22 | 22/22 |
| Reference-narrow Student retained | 8/8 | 4/8 | 8/8 |
| Reference-narrow recent SAM retained | 64/66 | 56/66 | 64/66 |
| Reference-broad SAM admitted | 5/19 | 2/19 | 3/19 |
| Reference-broad Student admitted | 4/10 | 2/10 | 4/10 |

Two additional seeds show that the batch check adds only a modest benefit to
the existing precision estimate. Across three seeds, earlier reference-broad
SAM admissions fall from 11/57 to 9/57, with the same 59/66 reference-narrow
retention. Student and recent admission decisions are unchanged. On the wider
259-case reference subset, SAM reference-narrow retention falls from 106/162
to 105/162. The newer fourteen-read Student million-draw references still have
only 34/39 reference-narrow directions admitted and 6/21 reference-broad
directions admitted under either precision method. Those repeated seeds are
numerical trials on the same physical reads, not independent recordings.

The candidate remains experimental. It isolates the earlier regression without
solving the remaining proposal-tail problem or establishing real gaze accuracy.
No new canonical human labels or independent scale measurements were available;
there is no claimed localization or SN-FEIDA improvement. All 18 new native
candidate files preserve their geometry and source fields in 9,084 event
comparisons. The production replay remains exactly identical in its 1,514
events apart from elapsed time, including the complete posterior. Six native
stereo UI tests, twelve joint-publication tests and the SAM-enabled native build
pass. The initial native compile encountered concurrent optical-clock edits;
their author repaired those errors before the successful retry. This work
does not restart the live viewer or alter camera ownership.

Artifacts are under `outputs/posterior-replicates-20260912`: the original and
65,536-ceiling separate-stream comparisons, `original-draws-v1`, the two
`original-seed-*-v1` directories, `synthetic-original-comparison-v1.json`,
`original-control-summary-v1.json`, the inspected
`original-control-comparison-v1.png`/`.svg`, and
`production-original-identity-v1.json`. The source replay accepts
`integrated-inner-global-conditional-replicated-original` and its
`-precision` variant. `BUTTERCUP_POSTERIOR_REPLICA_ORIGINAL_DRAWS=1` selects the
same control in `same_eye_and_complementary_stereo_replicated_diagnostic`.

### Broader scene sampling plus numerical precision is still not a default

The next comparison combines the conditional global-scene proposal with both
the existing numerical margin and the paired batch check. The original local
Student-t, conditional local and global Student-t components remain; the
additional global component retains Student-t target coordinates and Gaussian
conditional nuisance innovations. Every sample uses the complete frozen
mixture density. The joint loss, priors, source association and selected MAP
geometry remain unchanged. New source-replay recipes select this combination
explicitly instead of accidentally dropping the scene component when a
precision suffix is present.

All three frozen cohorts run with three numerical seeds at the same 8,192-draw
ceiling. They contain 628 unique physical reads: 129 shared by the SAM and
Student providers, plus 499 recent SAM reads. The extra default-seed ungated
paired control preserves all posterior fields in the earlier scene-proposal
replay, excluding only batch telemetry and elapsed time. Across the 21 new
candidate files, all 10,598 source events preserve their geometry and source
fields. Numerical repetitions are not additional recordings or fresh evidence.

For the paired precision gate, the three-seed totals are:

| Outcome on earlier fixed references | Previous proposal | Also conditional global scene |
| --- | ---: | ---: |
| Reference-narrow SAM retained | 59/66 | 50/66 |
| Reference-narrow Student retained | 24/24 | 22/24 |
| Reference-narrow recent SAM retained | 188/198 | 176/198 |
| Reference-broad SAM admitted | 9/57 | 13/57 |
| Reference-broad Student admitted | 8/30 | 1/30 |

The wider fixed 259-case reference subset also retains losses: narrow SAM
retention changes from 105/162 to 102/162, Student from 65/81 to 52/81, and
recent SAM from 553/654 to 543/654. The newer fourteen-read Student
million-draw references show fewer broad admissions, 6/21 to 1/21, but narrow
retention falls from 34/39 to 30/39. The version without the batch check still
has these Student losses. Selecting only the improved ambiguity counts would
hide the losses on numerically reference-narrow native directions.

The synthetic comparison covers all four support cases, three seeds, both
precision settings, one/four batches and the existing 8,192/65,536/1,048,576
budget schedule. All 36 ungated one/four-batch posterior comparisons remain
exact. Complete support stays admitted. Two of three seeds on the partial
same-eye pupil/outer fixture pass at the small budget, versus all three with
the previous proposal. Longer numerical runs put this fixture near the
admission threshold; that difference alone is not a demonstrated regression.
Weak complementary-eye and outer-only mirror cases
remain broad and withheld.

A new diagnostic also exports each selected gaze and its error against the
independent 3D-circle forward fixture. All four noiseless fits recover their
constructed directions to numerical precision (below 0.00001 degrees). This
does **not** make the weak or mirror solutions identifiable: a correct selected
mode can coexist with substantial competing mass. These favorable fixtures
use the specified pinhole camera and anatomy; their near-zero error is not
native gaze accuracy or evidence of performance under unknown anatomy.
`BUTTERCUP_POSTERIOR_REFERENCE_FITS_ONLY=1` runs these four fits without repeating
the integration matrix, and its log explicitly distinguishes fit truth from
posterior validation.

The change inventory records every newly admitted/lost direction, including
those outside the fixed reference sets. Among the paired candidate's lost
recent-SAM admissions across seeds, 179 fail the integration-quality checks,
43 fail only the numerical margin and five have a broad sampled direction.
It also newly admits 167 recent directions. Thus the small net change hides
substantial sampling churn; changing the margin alone would not fix it.
These counts describe repeated numerical evaluations, not independent eyes.

This combination remains test-only. There are no new canonical human labels,
independent scale measurements, localization or SN-FEIDA improvements, or
calibrated gaze probabilities. The 140 core tests pass; the native SAM-enabled
fit diagnostic and production builds pass. The rebuilt production evaluator
is byte-identical to the preceding executable whose complete 1,514-event
posterior replay was verified. No live viewer is restarted.

Artifacts live under `outputs/posterior-scene-precision-20260912`: each
`seed-*-v1/scene-precision-comparison-v1.json`, `synthetic-comparison-v1.json`,
`synthetic-fits-v2.log`, `scene-summary-v1.json` with the complete change
inventory, and the inspected `scene-comparison-v1.png`/`.svg`. The added
source recipes are `integrated-inner-global-conditional-scene-precision`,
`integrated-inner-global-conditional-scene-replicated-original`, and its
`-precision` variant. The corresponding synthetic test is
`same_eye_and_complementary_stereo_scene_replicated_diagnostic`, with
`BUTTERCUP_POSTERIOR_REPLICA_ORIGINAL_DRAWS=1` selecting the paired draw control.

### Fit eye geometry before proposing around a discarded pilot

The next bounded experiment tests a specific sampling failure: a high-weight
pilot point can lie in an underrepresented region while still having a poor
local eye-geometry fit. Merely moving an existing proposal to that point, or
computing its curvature there, spends too much of the final budget on poor
configurations. Four test-only recipes separate the effects: discard a pilot
without adapting, recenter the original shape, refit the shape at the raw
pilot, and refine nuisance geometry before refitting the shape.

All recipes spend at most 2,048 pilot draws from the original proposal mixture.
Adaptive recipes add up to four components, choosing successive pilot points
by their importance weight against the currently augmented mixture. Pilots
are discarded. The original components remain and every fresh final draw is
weighted by the complete frozen mixture density. These are importance proposals
for the same generalized posterior; they neither add observations nor average
independent gaze solutions.

Using high-weight configurations to place new components is motivated by
[incremental mixture importance sampling](https://arxiv.org/pdf/1611.06874).
This implementation uses bounded conditional Gauss–Newton refinement and a
discarded pilot, rather than that paper's Langevin moment equations or reuse
of adaptation samples in the final estimate.

The conditional-refit variant holds the three coordinates of the **one shared
fixation** fixed and runs at most six existing refinement iterations on eye
geometry. Only that temporary proposal-fitting problem has fixed coordinates.
The original model supplies the full-dimensional curvature and final density;
its priors, source associations and selected MAP remain unchanged. The bounded
refinement is not claimed to find a global conditional optimum, especially
after analytic elimination of an unobserved inner radius. Correct importance
weights still target the original density. Four six-iteration refinements add
CPU work beyond the pilot/final draw counts; equal draw ceilings do not imply
equal runtime.

At the 8,192-draw ceiling, the three-seed fixed-reference comparison is:

| Outcome | No pilot + paired precision | Pilot only | Shape at raw pilot | Refine eye geometry first |
| --- | ---: | ---: | ---: | ---: |
| Earlier reference-narrow SAM retained | 59/66 | 62/66 | 48/66 | 54/66 |
| Earlier reference-narrow Student retained | 24/24 | 18/24 | 16/24 | 24/24 |
| Earlier reference-narrow recent SAM retained | 188/198 | 182/198 | 172/198 | 188/198 |
| Earlier reference-broad SAM admitted | 9/57 | 16/57 | 21/57 | 6/57 |
| Earlier reference-broad Student admitted | 8/30 | 8/30 | 6/30 | 3/30 |

The recentered proposal also loses reference-narrow directions (40/66 SAM,
10/24 Student, 170/198 recent) and remains in the complete report. Conditional
refinement is substantially better than both unrefined adaptations. On the
wider 259-case reference subset it changes narrow retention from 105/162 to
109/162 SAM, 65/81 to 66/81 Student, and 553/654 to 615/654 recent SAM. On the
newer fourteen-read Student references, narrow retention stays 34/39 while
broad admissions fall from 6/21 to 0/21. One direction categorized as unresolved
by those newer references is now admitted; zero broad admissions does not
certify every newly admitted case.

The conditional variant's remaining earlier-reference SAM misses occur on
sources 4406, 4410, 4452, 4480 and 4495, depending on seed. They are estimated
posteriors withheld by the numerical margin, with reference masses close to
the 0.9 threshold. This accounts for real small-budget losses but is not proof
that the earlier small-budget admissions were correct. The complete inventory
also retains changes outside the fixed reference subsets. No new canonical
human labels or independent scale measurements were used, so this experiment
does not establish gaze accuracy, localization or SN-FEIDA improvement.

All four recipes cover the four existing synthetic support geometries, three
seeds and both 8,192/65,536 budgets. Complete same-eye pupil/outer support stays
admitted; weak complementary-eye and outer-only mirror support stays withheld.
The separated pupil/outer fixture requires a correction to the interpretation
of earlier experiments: its three existing million-draw mass estimates are
0.9260, 0.9097 and 0.8779. It is threshold-sensitive, not a proven must-admit
positive. Forcing three small-budget admissions would be an invalid objective.
With conditional refinement it admits one of three small-budget seeds and
none at the larger budget. One weak complementary-eye larger-budget run still
fails the integration-quality check despite spending the full budget.

The 36 candidate files preserve all source and MAP fields in 18,168 event
comparisons on 628 unique physical reads. These comprise 129 reads shared by
SAM and Student plus 499 recent SAM reads. Repeated seeds are numerical trials,
not new recordings. The 142 core tests include the recentered-density check
and a regression proving that nuisance refinement improves fit without moving
the shared target or changing the original model bounds.

The SAM-enabled native build, the same refinement regression, six stereo UI
tests and twelve joint-publication tests pass. The newly built production
executable has a different hash, so it was replayed rather than assumed
identical: all 1,514 complete production events, including posterior fields,
match the preceding verified replay exactly apart from elapsed time.

Artifacts are under `outputs/posterior-tail-refinement-20260912`: the original
three `original-seed-*-v1` reports, `conditional-refit-v3`, the four synthetic
logs, `tail-summary-v3.json`, and the inspected `tail-comparison-v3.png`/`.svg`.
The new source recipe is
`integrated-inner-global-conditional-tail-conditional-refit-replicated-original-precision`;
the synthetic diagnostic accepts
`BUTTERCUP_POSTERIOR_TAIL_PROPOSAL=conditional-refit`. Production defaults remain
unchanged while numerical failures and native accuracy remain under evaluation.

The complete 65,536-draw follow-up disables early stopping. Its 4,437 pilot
sequences, including conditional-refinement decisions, are exactly equal to
the corresponding 8,192-draw runs. All 4,542 additional native events preserve
source and selected geometry. This isolates the final draw budget and stopping
policy from changing pilot proposals. All sampled solves spend at least 65,000
draws, so the remaining failures cannot be attributed to early termination.

| Conditional-refit outcome | 8,192 ceiling | 65,536 fixed budget |
| --- | ---: | ---: |
| Earlier reference-narrow SAM retained | 54/66 | 60/66 |
| Earlier reference-narrow Student retained | 24/24 | 18/24 |
| Earlier reference-narrow recent SAM retained | 188/198 | 188/198 |
| Wider reference-narrow SAM retained | 109/162 | 103/162 |
| Wider reference-narrow Student retained | 66/81 | 48/81 |
| Wider reference-narrow recent SAM retained | 615/654 | 621/654 |
| Earlier reference-broad SAM admitted | 6/57 | 0/57 |
| Earlier reference-broad Student admitted | 3/30 | 2/30 |

Newer Student reference-narrow retention also falls from 34/39 to 27/39, while
its broad admissions remain 0/21. Across the complete corpus, Student has 55
lost and 18 new direction admissions; recent SAM has 120 lost and 74 new.
Some changing directions lie outside the fixed references or near their
thresholds. The aggregate improvements therefore do not justify promotion.

The dominant-weight diagnostic is now joined to exact native publications,
with matching seed, sample weight, pilot and arc order. At 65,536 draws, 32
Student publications and 55 recent-SAM publications still contain a single
sample with more than 20% of total weight. In 24 and 43 of those respectively,
that sample rejects arcs used by the selected MAP. The newly rejected arcs
include 23 Student and 73 recent pupil arcs. These counts include both arrivals
of a physical read and repeated seeds; they are not independent observations
or labels proving that a detected pupil was wrong. They identify a concrete
proposal-coverage problem involving alternative contour/outlier explanations,
which the next experiment must explore while scoring the complete original
model. Removing those samples or their pupil evidence would conceal it.

The follow-up artifacts are `conditional-refit-65536-v3`,
`conditional-budget-comparison-v3.json`, `dominant-samples-v3.json`, and
`verdict-v3.json`. Across all 45 candidate files in this phase, 22,710 source
event comparisons preserve geometry. The broader stereo objective remains
unfinished; production sampling and camera ownership are unchanged.

### Competing contour proposals need more than omitted factors or reallocation

The next comparison explores the pupil/outlier alternatives exposed by the
dominant-weight audit. Three test-only recipes change **proposal fitting**;
every final sample still uses the complete original current-contour density,
including pupil factors, priors, nested-radius bounds and outlier charges.
Neither ROI association nor the selected shared-fixation MAP changes.

`OutlierRefit` temporarily omits groups already rejected at a selected pilot
point while refining nuisance geometry at that same shared fixation. Its full
proposal curvature uses that temporary objective. This mostly reproduces the
constant-cost plateaus of the existing capped loss: 22/24 synthetic posteriors
remain exactly equal, and native changes are small. It is not a way to remove
the pupil's penalty from the final estimate; a dedicated regression checks
that the original objective continues to pay the complete rejected-group cost.

`BoundaryRefit` also deliberately relaxes all groups for one boundary kind
from one eye during proposal fitting. The bounded attempt order is right
pupil, left pupil, right outer, left outer, then available inner-limbus groups
when earlier kinds are absent. There are at most four attempts; available
masks repeat if fewer than four exist. Mixed-kind alternatives stay together.
All other-eye geometry, original scene bounds and final evidence remain.
This expands proposal coverage but loses too many numerically supported solves.
Some relaxations also lack valid full-dimensional curvature: 324 SAM and
1,644 recent-SAM refinement attempts cannot create a component across the
three seeds. They are skipped without adding a ridge or asserting certainty.

`BoundaryDefensive` isolates a possible allocation problem. It uses precisely
the same pilot, nuisance refinements and proposed boundary alternatives, then
triples each original component's allocation in the final mixture. Repeated
slots are integer mixture weights, not new modes or observations. Draws use
the complete weighted density. The known-uniform-target regression now checks
both equal and 3:1 component allocations. All 4,437 native pilot/refinement
sequences match `BoundaryRefit` exactly; only final allocation and subsequent
stopping change. This control does not repair the regressions.

At the same 8,192-draw ceiling and across the same three numerical seeds:

| Outcome | Prior conditional refinement | Pilot outlier pattern | Explicit boundary alternatives | Triple original allocation |
| --- | ---: | ---: | ---: | ---: |
| Earlier narrow SAM retained | 54/66 | 54/66 | 50/66 | 54/66 |
| Earlier narrow Student retained | 24/24 | 24/24 | 20/24 | 20/24 |
| Earlier narrow recent SAM retained | 188/198 | 186/198 | 184/198 | 176/198 |
| Earlier broad SAM admitted | 6/57 | 7/57 | 17/57 | 16/57 |
| Earlier broad Student admitted | 3/30 | 2/30 | 5/30 | 4/30 |
| Wider narrow Student retained | 66/81 | 66/81 | 52/81 | 60/81 |
| Wider narrow recent SAM retained | 615/654 | 611/654 | 552/654 | 537/654 |

The wider SAM subset improves from 109/162 to 120/162 with explicit boundary
alternatives, but selecting that result alone would conceal Student and recent
losses. On the newer Student references, narrow retention changes from 34/39
to 29/39 and 32/39 for explicit/weighted alternatives; broad admissions rise
from 0/21 to 1/21 and 2/21. Across all reads, those two alternatives lose 355
and 329 recent-SAM direction admissions to integration insufficiency alone.
The prior refinement therefore remains the more useful experimental control.

The three recipes each run all four synthetic support cases, three seeds and
8,192/65,536 budgets. Complete same-eye pupil/outer support stays admitted;
weak complementary-eye and outer-only mirror cases stay withheld. The partial
same-eye fixture remains threshold-sensitive and has no larger-budget
admissions. Reallocation makes all three larger-budget weak-stereo estimates
numerically available, but they remain broad; the other two recipes still
have one insufficient-sampling seed. This is neither a new positive stereo
success case nor evidence of anatomical accuracy.

The 27 candidate files preserve geometry and source fields in 13,626 event
comparisons. All 1,514 complete prior-control events are reproduced after each
of the three source changes, including posterior fields apart from elapsed
time. The rebuilt production evaluator also preserves its complete 1,514-event
replay. The 144 core tests, fifteen SAM-enabled proposal tests, six stereo UI
tests and twelve joint-publication tests pass; the production builds pass.
No new source-tree paths, training material or camera changes are introduced.

These are still 628 unique physical reads, including the 129 shared by SAM and
Student and 499 recent SAM reads. Numerical repetition does not add labels,
independent scale, source timing information or measured gaze accuracy. The
next useful experiment must improve the shape and discovery of proposal
support, rather than treating arbitrary contour omission or simple allocation
as a validated remedy. All three variants remain test-only.

Artifacts are under `outputs/posterior-outlier-proposals-20260912`:
`comparison-v3.json`, `original-seed-*-v1`, `boundary-v2`, `defensive-v3`,
the three synthetic logs, complete control reproductions, and
`production-identity-v3.json`. Source recipes append
`tail-outlier-refit`, `tail-boundary-refit` or `tail-boundary-defensive` to
`integrated-inner-global-conditional`, followed by
`-replicated-original-precision`. The corresponding synthetic environment
values are `outlier-refit`, `boundary-refit` and `boundary-defensive`.

### Curved conditional proposals preserve density but do not resolve the native losses

The next matched experiment adds test-only `ProfileAffine` and
`ProfileQuadratic` recipes. Neither is promoted. Both first perform the
unchanged discarded pilot and four conditional refinements. Only after all
ordinary components have been fitted do they transform the newly added
components. Original local, conditional and global components remain present.
The two variants therefore share their pilot, anchors, covariance, allocation
and profile-fitting probes; the quadratic coefficients are their only
difference before final draws.

For target coordinates `t` and retained nuisance parameters `u`, the transport
is `S(t,u) = (t,u+d(t))`. Its block-triangular Jacobian has determinant one,
so the transported proposal density is exactly
`q_S(t,u) = q_0(t,u-d(t))`. Fixation coordinates are unchanged. The sampler
applies the forward shift after any delegated full-dimensional marginal draw;
the density applies the inverse shift before evaluating the base density.
Omitted inner radii are never shifted. This is still one shared latent fixation,
with the original evidence, priors and feasibility checks in every final
importance weight; it does not average separately estimated eye gazes.

Each component probes its center and both signs of up to three whitened target
axes. At each fixed target, the original full conditional robust objective gets
at most six nuisance-refinement iterations. The resulting displacements fit a
constant, affine terms and diagonal quadratic terms. These are conditional
mode fits, not exact conditional posterior means. Probe steps keep both signs
inside the original target bounds; bound-active axes may be skipped. Shift
features are clipped to twice the probe step, while the fixation itself is
never clipped by this transform. There are no quadratic cross terms and no
target-dependent nuisance covariance. An invalid profile retains its proper
base proposal.

The analytic regression samples a correlated mixture against a uniform target
in the final coordinates and checks its known means, second moments and cross
moment. It covers affine and quadratic shifts, Student-t and conditional
Gaussian nuisance proposals, delegated marginal draws, non-unit parameter
scales, inverse round trips and both clipping tails. It also checks that target
coordinates and RNG state remain exactly paired.

All 4,437 native pilot sequences match the previous conditional-refinement
control. All 17,746 fitted transports match between affine and quadratic arms,
apart from applying curvature. Every transport fit succeeds; 17,707 fit all
three axes and 39 fit fewer. The work adds 124,056 conditional fits and 830,728
line-search steps per recipe across the three seeds. This optimization cost is
additional to the 8,192-draw ceiling, not part of a fixed CPU-time comparison.

| Outcome across three seeds | Previous conditional refinement | Affine profile | Quadratic profile |
| --- | ---: | ---: | ---: |
| Earlier narrow SAM retained | 54/66 | 54/66 | 54/66 |
| Earlier narrow Student retained | 24/24 | 22/24 | 22/24 |
| Earlier narrow recent SAM retained | 188/198 | 188/198 | 186/198 |
| Earlier broad SAM admitted | 6/57 | 6/57 | 7/57 |
| Earlier broad Student admitted | 3/30 | 4/30 | 3/30 |
| Wider narrow SAM retained | 109/162 | 108/162 | 108/162 |
| Wider narrow Student retained | 66/81 | 61/81 | 61/81 |
| Wider narrow recent SAM retained | 615/654 | 606/654 | 610/654 |

On the newer Student references, narrow retention is 34/39, 34/39 and 35/39,
with no broad admissions in any arm. That isolated improvement does not erase
the wider losses. Across all reads, affine/quadratic transport loses 10/8
Student and 29/21 recent-SAM direction admissions to insufficient integration;
the remaining losses come from a broad sampled direction or the numerical
margin. `comparison-v1.json` retains every changed direction and its before/
after posterior. These fixed reference classes are finite numerical estimates,
not measured gaze truth; a changed admission alone cannot establish accuracy.

All 48 synthetic case/seed/budget evaluations finish. Complete same-eye outer
and pupil support remains admitted for all seeds at 8,192 and 65,536 draws.
Weak complementary-eye and outer-only mirror cases remain withheld. Quadratic
transport makes the previously insufficient larger-budget weak-stereo estimate
available, but broad; this is numerical progress, not a new positive stereo
identifiability case. The threshold-sensitive partial-pupil fixture admits
1/3 seeds with affine transport and 0/3 with quadratic at the small budget;
both withhold all three at the larger budget. Do not tune this fixture toward
unanimous small-budget admission.

The 18 candidate source files preserve geometry and source fields in all 9,084
event comparisons. All 1,514 complete conditional-control events reproduce
exactly apart from elapsed time, and the rebuilt production evaluator also
preserves all 1,514 production events. The 145 core tests, sixteen native
proposal tests, six stereo UI tests and twelve joint-publication tests pass;
the live-worker replay remains explicitly ignored. Native and evaluator
production builds pass. Source-tree and diff checks pass. The first synthetic
invocation was rejected before solving by an outdated test recipe allowlist;
v2 corrects only that guard, and its solver/evaluator sources match v1 exactly.

The actual subset is unchanged: 628 unique physical reads, comprising the 129
shared by SAM and Student plus 499 recent SAM reads from three frozen captures.
There is no observed true inner-limbus boundary in these caches, only outer
limbus and pupil evidence. This experiment adds no canonical human localization
labels, independent scale, native gaze truth or measured SN-FEIDA improvement.
Correct transport density and lower profile costs do not establish posterior
convergence or empirical probability calibration. The next investigation must
address the competing posterior mass behind the saved failures, rather than
treating local curvature or higher effective sample size as sufficient proof.

Artifacts are under `outputs/posterior-profile-transport-20260912`:
`comparison-v1.json`, the inspected `comparison-v1.png`/`.svg`, the 18 source
files and complete control reproduction, `synthetic-profile-*-v2.log`,
`production-identity-v2.json`, both frozen test executables, and source hashes.
Recipes append `-tail-profile-affine` or `-tail-profile-quadratic` to
`integrated-inner-global-conditional`, then `-replicated-original-precision`.
Synthetic recipe values are `profile-affine` and `profile-quadratic`.

### Independent annealed paths improve integration but expose retained-contour ambiguity

An offline reference now uses [Neal's annealed importance sampling](https://arxiv.org/abs/physics/9803008)
to check the preceding importance-proposal failures. This is test-only
`posterior::annealed`, not a new production confidence source. It retains one
shared fixation and the complete original robust contour/prior density.
No independent-eye gaze average, contour omission, MAP change or default
promotion is involved.

The frozen conditional-refinement proposal mixture is the starting density.
Independent paths traverse bridges `q^(1-beta) p^beta`, with
`beta=(stage/steps)^2`. Each path accumulates the incremental log density ratio
before a Metropolis transition. Every fourth transition uses an independence
proposal from the entire frozen mixture; the others use symmetric Gaussian
walks with fixed component covariances and widths. Hard-bound violations are
rejected, never projected. Initial infeasible draws remain zero-weight paths
in the denominators. There is no resampling or adaptive stopping. Each path
contributes one endpoint with its full path weight; transitions do not count
as independent observations. The initial proposal draws are exactly matched
between zero, 32 and 128 transitions. Zero transitions are ordinary importance
sampling from those same draws.

The selection was fixed before these replays: all lost admissions in the
preceding quadratic experiment, plus two additional reads per available
numerical-reference class. SAM and Student use the union of their selections.
There are 34 SAM, 34 Student and 29 recent-SAM provider reads, representing
63 unique physical reads. One selected SAM read has unavailable geometry and
remains in the coverage denominator. This deliberately difficult subset is
not a representative performance benchmark. Each condition requests 512
paths, rounded down to an equal allocation across proposal components, and
runs three fixed seeds.

| Native trials passing the path check | 0 transitions | 32 transitions | 128 transitions |
| --- | ---: | ---: | ---: |
| SAM, including unavailable geometry | 62/102 | 77/102 | 92/102 |
| Student | 23/102 | 42/102 | 74/102 |
| Recent SAM | 32/87 | 47/87 | 60/87 |

The path check requires effective path count at least 24 and maximum path
weight at most 0.2. Passing does not establish convergence. Median effective
path counts rise from 27.3 to 80.3 for SAM, 14.3 to 39.4 for Student and 18.5
to 39.8 for recent SAM. The 128-transition diagnostics require 2,548,852,
2,537,080 and 2,064,487 model evaluations respectively across the selected
three-seed trials, in addition to the unchanged ordinary posterior control.
This is not an equal-work comparison. Candidate admission counts in the
report use per-path numerical error; the ordinary control also uses its
four-batch precision check, so their counts are not matched confidence levels.

Longer paths do not resolve every disagreement. Student read 189 has
within-15-degree masses 0.977, 0.883 and 0.975 at 128 transitions; all three
path checks pass. At 32 transitions all three runs had passed the candidate
direction rule. Student read 255 likewise remains inconsistent at 0.982,
0.829 and 0.951. Conversely, reads 79 and 249 agree across the three longer
runs, with joint-weight pooled masses 0.990 and 0.979. Pooling combines
unnormalized path contributions, not averages of normalized posterior ratios.
The between-seed diagnostic has only two degrees of freedom and cannot bound
unvisited modes.

Mapping the saved rejection-pattern indices back to native support changes
the next investigation. `Problem::solution` emits one `ArcSupport` in model
group order, and the evaluator serializes that order without filtering.
Group kinds in this mapping name the MAP-selected alternative. In Student
read 189, seed two, only 0.995% of sampled mass rejects any MAP-used group,
while 11.729% lies outside 15 degrees of the selected first-eye gaze. Therefore
at least 10.734% of that *estimated* mass lies outside while retaining every
MAP-used contour group. Read 159 has corresponding retained-group tail lower
bounds 7.680%, 11.335% and 11.372% across its three seeds. These are Frechet
bounds calculated from saved marginals, not direct joint measurements or
confidence bounds on the true posterior. Competing sampled gaze cannot be
attributed solely to dropping pupil/outer groups; simply penalizing group
rejection would not address these examples.

The synthetic comparison explicitly distinguishes pupil evidence from a
true inner-limbus ring. All six selected MAP directions agree with the
independent noiseless forward fixture to below 0.000001 degrees, yet their
posterior behavior differs. At 128 transitions, full outer-plus-pupil support
has 90% angular radius 4.96--5.69 degrees; partial outer-plus-pupil support
remains threshold-sensitive at 14.53--17.29 degrees. Deliberately weak
complementary stereo is broad at 30.47--37.85 degrees. Full and partial
outer-plus-true-inner cases remain mirror-ambiguous at approximately
34--36 degrees, as does outer-only support. The two iris rings are coplanar
under this fixture and its measurement assumptions; these results do not
establish that independently sharper inner evidence could never help. They
do establish that selecting the correct MAP is insufficient evidence of
direction identifiability.

All 27 native replay conditions finish, with 864 exact exposure-key diagnostic
joins. Every source and ordinary posterior field is identical across all
13,626 event comparisons apart from elapsed time. All 1,514 rebuilt production
events also match the preceding production output apart from elapsed time.
All 54 synthetic configurations finish and the 36 repeated ordinary controls
match exactly. Three analytic tests check known mixture moments/normalizers,
invalid starting points and hard boundaries, and heterogeneous Student-t/
conditional-Gaussian proposal mixtures with retained marginal samplers.
The third test was added after the replays; the sampler body is byte-identical
to the frozen replay version. All 148 core tests and all three native sampler
math tests pass. The sixteen proposal, six stereo UI and twelve publication
tests pass; designated live/expensive diagnostics stay ignored. Native and
evaluator builds pass, as do source-tree and diff checks.

No native true-inner observations, canonical localization labels, independent
gaze truth or scale/timing measurements were added. Native inputs remain
outer-limbus/pupil evidence from the frozen SAM/Student caches. This does not
demonstrate calibrated probabilities, new-user readiness or improved SN-FEIDA.
Remaining work includes explaining the contour-preserving competing geometry,
resolving insufficient integration, and validating the intended positive
partial/cross-eye cases before promoting an inference change.

Artifacts are under `outputs/posterior-annealed-20260912`: fixed
`reference-selection-v1.json`, `comparison-v3.json`, inspected
`comparison-v4.png`/`.svg`, `evidence-patterns-v4.json`,
`production-identity-v3.json`, `source-hashes-v4.json`, all replay/test logs
and the frozen `annealed-tests-v3` executable. The ignored selected-native
recipe is `extraction_tests::recorded_selected_annealed_reference_replay`,
with `BUTTERCUP_POSTERIOR_REPLAY_DIR`, `BUTTERCUP_POSTERIOR_REPLAY_COHORT`,
`BUTTERCUP_POSTERIOR_REPLAY_SEED`, `BUTTERCUP_ANNEALED_PATHS` and
`BUTTERCUP_ANNEALED_STEPS` as shown in `run_native_v3.sh`. Synthetic execution
uses `same_eye_and_complementary_stereo_annealed_reference` and
`run_synthetic_v3.sh`.

### Endpoint geometry identifies the competing iris-plane branch

The next audit records the actual weighted endpoint geometry, instead of
inferring it from marginal rejection counts. Set the test-only
`BUTTERCUP_ANNEALED_GEOMETRY_TRACE=1` to add a bounded trace to the existing
annealed diagnostic (at most 4,096 requested paths). It records the shared
target, both eye centers/normals/gazes, sampled radii, pupil offsets/depths,
surface-axis alignment, original group choices/rejections/costs and frozen
scene priors. Tracing occurs after sampling and consumes no random draws.
The independently integrated inner radius is explicitly absent from sampled
radii: its canonical placeholder and embedded prior cost must not be treated
as a draw from that omitted conditional. The trace also retains the actual
marginal log target density.

The fixed subset consists of physical reads 79, 159, 189, 249 and 255 from
`student-shadow.FkzvzE`, each replayed through both frozen SAM and Student
providers. Reads 159/189/255 are the preceding ambiguous Student examples;
79/249 are Student controls. The matching SAM results are not assumed to be
equally certain. Thus ten provider cases represent five physical reads, not
ten independent captures. All three numerical seeds use 512 requested paths
and 128 transitions; the six synthetic support cases are traced as well.

For joint classification, a state is near only if **every modeled eye** lies
within 15 degrees of its selected gaze. A retained state keeps every group
used by the selected MAP below its original rejection cap. The four near/far
and retained/rejected categories partition the weighted sample distribution.
Three-seed pooling combines unnormalized path contributions using each run's
normalizer and initial path count, including infeasible draws. Category
effective path counts measure weight concentration; they are not extra
observations or calibrated confidence levels.

| Physical read | SAM far mass retaining MAP-used groups | Student far mass retaining MAP-used groups | Student conditional effective paths in that category |
| --- | ---: | ---: | ---: |
| 79 | 8.07% | 0.30% | 2.3 |
| 159 | 18.87% | 11.47% | 13.9 |
| 189 | 3.34% | 5.80% | 6.0 |
| 249 | 8.92% | 1.04% | 7.2 |
| 255 | 8.96% | 1.81% | 8.9 |

These estimates identify competing geometry, not precise tail probabilities.
In Student read 159 the retained far states have median iris-normal shifts
61.4 and 61.1 degrees for the two eyes. Near-state medians are 4.3 and
5.0 degrees. The far states also use a different pupil offset/depth mixture;
their first-eye median inward depth is 0.297 mm versus 0.701 mm nearby.
Alignment medians remain within about one degree, so this example is a large
iris-plane branch change rather than merely a small gaze/surface-axis offset.
The native assumptions allow pupil depth 0--1.5 mm with sigma 0.5 mm and
decentration up to 0.9 mm per axis with sigma 0.35 mm. These are engineering
allowances from `approximate_scene`, not measured subject anatomy.

One high-weight retained far state in read 159 changes the two normals by
64.57 and 63.52 degrees while projecting to similar outer/pupil outlines.
The figure shows forward-projected 3D circles, not new observed pixels or
human labels. Its independent projection of the best state's outer rings
matches the evaluator's two published ellipses with dimensionless algebraic
errors below `4e-13`. This checks the visualization's coordinate conventions;
it does not turn the selected gaze into ground truth. The sampled far branch
exists with all MAP-used contour groups retained, but its mass still has weak
numerical support: Student read 159's per-seed retained-far effective counts
are 7.9, 2.6 and 16.1, despite each whole-run path check passing.

A separate, explicitly hypothetical prior sensitivity calculation multiplies
the saved endpoint weights by the new/old prior density ratio, retaining the
original support bounds. It introduces no new measurements and changes no
viewer setting. Tightening the pupil-depth sigma from 0.5 to 0.15 mm around
the original 0.6 mm nominal moves Student read 189's joint near mass from
94.2% to 84.6%, with reweighted effective counts 178.9 and 51.1. Tightening
pupil-offset sigma to 0.10 mm around zero produces 98.6% near mass but only
11.5 effective paths. Across all five Student cases that offset experiment
has only 4.9--11.6 effective paths; combining both pupil changes falls to
1.2--14.0. Apparent certainty under those changes cannot justify adopting
them. Narrowing alignment sigma to one degree also reduces weight overlap
and moves the stable read 79's near mass from 98.9% to 94.8%. Stronger
unmeasured anatomy assumptions are not a validated substitute for evidence.
All ten unchanged-prior sensitivity controls reproduce the original pooled
joint near mass exactly to numerical tolerance.

All 48 original annealed diagnostics (30 native, 18 synthetic) are identical
after removing only the added geometry trace. All 1,548 native event/source/
ordinary-posterior comparisons match apart from elapsed time, and all
eighteen synthetic ordinary controls and six selected fits match. Every
recorded gaze reconstructs from its own eye center and the **same shared
target** within `1e-10`; group cost sums, MAP group activity and exposure joins
are checked. The sampler body is byte-identical to the previous experiment.
All 148 core tests, 67 native joint-conic tests, twelve publication tests and
six stereo UI tests pass; explicitly ignored diagnostics remain ignored.
Native/evaluator builds pass and all 1,514 rebuilt production events match
the preceding build apart from elapsed time. Source-tree and diff checks pass.

No inference defaults are promoted. These reads still lack independent native
gaze truth, newly canonical-labeled contours or new scale/timing measurements,
and no true inner-limbus observation appears in either native provider.
The concrete remaining numerical issue is reliable mass estimation across
the distinct iris-plane branches; the measurement issue is resolving those
branches using supported anatomy, calibration or other independent sign
evidence. Neither issue is solved by retaining contours or selecting a
precise MAP alone. The intended partial and complementary-eye capabilities
still require matched positive/negative validation before claiming readiness.

Artifacts are in `outputs/posterior-geometry-audit-20260912`:
`reference-selection-v1.json`, all six native logs/source replays,
`synthetic-128-v1.log`, `comparison-v1.json`, `prior-sensitivity-v1.json`,
the inspected `comparison-v1.png`/`.svg`, `projection-check-v1.json`,
`production-identity-v1.json`, source/sampler hashes and both frozen
executables. `run_native_v1.sh` and `run_synthetic_v1.sh` reproduce the
diagnostics; `report_v1.py`, `prior_sensitivity_v1.py` and `plot_v1.py` audit
and summarize them. Those scripts are offline artifacts, not runtime or
training dependencies of the repository.

## Refined integration in the live distribution (2026-09-12)

The production `solve_joint_conic_distribution` entry point now selects the
coherent ConditionalRefit recipe through `IntegrationConfig::live()`. Default
integration configuration retains the original method for explicit offline
comparison. Both SAM3.1 and Eye Student reach the same production entry point;
there is no provider-specific gaze averaging or confidence multiplier.

Within the existing 8,192-draw ceiling, at most 2,048 proposal-fitting draws
select up to four additional components. Each fixes one pilot's shared target
only while refining nuisance geometry for at most six iterations. All original
components remain in the frozen proposal mixture. Pilot draws never enter
posterior estimates. Fresh final draws use the complete mixture density and
the original robust observation factors, anatomical priors and hard bounds.
The selected joint MAP and historical source handling are unchanged.

Unobserved true-inner-limbus radii are analytically integrated between the
observed pupil/outer radius constraints using their existing Gaussian priors.
An actual inner-limbus observation prevents that elimination. Retained
Student-t proposals use marginal covariance, and the additional conditional
proposals decouple target-tail scaling from Gaussian nuisance scaling. The
ordinary broad scene proposal remains in the mixture. No stronger anatomy or
new observation is inferred from a segmentation confidence score.

Four proposal-balanced batches partition the final draw stream. Their estimates
retain joint importance weights and unequal normalizers. Direction admission
requires the usual finite 90% angular radius no larger than 15 degrees, an
estimated posterior, and mass minus twice the **larger** within-stratum or
between-batch standard error reaching 90%. Early stopping uses the same
precision checks. The inspector and `admission_direction_numerics` export that
actual admission margin. These errors are numerical diagnostics with finite
sampling and adaptive stopping; they are not calibrated gaze probabilities or
guarantees against unseen modes.

`outputs/posterior-live-integration-20260912/comparison-v1.json` compares this
complete recipe with the live baseline, rather than adding only the final
margin. It verifies **9,084 exact source/geometry event comparisons** across
both methods and three fixed numerical seeds. The corpus remains 628 unique
physical reads: 129 shared by SAM and Student, plus 499 recent SAM reads.
Counts below repeat the same directions across seeds; repeats are not new
physical evidence. Both original fixed reference subsets remain visible.

| Reference subset / provider | Reference-narrow admitted, baseline → live recipe | Reference-broad admitted, baseline → live recipe |
| --- | ---: | ---: |
| 65,536 draws / SAM | 115/162 → 109/162 | 14/81 → 2/81 |
| 65,536 draws / Student | 70/81 → 66/81 | 9/66 → 1/66 |
| 65,536 draws / recent SAM | 367/654 → 615/654 | 0/63 → 0/63 |
| 1,048,576 draws / SAM | 58/66 → 54/66 | 29/57 → 6/57 |
| 1,048,576 draws / Student | 24/24 → 24/24 | 12/30 → 3/30 |
| 1,048,576 draws / recent SAM | 180/198 → 188/198 | no reference-broad directions |

The precision-qualified reference classes are also reported, without replacing
the original classes or hiding coverage losses. Neither class is native gaze
truth. The two newly admitted directions classed as broad by the million-draw
Student reference are read 159 in the second seed: their reference intervals
still cross the threshold. The previous annealed/geometry audit nevertheless
found a substantial competing iris-plane branch in that read. It remains a
specific numerical failure; the margin does not certify its absence. There
are also lost narrow directions and still-insufficient Student integrations.

The recorded shared-resource medians rise from roughly 8–16 ms to 16–27 ms per
tracker event, with extra conditional fitting work despite the same draw
ceiling. Those archival measurements are not an isolated latency benchmark.
The selected subset lacks independent native gaze truth, new canonical labels,
new independent scale support and actual true-inner-limbus observations. This
promotion therefore establishes an integration improvement, not completion of
partial-eye, cross-eye or low-light gaze-accuracy validation. The distinct
iris-plane ambiguity and real measurement/localization errors remain separate
problems to resolve using supported evidence.

Native verification uses frozen before/after executables and the complete
original cohort. All **1,514 baseline events** reproduce the preceding build;
all **1,514 candidate source/geometry events** match; all **1,479 candidate
posteriors** match the recorded experiment exactly apart from two intentionally
updated explanatory strings. Another **2,958 admission checks** independently
recompute the exported larger-error margin and boolean decision. Selected
ellipses and source-time geometry are identical, so this change cannot claim
an improvement in localization or SN-FEIDA stability.

The sequential CPU31 replay, shared at low priority, measured these tracker
event times. It excludes detector inference and is not an isolated throughput
or latency guarantee for the whole viewer.

| Provider | Median ms, baseline → refined | 95th percentile ms, baseline → refined |
| --- | ---: | ---: |
| SAM | 8.42 → 19.09 | 23.22 → 53.14 |
| Student | 8.19 → 24.58 | 31.37 → 56.74 |
| recent SAM | 12.44 → 22.10 | 28.81 → 45.07 |

One native publication test initially failed because it presumed a perfect
synthetic outer/pupil pair under coarse anatomy priors had narrow support.
Matched 65,536- and 1,048,576-draw integrations kept its evidence, priors,
selected geometry and three seeds fixed. All six million-draw runs classify
it as broad: baseline near mass is 84.7–89.2%, refined near mass 87.3–89.6%.
The test now preserves the actual coarse-fixture abstention and checks the
publication contract with explicit posterior doubles, including precision,
angular spread, missing batches and insufficient sampling. Separate conic
tests exercise actual production-entry positive and ambiguous integrations.
The ignored `synthetic_publication_posterior_reference` diagnostic reproduces
the larger comparison; no anatomy prior or runtime gate was loosened to pass.

Final checks pass: 79 native conic tests, 12 publication tests, six stereo UI
tests and 14 source-history tests, plus native viewer/evaluator builds, source
tree audit and diff checks. Explicit diagnostics remain ignored in routine
runs; the publication reference was run separately at both budgets. The
earlier core check passed 148 tests before the native fixture expectation was
corrected. Current artifacts include `production-verification-v2.json`,
`publication-reference-summary-v1.json`, all six frozen native replays,
`native-checks-v2.log`, `integration-production-v2`, the inspected comparison
figure, and the source snapshots/hashes in the integration output directory.

## Native calibration after the integration change (2026-09-12)

`outputs/posterior-calibration-admission-20260912` carries the refined
integrator through the actual native bridge, RAW boundary extraction,
source/prompt/crop/clock checks, paired completion, sign acquisition and the
enclosing calibration phase machine. The explicit baseline is the original
probabilistic integrator, not the earlier no-posterior control. Original
recorded target positions enter only the separate monitor-fit scorer.

The cohort is the same three SAM3.1 sessions ending in
`1788826709-274018364`, `1788827178-325761427` and
`1788827261-537156697`: **499 physical reads / 998 ROI exposures**.
Each integration arm replays those exposures for both calibration focus eyes,
producing 1,996 diagnostic rows per arm, not more physical evidence.
`matched-calibration-report-v1.json` verifies the six input hashes, exact
recorded completion schedule and all **1,996 matched row pairs**. Selected
geometry and source metadata, including missing publications, are identical
after excluding the posterior diagnostic. The baseline also reproduces the
previous original-integrator target counts, estimates and fit outcomes exactly.

Qualifying sources below require both native frame admission and completed
acquisition. Repeated publications of one source never add votes. The times
are offsets in the frozen recorded arrival schedule; they do not include the
refined integrator's additional measured CPU work or simulate a new user's
waiting time.

| Clip / focus ROI | Unique qualifying sources, original → current | First acquisition ready, original → current | Final plane + affine with shared support |
| --- | ---: | ---: | --- |
| Dark / 1 | 59 → 102 | 2,724 → 1,650 ms | fails → fails |
| Dark / 2 | 62 → 104 | 2,724 → 1,650 ms | fails → fails |
| Middle / 1 | 116 → 163 | 5,259 → 2,664 ms | passes → passes |
| Middle / 2 | 117 → 159 | 5,259 → 2,429 ms | passes → passes |
| Bright / 1 | 94 → 102 | 1,544 → 1,544 ms | passes → passes |
| Bright / 2 | 95 → 103 | 1,317 → 1,317 ms | passes → passes |

The current dark replay supplies at least seven qualifying sources at every
target, and all nine produce native stable-cluster estimates. This recovers
coverage but not the final fit. The current middle replay still lacks a stable
first target; the remaining eight provide the required distributed coverage.
Across the six focus-eye replays there are 212 gained and 22 lost qualifying
source observations. Do not describe the increase as universal retention or
as additional independent physical reads.

Stored `source_window_ns` starts before target settling. The 500 ms scoring
variant matches the native source-settle requirement; zero ms is a deliberate
pre-settle sensitivity control. The current dark ROI 2 passes a zero-settle
final fit and a 500 ms first-stable-prefix fit in the original diagnostic, but
fails the 500 ms final-window affine fit. Neither alternate result replaces
that failure or proves what a differently timed live session would display.

The timing audit found that inclusive zero-settle boundaries could assign a
transition source to two consecutive targets. The test-only scorer now uses
`start < source <= end`, retains the positive settle condition, and asserts
that one source cannot vote in two windows for the same eye. This matches the
fact that the saved start is the current sensor frame at a target transition.
The production calibration already requires the positive source-time settle.
Original inclusive-control artifacts remain as dated diagnostics, with their
reused memberships explicitly counted in `target-separation-v1.json`.
All 12 native scorer runs pass after the correction. Independent source-set
checks remove 36 reused target memberships across the zero-settle controls;
every 500 ms target estimate, prefix diagnostic and fit result remains exactly
identical. Source hashes verify that this edit changes only the test module,
and the source-tree and diff checks pass. The corrected native results and
frozen test executable are under `windows-v2`.

The remaining dark failure is not an affine candidate rejected solely by gain
or conditioning. An independent least-squares enumeration reproduces all 48
native acceptance decisions across clips, arms, settles and prefix/final
estimates. Neither dark final fit at 500 ms has a candidate satisfying the
existing distributed-coverage and residual requirements. This is an
exploratory numerical cross-check; native Rust fitting remains authoritative.

For dark ROIs 1/2, median native within-target cluster RMS is **3.75° / 3.14°**,
while median separation of neighboring target estimates is **3.69° / 3.38°**.
Corresponding within-cluster values are 0.96° / 1.18° in the middle clip and
1.70° / 1.88° in the bright clip. These are observed direction variations,
not errors against independently measured fixation. All 87 qualifying dark
ROI 1 target-window sources use same-eye pupil factors; ROI 2 uses them in
75 of 89. The failure therefore cannot be explained simply by missing pupil
support. Boundary localization, motion and the anatomical projection model
still need to be distinguished with independent evidence.

No production geometry, anatomy prior, probability threshold or calibration
acceptance limit changed in this audit. There is no SN-FEIDA improvement claim
from identical conics, no new scale measurement, no independent native gaze
truth and no actual true-inner-limbus observation in this subset. Eight dark
native RAW triplets remain prepared in the canonical annotator with recorded
predictions hidden; all were unreviewed when checked. These three Rob-only
clips do not validate cross-user or student-model calibration accuracy.

Reproduction and inspection artifacts include `audit_v1.py`,
`matched-calibration-report-v1.json`, `affine-diagnostic-v1.json`,
`target-separation-v1.json`, the inspected `calibration-coverage-v1` and
`dark-target-directions-v1` PNG/SVG figures, and `verify_windows_v2.py` for
native source-ownership verification. These are offline artifacts beneath
the checked outputs link, not runtime or training dependencies.

## Separating contour motion from a supported branch switch (2026-09-12)

`outputs/stereo-direction-motion-20260912` joins the current qualifying
target-window sources to their exact cached detector conics. It retains the
native 500 ms settle and compares only source-time transitions inside one
recorded target window; gaps above 500 ms remain in the export but are excluded
from the short-gap summary. No fit, prior, threshold or gaze output changes.

The input ellipse supplies two unordered camera-facing circle-normal
hypotheses under the existing intrinsics and circularity assumptions. Their
minimum inter-frame angle is a lower bound on input orientation change, not
an independently resolved sign. The independent eigendecomposition reproduces
the native projected outer-circle normal in 1,314 coordinate checks, with
maximum disagreement about `1.1e-13` degrees. This checks the model and coordinate
conventions, not physical camera calibration or contour accuracy.

Median selected-gaze steps in the dark clip are 4.87° / 3.84° for ROIs 1/2;
the corresponding minimum input-outer-normal steps are 5.83° / 4.42°. The
bright clip's values are 0.91° / 0.89° and 0.76° / 1.01°, respectively. Thus
substantial dark-case variation already exists in the fitted detector conics.
These are observed variations, not ground-truth gaze errors. A recovered pupil
search ellipse can inherit the outer ellipse's ratio/orientation, so its
decomposed normals are not an independent pupil-orientation measurement.
The native pupil factors still use newly extracted RAW boundary samples.

There is a separate, concrete model failure candidate in middle-session
sequence 160. The first eye's admitted direction changes 85.47° from sequence
159 over 201.81 ms, then returns about 84.83° at sequence 162. Both eyes share
the switch through one fitted fixation. The outer hypothesis sets change
about 6.9° / 14.7° across 159→160; the selected solution changes iris-plane
branch. The calibration's dominant-cluster reducer excludes this sample, so
the session can pass its monitor fit while retaining this native outlier.

Three fixed numerical seeds at **1,048,576 draws per selected paired read**
test sequences 159, 160 and 162. All 998 source exposures remain available as
initialization context. There are three selected physical reads, not nine new
observations. The comparison verifies 117 geometry fields and 99 RAW boundary
factors exactly against the live bridge before using any posterior result.

| Sequence | Current mass within 15° of selected direction | Million-draw range over three seeds |
| --- | ---: | ---: |
| 159 | 98.59% | 97.38–97.72% |
| 160, switched branch | 97.92% | 93.89–95.72% |
| 162 | 99.02% | 99.43–99.65% |

Both eyes remain numerically supported in all nine paired references. Even
sequence 160's lowest approximate two-standard-error mass bound is 91.07%,
and its reference 90% angular radius is 6.31–6.78°. This does not prove correct
gaze. It shows that merely spending more integration work does not remove
this switch under the current observation factors and unmeasured anatomy.
The earlier angular-continuity counterexamples still rule out enabling a
sign lock or smoothing prior solely because it removes the jump.

An initial reference attempt selected provisional first-eye events 571/573/577
from the source-order evaluator. It is explicitly excluded in its manifest:
the native completion order differs from that evaluator's event order. The
verified paired events are 572/574/578. Their target, normals, gaze, costs and
factor identities match exactly; no provisional-only result supports the table.

Existing RAW motion records were also matched by complete frame metadata,
clock lineage, raw offset/length and SHA-256. Exterior-of-limbus motion is
unreliable for **both** eyes at sequence 160. Whole-ROI similarity is available,
but cannot substitute for an independently measured head/pivot displacement.
Missing exterior support must contribute no identity transform or stationary-
head claim. Both native before/target/after RAW triplets at sequence 160 are
prepared beneath that capture's `annotator/archive`, with predictions absent
and human labels confined to its `annotator/labels`. The existing dark review
remains a separate pending batch.

Artifacts include `direction-motion-v1.json`,
`paired-reference-comparison-v2.json`, `source-matched-motion-v2.json`,
the inspected `branch-reference-v2.png`/`.svg`, exact numerical manifests and
logs under `original-1m-v2`, and `annotation-preparation-v1.json`. The scripts
retain source hashes and all missing/large-gap cases. No new human labels,
measured fixation, independent head motion, scale, camera intrinsics, true
inner-limbus observations or Student calibration data were introduced. Neither
this diagnostic nor the unchanged conics demonstrate improved localization,
SN-FEIDA stability or native gaze accuracy.


## Native factor attribution and isolated pupil removal (2026-09-12)

`outputs/stereo-branch-factors-20260912` separates the exact native residual
costs of the selected branch from the lowest-cost returned alternative more
than 30 degrees away. This is a test-only diagnostic: it consumes each native
residual exactly once, including quadrature weights, correlated alternatives,
rejection caps and separate priors. It does not estimate the objective from
the exported unweighted pixel RMS or treat score differences as calibrated
probabilities. Production builds exclude the diagnostic field and helpers.

The same 258 SAM, 258 Student and 998 recent source events are replayed.
All **1,514** source/geometry/publication records match the promoted integrator
replay after excluding only posterior diagnostics and elapsed time. These
remain 628 unique physical reads; the SAM/Student shared clip and provisional
publications are not extra observations. Independent reconstruction checks
3,861 returned hypothesis costs and 89,012 factor records; maximum total-cost
disagreement is `2.85e-13`. The 67 joint component tests and six native stereo
UI tests pass; explicitly ignored corpus diagnostics were run separately.

For the middle clip, the signed table entries are alternative minus selected
cost. They attribute the optimized score gap, not integrated posterior mass.

| Sequence | Outer boundary contribution | Pupil boundary contribution | Scene/anatomy priors | Total gap |
| --- | ---: | ---: | ---: | ---: |
| 159 | 0.042 | -0.039 | 2.360 | 2.363 |
| 160, switched branch | 1.024 | 0.341 | 1.031 | 2.396 |
| 162 | 0.097 | 4.853 | 4.166 | 9.116 |

At 159, pupil decentration/depth priors dominate the score difference; the
actual pupil residual slightly favors the other branch. At 162, ROI 1's
pupil boundary contributes 4.843 of the 9.116 cost gap. ROI 1 has no cached
pupil guide at 159/160 and therefore contributes no pupil samples there.
The cache does not preserve the semantic decision reason for that absence;
a stateless RAW-component diagnostic cannot reconstruct that decision.

A second native replay removes pupil samples and initialization hints from
each eye separately, or both, **only for the current read** at 159/160/162.
The original full-evidence scene priors and preceding-source seeds remain
frozen. These counterfactuals never update tracker history. All 1,514 original
history records remain identical, all three full-evidence controls reproduce
the original posterior exactly, and both empty ROI 1 removals are exact no-ops.

The table shows ROI 1 model mass within 15 degrees of each variant's own
selected ray, using the default numerical seed. Both eyes have the displayed
admission outcome. This is conditional model support, not measured accuracy.

| Pupil evidence retained | 159 | 160, switched branch | 162 |
| --- | --- | --- | --- |
| All available | 98.6%, admitted | 97.9%, admitted | 99.0%, admitted |
| Only ROI 1 | 86.4%, withheld | 94.2%, admitted | 97.8%, admitted |
| Only ROI 2 | 98.6%, admitted | 97.9%, admitted | 95.3%, admitted |
| None | 86.4%, withheld | 94.2%, admitted | 36.5%, withheld |

Removing every pupil at 160 changes the two selected directions by only
0.041/0.100 degrees. Its approximate numerical lower mass bound is 90.11%,
so the single-seed counterfactual still passes the current gate. This is not
a new large-reference certainty claim. At 162 either eye's pupil alone keeps
the full-evidence branch within 0.28 degrees, while removing both changes the
two selected directions by 75.66/72.09 degrees and makes admission fail.
Thus pupil evidence is useful in this model, but blanket pupil removal does
not repair the isolated switch. Outer-contour uncertainty and binocular scene
assumptions remain the next targets for investigation.

The scope is three Rob-only physical reads with twelve current-read controls,
not a changed-history ablation, temporal area success or new gaze truth.
Native localization labels, current scale/head motion and measured intrinsics
remain missing for these comparisons. The canonical dark review was checked
live again: all eight targets remain unreviewed, three native context frames
each, recorded predictions hidden. Six RAW pieces for the separate middle
branch review were independently byte-verified against their native archive;
the four offered-cache pieces additionally match exact cache metadata/hashes.
Neither review was replaced or labeled by the agent.

Reproduction uses the ignored Rust `recorded_native_factor_diagnostic` test.
Set `BUTTERCUP_FACTOR_DIAGNOSTIC_DIR` to a new directory beneath outputs;
optional `BUTTERCUP_PUPIL_COUNTERFACTUAL_SOURCES` names the exact lineage/time
selection JSON. The matched artifacts are `matched-factor-audit-v2.json`,
`current-pupil-ablation-v1.json`, `counterfactual-selection-v1.json`, frozen
executables/logs, and the inspected `branch-pupil-attribution-v1.png`/`.svg`.
No geometry, probability gate, training default or live camera setting changed.

## Native scene sensitivity and source optical evidence (2026-09-12)

`outputs/stereo-scene-sensitivity-20260912` tests thirteen fixed scene and
outer-allowance variants on twelve preselected physical reads: nine from the
dark/middle/bright calibration clips and three shared SAM/Student reads. These
are fifteen provider/read cases, not fifteen independent captures. The full
1,514-event history stays identical to the preceding native factor replay;
all fifteen unperturbed posterior controls match the promoted integrator.
The 195 solutions pass 390 independent checks that both eye rays point at
their one shared latent fixation. No candidate seeds subsequent history.

These are sensitivity probes, not calibrated camera/anatomy confidence ranges.
Focal length changes recompute camera-dependent metric initialization from the
same native pixel/scale inputs. Other factors, pupil evidence, anatomy bounds,
preceding-source seeds and the numerical recipe stay fixed. Outer multipliers
change only outer localization and contour-band allowances; pupil allowances
do not inherit them. The table summarizes the middle clip's switched frame 160:

| Perturbation | Largest eye-direction change | Current numerical admission |
| --- | ---: | --- |
| Both focal lengths ±10% | 1.414° | Both eyes admitted |
| Principal point ±80 px X / ±60 px Y | 0.456° | Both eyes admitted |
| Opposing focal-axis changes ±5% | 6.129° | Both eyes admitted |
| Outer allowances ×1.5 | 0.403° | Both withheld |
| Outer allowances ×2 | 0.956° | Both withheld |
| Alignment sigma ×1.5 | 0.138° | Both admitted |
| Alignment sigma ×2 | 0.246° | Both withheld |

Every tested variant retains the switched branch. Increasing outer allowances
reduces frame 160's conditional near-ray mass from 97.9% to 80.4% / 74.0% at
the live sampling budget, but loses six / fourteen supported eye directions
across the fifteen cases. These admission changes use one numerical seed;
they are not larger-reference probability estimates or accuracy improvements.
One focal-aspect probe also moves the older Student result at sequence 4434
by 30.14°. This strengthens the need for measured camera parameters rather
than establishing a replacement camera model. A filename inventory found no
intrinsic/calibration JSON in the checked runtime roots; that inventory does
not cover embedded archive metadata or external calibration stores. The middle
archive explicitly records camera intrinsics as unavailable, and its metadata
does not provide exposure-bound lens-position/settling measurements.

The exact RAW frames show a conspicuously blurred frame 160 between sharper
159 and 162. An ignored native viewer diagnostic now recomputes the existing
`measure_limbus_optical_focus` and `provisional_focus_score` on the source bytes,
without using joint results, targets or labels. All 1,514 source records match
their frozen caches; the 258 matching SAM/Student exposures give identical
RAW-only provisional scores. The conic-guided optical metric may differ by
provider because its lateral sampling positions depend on the upstream conic.

| Native source | ROI 1 edge concentration | ROI 2 edge concentration |
| --- | ---: | ---: |
| 159 | 0.366 | 0.414 |
| 160 | 0.135 | 0.238 |
| 162 | 0.395 | 0.409 |

The provisional scores at 160 also fall to 78.5% / 79.3% of their preceding
values. However, the dark clip's median edge concentration is only 0.141.
These are engineering optical measurements, not calibrated boundary variance;
a single cutoff would confound this dark-scene case with the isolated blurred
source. The useful next experiment is source-dependent outer evidence
uncertainty, checked against localization, dropouts and the same-eye and
complementary-support cases. It must retain independently sharp pupil/inner
evidence and avoid making its uncertainty inherit a weak outer contour.
The earlier blanket RAW weighting regression still applies.

No new human labels, measured fixation, fresh independent scale, true-inner
observations or current RAW Student model results were added. The original
SN-FEIDA outputs remain diagnostics on the existing acquisition scale; neither
this optical audit nor withholding a blurred solve proves an area/localization
improvement. The native model and live viewer were not changed or restarted.

Reproduce the scene variants with `recorded_native_factor_diagnostic`, setting
`BUTTERCUP_SCENE_SENSITIVITY_SOURCES` to the exact lineage/time selection JSON.
Reproduce optical measurements with the ignored viewer test
`native_stereo_optical_focus_diagnostic`, `BUTTERCUP_OPTICAL_AUDIT_CACHES` (a JSON
array of cache paths), and a fresh `BUTTERCUP_OPTICAL_AUDIT_REPORT` under outputs.
The 67 joint tests, both explicit native corpus diagnostics, four optical
component tests, production evaluator check and source-tree audit pass.
Matched reports are `matched-sensitivity-v1.json` and
`matched-optical-audit-v1.json`; frozen executables and logs accompany them.
The inspected comparison is `optical-branch-v1.png`/`.svg`. The corrected RAW
preview is `observed-boundaries-v2.png`; `preview-provenance-v2.json` explicitly
invalidates the initial Python preview's incorrect MIPI packing assumption.

## RAW outer transition width versus boundary position

Two explicit offline experiments separate optical transition width from
displacement relative to the detector's original contour points. Neither is
enabled in `JointTracker` or the live viewer. Both preserve all pupil/true-inner
factors, contour positions, conic hints, correlation groups and source timing.
They do not smooth gaze, average eye solutions or normalize by a candidate radius.

`--raw-outer-spread` measures the gradient's second moment about the observed
point, using 17 normal samples over ±16 native pixels and the median of three
tangential profiles. It replaces only an outer arc's optical allowance with
the larger of its existing allowance and the 75th-percentile measured spread.
This **fails validation**: even an ideal sharp synthetic eye receives about
2.30 px instead of its original 0.75 px optical allowance, and its otherwise
supported complete outer-plus-pupil direction becomes unsupported in all
three numerical seeds. The width describes the optical transition and sampling
filter, not the variance of the boundary's position.

`--raw-outer-position` instead measures the signed gradient centroid relative
to the original point, while exporting the centered optical spread separately.
Only the 75th-percentile absolute displacement can widen the old allowance.
Both experiments require a measurable rise of at least 7 RAW10 levels,
net/total variation of at least 0.75, and at least three valid profiles covering
half the requested samples. Position additionally rejects a window-censored
transition when either endpoint gradient exceeds 20% of the peak. Unknown is
reported and retains the original engineering allowance; it is not measured
precision. No fitted ellipse, solver residual, temporal source or label enters
these measurements. Neither recipe supplies calibrated localization covariance.

The matched replay covers 1,590 source events per arm: SAM and the older Student
each see the same 129 physical reads, three calibration clips contribute 499
reads, and the reviewed subset contributes 49. There are 677 unique physical
reads and 1,332 distinct native ROI byte ranges; all 1,590 cache receipts match
their RAW SHA-256 values. Each arm updates its own full source-order history.
Counts below use the final publication once per provider/source, including
unavailable outputs, rather than counting provisional revisions as new evidence.

| Provider/subset | Baseline supported eye directions | Width candidate | Position candidate |
| --- | ---: | ---: | ---: |
| SAM, 129 reads | 80 | 7 | 78 |
| Older Student, same 129 reads | 74 | 6 | 74 |
| Calibration, 499 reads | 791 | 626 | 789 |
| Reviewed, 49 reads | 13 | 9 | 15 |

Width loses 325 previously supported directions and gains 15. Position loses
eight and gains six. More retained directions do not establish greater accuracy.
Position obtains an arc-level measurement on only 529 of 3,309 outer arc
observations; 2,780 remain unknown, predominantly because transitions are
nonmonotone or window-censored. These are correlated arc/provider observations,
not 3,309 independent trials. The six measurable reviewed arcs are particularly
weak coverage for validating an uncertainty model.

Of sixteen existing reviewed labels, ten match this stereo RAW subset and
eight have available fits in all arms; the two missing low-light fits stay
missing. Width worsens six of the eight visible-boundary RMS errors, improves
one and leaves one unchanged; the largest regression is 4.538 px at sequence
247. Position leaves seven unchanged and improves that one fit by only 0.052 px
(3.780 to 3.728 px). The baseline's 29.060 px error at sequence 317 is unchanged
by position. This is not evidence of general localization improvement.

At the middle clip's source 160, position leaves both the selected rays and
the 97.9% conditional near-ray mass exactly unchanged; the approximately 85°
switch remains admitted. Width keeps the switched branch but withholds it.
Source 159 is also exact under position, while 162 changes by only
0.022/0.048°. No new independent gaze truth was obtained. Common fresh
SN-FEIDA log-step comparisons use the original acquisition scale and gaps of
at most 500 ms: width generally worsens median steps; position is unchanged
or slightly worse in the three temporal cohorts. The reviewed subset has no
eligible adjacent pairs. Neither establishes area accuracy or improved motion
alignment without independent scale support.

The expanded synthetic audit has six cases—complete and partial outer-plus-
pupil, complete and partial outer-plus-true-inner, complementary stereo, and
outer-only—at three seeds. Width has 72 case/optical-arm/seed rows. Position has
90, adding an optical edge displaced eight pixels from the unchanged measured
contour. Position retains the ideal complete-pupil admission for sharp, dim and
centered blurred edges, but withholds the displaced case. The partial,
complementary and true-inner fixtures remain unsupported under the live
integration recipe even at an exact synthetic MAP; these are unresolved or
broad distributions, not demonstrated successful gaze cases. Outer-only mirror
ambiguity also remains unsupported. A zero MAP error is not sufficient proof
of probabilistic support or of performance on real occlusions.

All 67 joint tests and nine RAW allowance tests pass. The initial position unit
run exposed a too-tight incidental subpixel assertion: integer quantization of
the dim RAW edge shifts its centroid by 0.267 px. The corrected test checks the
existing 0.75 px engineering floor; no candidate algorithm or native threshold
changed. Failed v1 artifacts remain alongside the successful v2 run.

Reproduce with `outputs/outer-raw-spread-20260913/run_v1.py` and
`outputs/outer-raw-position-20260913/run_v2.py`; the latter verifies native RAW
hashes before replay. The corresponding ignored Rust diagnostics are
`raw_outer_spread_support_cases_diagnostic` and
`raw_outer_position_support_cases_diagnostic`. Frozen executables, exact cache
lists, matched audits, canonical-label scores and source receipts accompany
both experiments. The inspected comparison is
`outputs/outer-raw-position-20260913/optical-position-comparison-v2.png`/`.svg`.
The width candidate is rejected and the position candidate remains diagnostic
only. The next model needs explicit treatment of ambiguous or missing boundary
location evidence; neither optical width nor a centered local gradient is a
substitute for it. Training, camera ownership and live gaze behavior are unchanged.

### Selected-mask logit boundary receipts

`BUTTERCUP_OUTER_BOUNDARY_LOGITS=1` now retains source-native boundary profiles
from the actual selected SAM video mask or Student outer-mask head. The default
is off, before any extra tensor transfer. Both existing replay exporters append
`outer_boundary_logits` only when a selected fitted mask supplied the evidence.
The live conic likelihood does not consume this diagnostic field yet.

The initial v1 exporter in `src/sam31_boundary_logits.rs` sampled 33 positions from −16 to +16 native pixels
along normals formed by adjacent measured points within each retained run.
Run endpoints were omitted so a tangent could not bridge an occluded gap. Sampling
uses the pixel-center transform `(x + 0.5) * mask_width / source_width - 0.5`
and its independent vertical counterpart, with edge clamping inside the source.
The packet records native source identity, sensor origin, generation, published
query, arc and point indices, original point coordinates, logits and every
crossing of levels −1, 0 and +1. Missing samples, multiple crossings, threshold
plateaus and budget exclusions remain explicit. That version used at most sixteen
profiles per arc and 256 per source. Sampling follows the published contour
after any explicit refinement and records whether refinement was applied.

These are sensitivity alternatives of **one selected mask**, not additional
independent observations or calibrated probabilities. In particular, the width
between two logit levels is not automatically localization variance. Contour
selection, RAW gates, pupil history and pupil level-set selection are unchanged.
That export-only experiment did not insert mask-level alternatives into the gaze
solver or average another eye's independently solved direction.

The frozen experiment is `outputs/outer-logit-boundaries-nlOH4wff/`. Four matched
on/off replays cover 662 provider/source events per arm: SAM sees the complete
129-read shadow recording, the 35-read completed-worker prefix of the middle
calibration clip through sequence 162, and 49 reviewed-context reads; CPU-only
Student sees the same complete shadow recording. This is 213 distinct physical
reads and 404 distinct native ROI byte ranges, all SHA-verified. The middle
prefix is the previous offered worker's completed exposures, not every camera
frame. Replays here are completion-paced and do not measure offered latency.

All 662 on/off records have exactly the same retained/censored contours,
ellipses, pupils, admission decisions and source provenance. All 17,328 sampled
points exactly match their published native contour coordinates. The 404 SAM
results also exactly match the corresponding older cached geometry and pupils.
The current CPU Student matches 103 of 258 older Student cache records exactly;
the remaining historical differences are separate from the diagnostic toggle.
Its on/off comparison is exact on all 258 frames, using the original pinned
`b780957a…334cf3de` model, with no training or promotion.

| Provider/subset | Frames with a fit | Profiles | Single-band profiles | Median −1/+1 width |
| --- | ---: | ---: | ---: | ---: |
| SAM calibration prefix | 70/70 | 1,832 | 1,721 | 3.38 px |
| SAM reviewed contexts | 62/76 | 1,956 | 1,156 | 4.43 px |
| SAM shadow recording | 242/258 | 6,764 | 5,495 | 2.92 px |
| CPU Student, same shadow recording | 243/258 | 6,776 | 6,458 | 1.81 px |

At middle-clip ROI 1 sources 159, 160 and 162, median widths are respectively
3.50, 6.14 and 4.92 px. Source 160 therefore exposes broader mask sensitivity
at the existing suspicious gaze switch. This does not repair the switch or
establish that mask width is a calibrated error estimate. On ten exact native
human-label matches, 43 visible labels fall within two tangential pixels and
sixteen normal pixels of an actually sampled profile: nine are inside its
single −1/+1 band, 27 outside, and seven have ambiguous/missing bands. These
local, correlated comparisons are sparse and selected; they are not a coverage
calibration. They demonstrate residual model discrepancy that raw logit widths
alone cannot be assumed to explain. Unlabelled SAM/Student disagreement is
also not ground truth, and their different activation scales cannot be ranked
as confidence without validation.

`build_v3.py` records fourteen passing profile, source-publication, pupil-levelset
and replay-evidence tests, the production replay build, viewer compile check,
tree audit and whitespace check. Earlier failed checks are retained: v1 used a
utility test crate missing existing shared-test modules; v2's new fixture
compared native points against pre-conversion model points. The corrected
fixture retains exact equality after the existing conversion; runtime logic
and tolerance were not changed. `prepare_v1.py`, `replay_v1.py` and
`analyze_v1.py` record asset hashes, native receipts, exact comparisons and
post-inference label joins. The inspected figure is
`boundary-logit-evidence-v2.png`/`.svg`.

The following experiment consumes these alternatives while retaining the
original model-discrepancy allowance. Exporting them alone does not establish
gaze accuracy, cross-user readiness or anatomical truth.

### Correlated mask-level likelihood: offline candidate

`buttercup_stereo_conic_eval --mask-levels` attaches selected-mask evidence to
the existing native outer arcs before optional training/validation decimation.
Use it with `--source-order-replay --all-boundary-samples --probabilistic` to
compare full source history. The live adapter still leaves this field absent.
The experiment and frozen executables are under
`outputs/mask-level-joint-pQWa7TJ7/`.

The v2 profile export keeps the 256-profile source budget, prioritizing the exact
sixteen samples per run that the native adapter retains before using remaining
space for inspection profiles. Endpoints use one-sided, within-run tangents.
`run_point_index` distinguishes repeated coordinates at a closed run's ends.
The receiver checks source identity, crop, selected query and native coordinates;
missing profiles, nonunique crossings and threshold plateaus supply no invented
alternative. Version 1 receipts remain readable with their limited coverage.

Levels −1/0/+1 define three sensitivity states for one source mask. Their
displacements are relative to that profile's zero-logit crossing; the central
state preserves the original raster contour exactly. All profiled groups of
the same eye and boundary share one state. Each state evaluates the same joint
fixation, original point mass, localization allowance and robust group cap.
Pupil and true-inner factors remain separate. With state cost `E_s`, the mask
contribution is `−2 log(mean_s exp(−E_s/2))`. The equal state weights are declared
engineering assumptions, not calibrated segmentation probabilities.

The residual representation uses conditional state weights plus their KL
penalty. Only local optimization/proposal derivatives freeze those weights and
arc activity; actual trial and posterior densities recompute the full mixture.
The frozen-state metric is not reported as a marginal Gaussian covariance:
`local_uncertainty` reports `mask-level-mixture-requires-distribution` instead.
The full distribution remains responsible for directional uncertainty.

Current validation includes 173 ordinary offline tests, nineteen selected-mask
publication/export tests in the SAM-enabled viewer build, and ninety synthetic
results covering complete outer+pupil, complete outer+true-inner, partial
same-eye support, complementary eyes and outer-only ambiguity. Three displaced
outer-only controls have no feasible fit and are recorded as dropouts. In the
complementary displaced case, the best-fit error falls from about 6.9 degrees
to zero, but no seed passes the directional uncertainty gate. Strong complete
pupil support remains usable; broad partial cases still have sampling failures.
An accurate best fit is not sufficient evidence of a useful gaze distribution.

New SAM/CPU Student inference on the same four subsets preserves all 662 prior
control records exactly, including contours and pupils. All 54,509 exported
profile points join their native sources; all 18,183 solver-retained points have
profiles, of which 15,102 provide usable three-level alternatives. There are
404 distinct SHA-verified ROI byte ranges. The current solver with the flag off
matches its frozen predecessor on all 662 source events, excluding timing.
Candidate/control comparisons preserve 6,890 common factor masses and verify
1,836 eye rays against their one shared fixation.

| Subset | Final directions passing the existing gate, control → candidate |
| --- | ---: |
| SAM calibration prefix | 48 → 50 |
| SAM reviewed contexts | 13 → 9 |
| SAM shadow recording | 80 → 76 |
| CPU Student shadow recording | 75 → 68 |

These are final publications once per provider/source, not held-frame votes.
The total falls from 216 to 203, with 33 gains and 46 losses; insufficient-sampling
results rise from 27 to 38. Of eight fitted exact-RAW human-label matches, five
improve, two worsen and one is unchanged; two additional matched labels remain
without a fit. The calibration prefix's ROI 1 jump at source 160 remains large:
85.47 degrees in the control and 84.25 degrees in the candidate. Median common
source-to-source SN-FEIDA log changes worsen in all three time series. The
Student 95th percentile improves while its maximum worsens. The reviewed contexts lack usable consecutive-scale
pairs. Original acquisition scale hints are preserved; no new metric head scale
or measured gaze truth is available.

This candidate is **not promoted**. The following experiment checks coverage
of alternate mask states against these same native and synthetic inputs.
Current results do not establish calibrated confidence, reliable
partial/true-inner gaze, or a repair of the native branch switch.

### Conditional mask-state proposals: offline numerical experiment

`outputs/mask-state-proposals-okq3cr2j/` freezes the implementation, inputs,
matched audit, independent integration references and inspected native ROI
figure. `IntegrationConfig.mask_state_proposals` defaults to false, including
in `live()`. The live adapter still does not attach mask profiles.

The proposal builder conditions a temporary model on each coherent mask-level
assignment and refines from up to two existing joint basins. It covers all
assignments for one or two mask families, with a bounded sixteen-state subset
for larger families. Each family changes together; it never lets each contour
fragment independently choose its preferred mask. Two scalar radius steps
initialize displaced states that might otherwise start beyond a robust group's
zero-gradient cap. These steps and subsequent conditional fits only locate
sampling proposals. Every pilot and estimation draw still uses the original
full marginal target and complete frozen proposal-mixture density. No completed
ellipse hints, observation weights, priors, source-history votes or independent
eye targets are added. Conditional models preserve the original polyline
quadrature and mass even when sample coordinates move.

The default and no-profile candidate match the previous production control
exactly on all 662 source events, excluding elapsed time and test-only factor
attribution. The masked control matches the preceding mask experiment exactly.
The new masked proposals preserve all 662 fitted geometries, source identities
and factors. Running the independent reference beside them leaves their normal
sampling output unchanged on those same 662 events.

| Subset | Final supported eye directions, masked control → new proposals | Insufficient sampling, control → proposals |
| --- | ---: | ---: |
| SAM calibration prefix | 50 → 58 | 4 → 0 |
| SAM reviewed contexts | 9 → 13 | 5 → 5 |
| SAM shadow recording | 76 → 106 | 7 → 3 |
| CPU Student shadow recording | 68 → 84 | 22 → 23 |

The total changes from 203 to 261, with 74 gains and 16 losses. These counts are
conditional engineering admission decisions, not accuracy improvements. Median
elapsed times increase despite fewer median sampling draws, because conditional
proposal fits add work; the shared-host timings are not an isolated benchmark.
Sixty-four final publications have a conditional proposal center with lower
full marginal cost than the selected bounded-search fit. These are recorded
search limitations; this proposal-only experiment deliberately does not replace
the selected fit. Human-label localization, coverage and SN-FEIDA therefore
remain identical to the previous mask candidate, including its regressions
against the nominal-boundary control. No new metric scale or gaze truth was
introduced.

The ninety synthetic results retain the same six cases, five boundary variants
and three seeds. All geometry and availability results match; the three
displaced outer-only dropouts remain. Sampling failures decrease from six to
one. Complete same-eye pupil support remains admitted, while the ambiguous
true-inner, partial and complementary cases do not acquire spurious support.
This verifies numerical behavior on those fixtures, not reliable solutions for
every requested partial/true-inner scenario. There are 175 passing ordinary
offline tests, six passing stereo viewer tests, and a passing SAM-enabled
production viewer check and tree audit.

Independent annealed integration uses two seeds, 4,096 paths and 128 bridge
steps on eleven selected native reads. The selections include SAM support
gains/losses, Student gains/losses and the calibration jump. Most selected SAM
gains agree with these model references, but SAM source 4419 varies by seed and
source 162 has one insufficient-path result. The Student source 4415 ROI 2 gain
is suspect. Increasing its references to 16,384 paths and 256 steps estimates
15-degree model mass at 0.9103 ± 0.0299 and 0.8755 ± 0.0385 (twice numerical
standard error). Both fail the existing conservative 90% admission margin.
The reference paths use independent random streams and invariant transitions,
but share the target and proposal family; they cannot certify unvisited modes
or calibrated gaze accuracy.

Source 160 still switches by about 84 degrees, and both independent references
place about 97.6–97.7% model mass near that selected ray. Removing all pupil
evidence from that current read preserves the opposite branch and its tight
support. Thus neither sampling noise nor optional pupil evidence alone explains
this failure. The six SHA-verified native ROI images show abrupt brightening
and flatter fitted outer contours at source 160. This directs the next
experiment toward outer-shape/occlusion and scene assumptions, retaining both
the shared-target constraint and the native source chronology. No measured gaze
truth is available for these three reads.

The proposal experiment remains **offline and unpromoted**. Further work must
resolve the Student numerical admission discrepancy, evaluate useful partial
and true-inner support, and improve native geometry under illumination and
occlusion before integrating the mask likelihood into live gaze.


### Coherent spatial mask sensitivity: offline comparison

`outputs/mask-spatial-sensitivity-yccfttua/verdict-v1.json` and its matched and
independent-reference audits record this experiment. `--mask-spatial` adds four
outer-boundary displacement fields to the original three uniform threshold
states: positive/negative cosine and sine of twice the measured contour-normal
angle, bounded by the original minus/plus-one-logit displacement envelope.
Every fragment in an eye/boundary family shares its state. The uniform
seven-state prior is an engineering sensitivity assumption, not extra observed
samples or calibrated SAM probabilities. This finite basis is not a
rotation-invariant continuum. Pupil and true-inner evidence, point mass, noise,
source history and scene priors remain unchanged. The exact likelihood includes
all state combinations; the proposal initializer covers a bounded sixteen
assignments when more exist.

Both default/no-profile and uniform-mask controls match their frozen
predecessors on all 662 source events. Validation includes 176 ordinary offline
tests, six stereo viewer tests and a SAM-enabled production check. Exhaustive
synthetic likelihood checks reconstruct all 49 combinations for two spatial
families. The native audit checks 6,891 common factor masses/noise values and
1,836 eye rays against their one shared fixation.

| Subset | Final supported directions, uniform → spatial |
| --- | ---: |
| SAM calibration prefix | 58 → 48 |
| SAM reviewed contexts | 13 → 14 |
| SAM shadow recording | 106 → 108 |
| CPU Student shadow recording | 84 → 86 |

The total falls from 261 to 256, with 35 gains and 40 losses. Final publications
are counted once per provider/source, not as new physical recordings or held
frame votes. Median common source-to-source SN-FEIDA absolute log changes improve
from 0.01433 to 0.01069, 0.01242 to 0.01094 and 0.01775 to 0.01616 in the three
time series, but Student's 95th percentile worsens from 0.14886 to 0.15024. Of
eight fitted exact-RAW human-label matches, two improve, five worsen and one is
unchanged; two additional matches remain without a fit. No new scale
measurement or gaze truth was introduced. The source 159→160 ROI 1 jump grows
from 84.25 to 84.87 degrees.

The 252 synthetic results compare six support cases, seven boundary variants
and three seeds in each arm. A deliberately constructed spatial outer-boundary
distortion with strong complete-pupil support improves from 17.255 degrees to
zero in all three seeds. This is known fixture geometry inside the sensitivity
family, not native accuracy. Partial/complementary cases retain ambiguity and
some unbiased cases develop worse MAP fits. Complete true-inner support still
does not resolve its mirror pair.

Eight independent annealed references cover four native reads using two seeds,
4,096 paths and 128 steps. The second reference arm also changes the ordinary
sampler seed. All 808 source events preserve geometry, factors and source
identity; complete posterior parity is asserted only for the 404 events using
the original seed. Source 160 has 0.8973 and 0.9137 reference mass within fifteen
degrees of its selected ray. Both fail the existing mass-minus-two-standard-errors
admission margin despite admission by the primary sampler. Reviewed source 246
is withheld, with about 0.2–0.3% reference mass near its selected MAP.

Student source 4455 changes to a branch that agrees with adjacent fresh reads
and SAM on the same native recording. That is promising temporal/provider
agreement, not independent gaze truth. Its reference masses are about 0.929 and
0.932, but twice the numerical standard errors are about 0.074–0.081, so neither
reference establishes the admission margin. Student source 4415 retains weak
numerical support, including one insufficient-path reference.

This spatial candidate remains **offline and unpromoted**. Better area stability
does not offset localization regressions or optimistic admission. The next
comparison uses conditional mask fits only to initialize refinement of the
original full marginal joint objective, testing observed search gaps without
changing evidence, priors, association penalties or the shared fixation.


### Coherent mask-state initialization of the marginal fit

`outputs/mask-marginal-refinement-9l0ay_fq/verdict-v1.json` freezes this comparison.
`IntegrationConfig.mask_state_refinement` defaults to false, including in `live()`.
The diagnostic native replay enables it with `BUTTERCUP_MASK_REFINEMENT=1`.

The additional initialization improves the bounded model search, but this comparison does not establish better native gaze. The option remains offline and disabled by default.

Each coherent mask state supplies a temporary starting fit. That fit is then refined against the original complete marginal likelihood before it can compete with the original candidates. A single additional budget of at most 24 starts is shared across existing eye associations. Every omitted eye keeps its original full omission penalty; every result still has one shared fixation. Both conditional and full refinement work are included in diagnostics. No observations, scene priors, per-point noise or camera controls change.

All 177 ordinary solver tests pass. The SAM-enabled viewer build, six stereo UI tests, production check and tree audit pass. Native no-profile, uniform-control and spatial-control outputs each match the frozen predecessor on all 662 events. There are 1,324 matched native event pairs, 13,782 common factor mass/noise checks and 3,672 checks that each eye ray points to the single shared target. All 252 synthetic control results match their frozen predecessors within the 504-result comparison.

| Mask sensitivity | Final supported directions | Lower-cost final fits (> 1e-8) | Human visible-label RMS: improve / worsen / same |
| --- | ---: | ---: | ---: |
| Uniform | 261 → 257 | 208 of 325 | 2 / 3 / 3 |
| Spatial | 256 → 253 | 249 of 325 | 3 / 4 / 1 |

Many cost changes are numerically tiny. Ten labels match exact native RAW sources; eight have fitted results in both arms and two remain unavailable. These are Rob-only data and existing reviewed labels, including historical labeler provenance. They do not supply independently measured gaze. The subset repeats physical recordings across providers/cohorts, so publication counts are not new capture counts.

The source 159→160 ROI 1 jump grows from 84.25 to 84.69 degrees with uniform mask states, and from 84.87 to 85.94 degrees with spatial states. Uniform Student median/p95/max source-to-source SN-FEIDA changes improve, while several SAM metrics and the spatial Student median worsen. Original acquisition scale hints are used; there is no own-radius normalization or new metric scale evidence. The native replay preserves the source-history policy, but changed earlier joint fits become different subsequent initialization values. Small end-to-end cost increases therefore remain in the audit; the largest is 0.000002603 at a provisional source-160 publication. The unchanged fixed-request synthetic objective never increases.

Strong correctly modeled complete-pupil synthetic cases retain zero-error solutions. The uniform model still admits a spatially distorted complete-pupil example with about 16.26 degrees of error. Complete true-inner and complementary cases remain directionally ambiguous; some partial/complementary MAP errors worsen even at lower cost. Spatial partial-pupil admission remains one of three seeds. These fixtures do not establish the requested robust partial/cross-eye gaze behavior.

Fourteen independent annealed references cover all seven newly admitted source/kind combinations, with two seeds, 4,096 paths and 128 steps. Both seeds establish the existing numerical margin for spatial SAM sources 134/138 and uniform SAM source 4504. Uniform source 124 and spatial Student source 4463 are seed-dependent. Neither seed establishes the margin for uniform reviewed source 223 or spatial reviewed source 317. All 1,616 reference replay events preserve geometry, factors and source identity; the 808 events with the original ordinary-sampler seed also preserve the complete posterior. The reference shares the target and proposal family and cannot certify unvisited modes or calibrated accuracy.

The inspected source-317 figure is a concrete failure: retained SAM samples sit inside the human-marked iris boundary; visible-label RMS worsens from 29.01 to 29.17 pixels while the bounded sampler newly admits ROI 1. The independent references estimate only 0.714 and 0.758 mass near that ray, with substantial numerical error. The figure uses an independently SHA-checked native RAW10 buffer and the existing reviewed label file. Labels were read, not edited.

The next work is to address missed posterior mass on fixed native targets using bounded integration that is checked against these references, while retaining the positive sources 134/138/4504. Source 317 also remains a boundary-interpretation failure; more optimizer starts cannot add the missing anatomical evidence. Live mask attachment, reliable partial/true-inner/cross-eye behavior and the full stereo-gaze goal remain incomplete.

Artifacts: matched-audit-v1.json, reference-audit-v1.json, native-label-failure-v1.png, native-label-failure-v1.svg, and native-label-failure-provenance-v1.json. Builds, replays, references and audits are terminal. One initial synthetic launch used an unavailable CPU affinity and was corrected after that launch exited; audit assumptions about fitted publication targets and evolving warm starts were corrected before the final audit. No camera access, training, live viewer restart or deployment occurred.

### Independent annealed populations: confidence improvement with coverage and cost limits

The bounded SMC candidate lives in `joint/posterior/populations.rs` and remains
disabled in `IntegrationConfig::default()` and `live()`. Its full report and
audited results are in `outputs/posterior-populations-bu6ffb52/`. The native
source-223 and source-317 numerical false admissions improve, but broader
coverage, convergence and latency do not justify live promotion.

Both arms enable the existing experimental coherent mask-state proposals and
marginal MAP refinement, isolating numerical integration on that same target.
Live default mask attachment and refinement have not been promoted; this is
not evidence of a changed live-camera outcome.

The sampler follows the fixed-space SMC construction in
[Del Moral, Doucet and Jasra (2006), sections 3.2 and 4](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).
Each independent population starts from the complete frozen proposal mixture.
The bridge is `q^(1-beta) p^beta` with `beta=(stage/steps)^2`. Incremental
weights precede unbiased stratified resampling and invariant Metropolis moves.
Resampling occurs below half the initial particle-count ESS, except at the
final stage. Hard bounds reject proposals; invalid starts remain zero-mass
attempts and extinct populations are never retried. A zero-stage control is
ordinary stratified importance sampling. Temporary proposal fitting, contours,
the full marginal objective, priors and selected joint MAP are unchanged.

Population estimates are pooled by estimated normalizers, not by averaging
their posterior ratios. The existing between-replica delta-method calculation
now operates on independent populations. Duplicating descendants leaves its
error unchanged. `effective_samples` and `maximum_sample_mass` explicitly refer
to population normalizer shares in this mode; endpoint weight concentration
within a population is not independent-trial precision. The experimental
admission gate requires at least eight effective populations, no population
above 25% of normalizer mass, the existing angular-radius condition, and model
mass minus twice the between-population error of at least 90%. These are
engineering diagnostics, not calibrated probabilities or unvisited-mode bounds.

The smaller comparison uses 16 populations × at most 256 particles × 16
stages, at most 69,632 model evaluations after the unchanged pilot. The larger
selected-case comparison uses 32 × 256 × 64, at most 532,480. These exceed the
ordinary 8,192-draw cap and are not equivalent-work performance comparisons.
`BUTTERCUP_POPULATION_PARTICLES`, `BUTTERCUP_POPULATION_STEPS` and
`BUTTERCUP_POPULATION_COUNT` select the candidate in the ignored
`extraction_tests::recorded_mask_state_proposal_replay`; an explicit
`BUTTERCUP_MASK_REFERENCE_SOURCES` selection bounds native application to final
ROI2 publications. All other source events retain ordinary integration.

The seven existing numerical-reference cases receive both population budgets
and two seeds, giving 28 matched runs against the frozen independent AIS
references. The larger budget withholds both eyes at reviewed 223 and 317 in
both seeds. The smaller budget still admits source-317 ROI2 in one seed.
Sources 138/4504 retain support; 124 and Student 4463 also pass both population
budgets/seeds. Larger source 134 loses support in one seed because one
population carries 60.5% of normalizer mass (effective population count 2.69),
despite nearly all sampled directions agreeing. Source-317 mass estimates and
source-223 variance also remain sensitive to numerical work and seed.

The wider replay preserves all 662 native events per mask likelihood. The
candidate applies to 321 final ROI2 reads: 311 fitted and ten unavailable.
Twenty-one reviewed sources ending in ROI1 remain exact controls and are
excluded from candidate coverage. The eight matched cohort/likelihood replays
verify 1,324 geometry/factor/source event pairs and 1,206 explicit shared-target
ray identities. Admitted eye directions change as follows:

| Paired subset | Uniform control → population | Spatial control → population |
| --- | ---: | ---: |
| SAM middle | 56 → 54 | 50 → 48 |
| SAM reviewed | 10 → 12 | 12 → 9 |
| SAM shadow | 108 → 95 | 108 → 84 |
| CPU Student shadow | 82 → 83 | 82 → 60 |
| Total | 256 → 244 | 252 → 201 |

Uniform has 27 gains/39 losses; spatial has 13 gains/64 losses. All fitted
human-label localization, SN-FEIDA, timing alignment and gaze geometry are
unchanged. The approximately 85-degree source-159→160 jump still admits both
eyes. Median offline solve times rise from 79–264 ms to 446–1,214 ms across
cohorts, under low-priority shared CPU claims; this is not an isolated latency
benchmark. All data remain Rob-only, with no new independent gaze or scale truth.

All 181 ordinary solver tests and six stereo viewer tests pass, along with
SAM-enabled production checking and the tree audit. The 504-result synthetic
comparison preserves all 252 frozen controls and all candidate geometry/costs.
Complete-pupil cases keep 21 admissions per likelihood, including the existing
incorrect uniform shape-bias example. Four seed-dependent spatial partial-pupil
admissions disappear; true-inner, partial and complementary cases remain
withheld. The full gaze objective is therefore still incomplete. The next
work must address misplaced segmentation evidence and support ambiguities;
these population and AIS controls provide a stronger audit of numerical claims.

### Rejected outer masks: RAW alternatives and observation-family ablation

`outputs/raw-outer-candidates-iu_wmh21/` contains two **offline, default-off**
experiments. Neither is promoted. The comparison classifies this goal turn as
progress through implementation and matched native evaluation; the full gaze
objective remains incomplete.

The selected SAM mask can fail its existing RAW ring gate while its retained
outer samples still enter the solver with a wider five-pixel band. Detector
scores cannot be treated as a shared probability: cached SAM video scores can
exceed one, and Student scores describe mean foreground activation. A score
threshold would therefore introduce an unsupported probability interpretation.

The first experiment, `--raw-outer-candidates`, searches four neighboring
ellipse guides at axis offsets of −8%, 0%, +8%, and +16% of the shorter native
ROI dimension. It reuses the bounded native RAW peak extractor with coherent
edge runs. Guides locate searches; they are not emitted as complete boundary
observations. Measured runs share eight fixed image-angle groups, with at most
four alternatives per group, preventing repeated guides from creating extra
votes. Flat and saturated inputs emit no arcs. The rejected SAM samples and
their mask profiles are replaced, while the original pupil extraction and
scene initialization remain fixed. This does **not** establish anatomical
identity: a positive RAW edge can belong to a reflection or internal texture.

The second experiment, `--withhold-rejected-outer`, removes only the outer
observation family and its conic search hints after the existing gate rejects
it. Inner/pupil observations, uncertainty, source metadata and original scene
inputs survive; hint indices are remapped. Accepted masks are unchanged. This
is an ablation, not a claim that whole-mask rejection is the correct final
policy. Both experiments still use the baseline-derived scene center, which
is explicitly not independent anatomical truth. Projected outer ellipses after
ablation are model predictions rather than observed outer boundaries.

The ignored `extraction_tests::recorded_mask_state_proposal_replay` exposes
these through `BUTTERCUP_RAW_OUTER_CANDIDATES=1` and
`BUTTERCUP_WITHHOLD_REJECTED_OUTER=1`. They are mutually exclusive with each
other and the earlier partial-outline experiment. Native contour likelihood,
uniform mask-level likelihood, and spatial mask-level likelihood are each
evaluated with the same existing source-history policy. Source 160 passed the
existing RAW gate, so these experiments leave its large branch jump untouched.

The frozen SAM-middle, SAM-reviewed, SAM-shadow and CPU-Student-shadow caches
contain 662 source events, representing 342 final provider/source publications.
All 404 unique native byte ranges (57,554,880 bytes) match their recorded SHA256.
Each experiment reproduces 1,986 frozen control events across the three
likelihoods. Its audit counts final publications once per source, checks source
pairing and duplicate suppression, and verifies rays from the single shared
fixation rather than independent gaze averaging.

Only six ROI1 frames trigger either experiment: reviewed 317, 318, 10125 and
10237; SAM-shadow 4499; and Student-shadow 4465. RAW replacement emits 102 arcs
across these frames. Ablation removes 22 outer arcs and preserves 119 pupil
points. At reviewed 318, the new fit selects a different existing RAW pupil
peak in group 102; the observations themselves are unchanged. The audit compares
noise and support weight only when the same physical alternative is selected.

| Likelihood | Admitted eye directions: control | RAW alternatives | Outer withheld |
| --- | ---: | ---: | ---: |
| Native contours | 216 | 215 | 214 |
| Uniform mask levels | 257 | 258 | 256 |
| Spatial mask levels | 253 | 251 | 250 |

Available final fits remain 325/342 with RAW alternatives and fall to 323/342
with outer withholding for each likelihood. The additional unavailable sources
are reviewed 10237 and SAM-shadow 4499, where no usable boundary remains.
Reviewed 10125 retains an other-eye solution but no ROI1 boundary prediction.
These are coverage losses, not evidence of improved gaze accuracy.

Human-label localization is unchanged on the other matched reviewed frames.
The main affected labeled failure, source 317, has these visible outer-outline
RMS values in native ROI pixels:

| Likelihood | Control | RAW alternatives | Outer withheld |
| --- | ---: | ---: | ---: |
| Native contours | 29.06 | 31.68 | 16.43 |
| Uniform mask levels | 29.14 | 31.67 | 35.49 |
| Spatial mask levels | 29.17 | 31.68 | 16.43 |

The inspected `native-raw-failure-v1.png` shows why extra RAW peaks are not
sufficient: several measured alternatives lie inside the reviewed outer edge.
`native-outer-comparison-v2.png` also shows the ablation's uniform-likelihood
regression. Both ablation variants with mask profiles withhold source-317 gaze;
a better outer-outline RMS in the spatial case does not prove a correct gaze.
These are historical reviewed outer labels, not new blinded labels. Source 317
has no independent scale hint, and no new independently measured gaze or native
true-inner evidence is available. Common SN-FEIDA temporal steps with acquisition
scale support are unchanged; there is no demonstrated area-stability gain.

The final evaluator passes 186 ordinary tests, including the RAW-guide support
tests and the new native-pupil preservation and interleaved inner/pupil hint
tests. Six stereo UI tests, SAM-enabled viewer production checking, tree audit
and whitespace checks pass. The evaluator binaries are frozen under the run
directory (`build-checks-v1.json` and `build-checks-v2.json`); the final SHA256 is
`744acad9b1a84a31ef13a73ad6a70271b6570e71e82f5701631da1fb200ce2ce`.
No camera session, training, viewer restart, or deployment is part of this work.

The remaining task is to identify trustworthy anatomical support and retain
the alternatives that pupil and cross-eye evidence cannot resolve. These native
counterfactuals show that simply expanding RAW searches or deleting rejected
outer families does not yet deliver the requested robust gaze solver.

### Correlated arc-alternative likelihood: implemented, not promoted

`outputs/arc-mixture-joint-ppzw25ge/report-v1.md` records an exact finite-mixture
experiment. The opt-in `IntegrationConfig::marginalize_arc_alternatives` replaces
the cheapest alternative within each evidence group with
`-2 log(mean(exp(-C_j/2)))`. The existing robust component costs, support masses,
shorter-coverage penalties and caps remain intact. Identical observations count
as one state; distinct alternatives use uniform engineering prior weights.
Coherent mask-family states enclose these mixtures. Entropy residuals preserve
the exact objective and its EM tangent gradient. This is an uncalibrated
generalized likelihood, not a learned anatomical identity or confidence model.

The flag defaults **off**, including in `live()`. Disabled/singleton paths retain
the original streaming selection without mixture-component allocation. The
experiment neither adds gaze averaging nor changes source-keyed history.

Four frozen SAM/Student caches contain 662 events and 342 final provider/source
publications (325 available fits). Across three likelihoods, 1,986 matched event
pairs preserve the source data; 404 distinct RAW ranges totaling 57,554,880 bytes
were rehashed. These are Rob-only diagnostics. Native true-inner observations,
independently measured gaze truth and new metric scale are absent.

| Likelihood | Admitted eye directions: control | Arc mixture |
| --- | ---: | ---: |
| Native contours | 216 | 211 |
| Uniform mask levels | 257 | 249 |
| Spatial mask levels | 253 | 251 |

Available fits stay 325/342 in both arms for each likelihood. Human outer-outline
localization changes are negligible and mixed. Reviewed source 317 remains at
29.06/29.14/29.17px RMS. The source159→160 ROI1 jump changes from 85.46971° to
85.46787°, with source160 still admitted for both eyes. Area stability is also
mixed: Student common-step SN-FEIDA p95 absolute log change is
0.235616→0.235463 native, 0.141433→0.149073 uniform and 0.150856→0.135858 spatial.
These steps use unchanged acquisition scale hints, which may be held; no
candidate-radius normalization is used. The reviewed subset lacks eligible
scale-supported steps. Stability alone cannot establish gaze or anatomical
accuracy.

144 synthetic results cover complete pupil/outer, complete true-inner/outer,
partial pupil, partial true-inner, complementary eyes and outer-only support,
with clean/competing/duplicate alternatives and three seeds per arm. Complete
pupil support stays admitted at the synthetic truth. Exact duplicate parity
holds in 36 comparisons. Competing outer arcs move the partial-inner MAP by
33.42° and complementary MAPs by about 2.4°; both cases remain withheld. The
partial-pupil candidate gains one admitted seed (2/3→3/3) while increasing MAP
error to 0.96°. These outcomes do not establish the requested broad recovery.

Final evaluator SHA256 is
`b673e0d59c866fac628fdf188f171e62f86e2eb40e021b192a0282aa7eb4d63b`.
190 ordinary tests and six stereo UI tests pass. The allocation optimization
reproduces all 3,972 native rows in both frozen arms exactly except elapsed_ms,
and all 144 synthetic rows exactly. Both production checks, tree audit and
whitespace checks pass. The audit additionally verifies 5,508 shared-fixation
ray identities, 20,659 unchanged common-factor checks and 2,520 normalized
mixture/source entries. Live promotion remains unjustified.

A separate unresolved issue is association uncertainty: the posterior currently
conditions on the winning ROI association. The MAP omission lower bound can
prune a losing association, but does not prove negligible integrated probability.
Any association follow-up must normalize different nuisance models before making
probability claims and retain one fixation within each hypothesis.

### Native association-conditionals: source160 narrows only when coupled

The offline adapter now accepts `BUTTERCUP_ASSOCIATION_DIAGNOSTIC_SOURCES`, using
the existing explicit array of clock-lineage/timestamp selections. It attaches
`association_conditionals` only to paired available publications. Three separate
solves use the original source-bound evidence: the unchanged control, ROI1-only
observations, and ROI2-only observations. Both coarse scene priors and both
strictly past target starts remain fixed. No diagnostic output updates the live
tracker, its source history or presentation. Each hypothesis has one fixation.
The new helper and adapter code are test-only.

`outputs/association-conditional-6j0eubd6/report-v1.md` and `audit-v3.json` preserve
the results. All 662 replay event rows match the frozen native control exactly
after excluding diagnostic output and elapsed_ms. There are 310 paired available
publications and 1,204 verified conditional shared-fixation ray identities. Each
conditional receives its own bounded 16-start search and the current live
posterior configuration with one numerical seed. It is a diagnostic work budget,
not a live latency benchmark or subtraction from the original joint search.

| Subset | Original admitted directions | ROI1 alone | ROI2 alone |
| --- | ---: | ---: | ---: |
| SAM-middle, 35 paired publications | 48 | 10 | 7 |
| Reviewed, 27 paired publications | 12 | 0 | 1 |
| SAM-shadow, 124 paired publications | 80 | 2 | 4 |
| Student-shadow, 124 paired publications | 75 | 8 | 0 |

No separately admitted eye direction differs by over 15° from an admitted
original direction. No conditional MAP beats the original after paying the
unchanged omitted-ROI cost. Those penalties are engineering MAP costs, not
normalized association evidence; nuisance normalization is still needed before
mixing the models or treating cost gaps as probabilities.

At source160, joint conditional 90% angular radii are about 5.8° for both eyes,
while ROI1 alone has a 93.45° radius and ROI2 alone 103.30°. Both single-eye
results are withheld. Their penalized cost gaps are +105.37 and +95.42. The
known jump is not explained by a strongly supported opposing eye hidden by
association selection in this diagnostic. Joint constraints supply the narrow
conditional uncertainty. At source162, where pupil support is available in both
eyes, both single-eye results are admitted with radii about 8.1° and 8.7°.

Student sources 4456 and 4464 admit ROI1 alone while the original withholds it,
with direction differences 6.76° and 69.09°. Neither has independent gaze truth
here, so they are unresolved cases, not rescued gaze estimates.

Historical human-label comparisons also prevent treating single-eye fits as
automatic repairs: source 317 outer RMS worsens from 29.06px to 44.48px and
source 247 from 3.78px to 4.86px. Area stability is mixed. Each eye's SN-FEIDA
comparison uses identical source intervals and acquisition scale, breaking
adjacency at missing/unpaired final publications or gaps over 500ms. Student
ROI2 p95 absolute log step improves 0.36481→0.34362 while its maximum worsens
0.41653→0.61497. No candidate radius normalizes the scale. Reviewed sources
have no eligible scale-supported intervals; native gaze truth remains absent.

190 ordinary evaluator tests, production checking, source-tree and whitespace
checks pass. The frozen evaluator SHA256 is
`810a1ab06e08d3a17c9d6dab05a660a1853170f9cb06236dbe4fd1ff377c0912`.
This diagnostic changes the next investigation toward joint uncertainty and
source-timed anatomical support; it does not justify a live association change
or complete the requested robust gaze recovery.
