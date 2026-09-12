# Source-aligned joint conic gaze solving

Development status (2026-09-08): implemented core and live adapter; the matched
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
