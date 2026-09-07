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

The full viewer suite currently has 967 passing tests, the same 41 failures as
the pre-change baseline, and 24 ignored tests. Live-adapter tests additionally
check source deduplication, different ROI sequences on one clock, crop transport,
missing-eye behavior, provider changes, radius units and shared gaze mapping.
No live user calibration or desktop-pointer trial has been performed for this
new path.
The standalone evaluator has 77 passing tests, including its source-replay,
shared live-tracker and partial-extraction tests. All 15 independent report tests pass and
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
report-stereo-conics.py candidate.jsonl comparison.json --baseline-evaluation baseline.jsonl --allow-extractor-changes
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test-stereo-conics-report.py
buttercup_raw10_preview --source-index frames.jsonl INDEX comparison.png output.jsonl
```

`--max-frames-per-cache N` permits bounded prefix comparisons of caches whose
writers have already completed that many records. It never counts the rest of
the corpus as evaluated. The coverage checker rejects missing, duplicate and
out-of-range source indices. RAW comparison panels are custom-decoded native
RAW / magenta SAM / green joint / yellow monocular; no ImageMagick is involved.
