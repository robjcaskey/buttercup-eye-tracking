# Calibration from recorded stimulus

`--offline-calibration-playback SESSION CACHE BRIDGE NEW_REPORT` reconstructs
a calibration from the stationary targets actually submitted during a native
RAW recording. It uses fresh worker evidence and the native stereo bridge;
recorded predictions and target coordinates never initialize the geometry.

The command checks native source identities, clock lineage, paired completion,
direction support, and cache/bridge membership. The source/ROI safety checks
come from the native bridge. Preview exclusion and sensor settling use the
same helper as live calibration. Each physical exposure contributes at most
once. Moving recovery targets, unseen targets and preview frames cannot train
the mapping. The first qualified paired result owns the sample.

Target windows come from `metadata.oim1` presentation/configuration records,
using monotonic host clocks. The hold can complete on a UI tick using samples
already collected; it does not require a new RAW frame after the timer. A
contiguous recorded target advance attests that hold under the checked recipe.
An interrupted recovery clears that target, retaining completed targets within
the same authority. Missing visits remain missing. The current native target
recipe and hold must match the recording.

The report applies the existing dominant angular cluster, target coverage,
display-plane, affine and shared 2D/3D support rules. It also refits with each
target excluded. A failed training fold or absent held estimate remains explicit.
In particular, removing the center can violate native coverage, and removing
one of only seven available targets leaves insufficient training coverage.

This is fixed-stimulus offline reconstruction. RAW host arrival and successful
buffer submission are not measured exposure/scanout times. Completion-paced
inference does not prove live deadlines or a changed closed-loop presentation
schedule. Acceptance on calibration cues does not establish independent gaze
accuracy or finish an interrupted live sequence.

## Reproduction

Use the checkout's normal LibTorch/runtime environment. Prepare an immutable
index with `--offline-calibration-prepare SESSION NEW_DIRECTORY`, then export
fresh paired evidence with `BUTTERCUP_OBELISK_DEVICE=cpu`,
`BUTTERCUP_STEREO_LIVE_REPLAY=combined` and
`--offline-stereo-sam-export SOURCES NEW_CACHE 0 FRAME_COUNT student`.
`student` is the internal provider name for Butter Obelisk. Record the model,
refiner setting and runtime device; the comparison below used CPU and refiner off.

Export source-qualified stereo bridge rows through the ignored native test
`joint_gaze_live::tests::recorded_calibration_acquires_from_source_timed_live_worker_evidence`.
Set `BUTTERCUP_JOINT_CALIBRATION_CACHE`, `BUTTERCUP_JOINT_CALIBRATION_REPORT`,
`BUTTERCUP_JOINT_CALIBRATION_INTEGRATION=live` and the zero-based
`BUTTERCUP_JOINT_CALIBRATION_EYE`. Specify the actual mounting policy with
`BUTTERCUP_CAMERA_MOUNT_TRIAL`; an alternate policy is a labeled experiment.
This test's acquisition result alone does not establish calibration acceptance.
Pass its report and matching cache to the playback command for target fitting.

For recordings with explicit orientation-plus visits, the existing
`BUTTERCUP_SCREEN_REFERENCE_TRIAL` harness setting also accepts `recorded`.
Each cache row must then supply a `recorded_orientation` receipt with exact
`source_ns`, `clock_lineage`, boolean `active`, and
`timing_basis=paired-raw-host-arrival-vs-recorded-submit`. Visibility controls
orientation even if a sign was acquired earlier; a stationary corner cannot
silently inherit an unpresented top-plus constraint. The offline preparation
recipe for the September 16 comparison is under
`outputs/calibration-sam-comparison-20260916/compare.rs`. This is a test-harness
mode, not a new live environment control or a measured scanout clock.

That comparison replays all 223 paired exposures from session
`1789468040-744875454` through CPU Obelisk and SAM3.1, with identical native
geometry gates and the recorded BelowEyes orientation schedule. SAM passes the
recorded upper-right visit in the native collector; Obelisk does not. Only two
targets exist, so neither can establish a full nine-target calibration. The
comparison's RAW sheets and area report also reveal substantial SAM fit
outliers. Better mask coverage is not permission to treat all teacher geometry
as truth or to skip bootstrap provenance and human localization validation.

September 14 results are under `outputs/calibration-stimulus-20260914-2050`.
The latest session `1789432526-545550713`, recorded above-eyes, supplies seven
stable targets and an accepted offline fit. The earlier nine-target session
`1789347789-185587461`, originally rejected with monocular features, supplies
nine fresh stereo estimates and an accepted fit with eight shared inliers.
Independent presentation lookup verified all 129 and 173 admitted left-eye
sources, respectively. No live setting or geometry confidence threshold changed.

The acceptance result is not the end of validation: the earlier run's held-out
lower-right target has 0.244 screen-fraction error. The latest seven-target run
cannot support a native leave-one-target-out fit with its available coverage.
The reports preserve these limitations, source-matched RAW/3D inspection sheets,
and an explicitly below-eyes comparison of the latest recording. Further work
must address the remaining mapping/geometry error rather than hide failed folds.

## Input-coordinate comparison

`camera_tangent_diagnostic` evaluates an explicit offline alternative: map
`gaze.x / gaze.z, gaze.y / gaze.z` through the native robust affine. It uses the
same source-qualified samples, display-plane fit and shared 2D/3D support policy
as the baseline. It does not select a live mapping or change direction confidence.
The shared prediction-support helper keeps the target coverage and disagreement
rules common to both evaluations.

The matched reports are in `outputs/calibration-mapping-20260914`. The tangent
candidate rejects the latest above-eyes run, whose baseline fit is accepted.
On the older nine-target run it reduces held-out lower-right error from 0.2442
to 0.1866 screen fraction, but reduces the number of accepted, evaluable held-out
folds from seven to five. On the explicitly alternate below-eyes version of the
latest run, it worsens lower-right error from 0.1498 to 0.1687 and reduces those
folds from seven to four. Missing estimates and failed folds are not counted as
zero error. This is not a consistent improvement and has not been promoted.
All three baseline fit and leave-one-target-out results remain exactly unchanged;
71 calibration tests pass, with five optional corpus tests ignored.

## What motion can carry into later frames

The current joint tracker retains source-keyed target solutions for up to 500 ms
as optimization starts. Those solutions are not residuals or independent evidence
in the next exposure's posterior. The solver does not currently learn a persistent
subject-specific eye model from a parallax sweep. The saved display calibration
maps a supported gaze to the screen; it does not make a subsequent ambiguous
surface direction supported.

Varying eye directions can provide geometric diversity for a future temporal eye
model. For example, [pye3d](https://docs.pupil-labs.com/core/developer/pye3d/)
estimates eye position from retained pupil observations at several time scales.
That implementation is for a different camera arrangement and is not evidence
that additional head movement will fix this remote stereo recording. Head motion
also changes eye position relative to our camera and would need to be modeled.

The existing stimulus already provides several eye directions and supports the
accepted offline fits above. Missing parallax has not been established as the
cause of the remaining failures. A temporal experiment should learn persistent
geometry while allowing current gaze, pupil size and head pose to change; keep
future frames and held targets out of the learned support, propagate uncertainty,
and compare source-matched RAW localization, coverage and gaze error. Carrying a
previous gaze or confidence forward without new support would not validate it.

## Camera hypotheses tested on the existing RAW

`outputs/calibration-intrinsics-20260914` contains 47 full native geometry
replays: seventeen fixed-camera trials on each recording, five explicitly
below-eyes trials on the latest recording, and four radial-lens approximations
on each recording. The detector caches are the same CPU Butter Obelisk,
refiner-off outputs described above; the native bridge recomputes joint RAW
evidence and geometry. These are not 47 independent recordings or detector runs.

The fixed grid includes focal lengths 2000, 2800, 4000, 5600 and 8000 pixels;
opposing focal-axis changes of 2% and 10% around 4000 and 5600; and principal-point
offsets of 500 pixels in X or Y around [4000,3000]. More supported directions did
not necessarily produce a usable mapping: for example, focal 2000 increases
the latest recording's current ready sources from 192 to 297, but supplies only
six stable targets and fails native calibration coverage.

Useful fixed-camera candidates, all with principal point [4000,3000]:

| Recording / assumption | Focal X / Y, px | Stable targets / fit | Evaluable held-target folds | Mean held error |
| --- | --- | --- | --- | --- |
| Earlier / recorded below | 4000 / 4000 | 9 / accepted | 7 | 0.1040 |
| Earlier / recorded below | 5712 / 5488 | 9 / accepted | 8 | 0.0685 |
| Latest / recorded above | 4000 / 4000 | 7 / accepted | 0 | unavailable |
| Latest / recorded above | 5488 / 5712 | 8 / accepted | 6 | 0.0762 |
| Latest / alternate below | 4000 / 4000 | 8 / accepted | 7 | 0.0857 |
| Latest / alternate below | 5600 / 5600 | 8 / accepted | 7 | 0.0634 |

Errors are Euclidean distances in normalized screen coordinates, conditional
on fixation at the recorded cue. Different fold counts are explicit and must
not be mistaken for matched populations. The earlier fixed candidate reduces
the lower-right held error from 0.2442 to 0.1079. Choosing camera parameters by
the remaining training targets, rather than that held target's error, selects
[4080,3920,4000,3000] for that fold and yields 0.1323. Across eight evaluable
earlier folds, camera selection gives mean 0.0666; the other fold lacks the
required center in its training set. The grid was explored using these same
recordings, so this is exploratory validation, not a preregistered independent
test set or a measurement of true camera intrinsics.

The latest above-eyes candidate recovers a stable estimate at target eight and
an accepted eight-target fit. That target participates in this full fit. When
target eight is excluded from camera selection, the selector prefers isotropic
5600 and has no supported held estimate there. Target nine was never presented.
Thus the recovered eight-target fit does not establish independent accuracy at
the previously failing target or complete the original interrupted UI sequence.

### RAW checks and rejected transfer

The native `--offline-calibration-geometry-compare` command checks exact source,
ROI and clock identity, scores both projections on the same baseline RAW arc
supports, and compares adjacent SN-FEIDA steps using the same RAW similarity
determinant in both arms. Motion uses a fixed exclusion of the baseline iris
and its margin. There are 771/852 reliable ROI motion records in the latest
clip and 422/460 in the earlier clip; each geometry comparison further reports
its matched available pairs. No human localization labels exist in this check.
The pixel residual is a shared radial approximation, not labeled boundary truth.

For the earlier [5712,5488] and latest-above [5488,5712] candidates, native
projection coverage is unchanged. Mean outer point residual rises by about
0.034 px per eye in the earlier clip and 0.007/0.018 px in the latest clip.
Independently normalized area-step changes are mixed and small; the detailed
per-eye values and worst source regressions remain in the reports. This is a
tradeoff, not evidence that every geometry diagnostic improves.

Transferring [5712,5488] to the latest clip under below-eyes is **rejected** despite
lower held-target error: it reduces right/left projection counts from 360/414
to 288/350, with an 18 px outer-support miss at left source 1004. Its source-matched
RAW sheet shows the right projection disappearing and the left fitted ellipse
collapsing onto part of the iris. Isotropic 5600 under that same assumption
retains 394/415 projections and avoids this particular failure. The actual live
above-eyes selection has not been changed.

### Shared radial approximation

The radial trial uses one base camera [5712,5488,4000,3000] and coefficients
0.1, 0.3, 0.6 or 1.0. At the measured ROI-center reference it inverts
`r_distorted = r * (1 + k1*r*r)`, matches the reference ray and diagonal local
projection derivatives, and supplies the resulting effective pinhole camera
to the shared solver. It omits off-diagonal shear and curvature across the ROI;
it is a bounded diagnostic, not full lens undistortion. Camera reference and
effective intrinsics can vary with source position while the base parameters
and coefficient remain fixed.

Coefficient 1.0 preserves projection coverage and accepted fits in both actual
recording configurations. The earlier held mean is 0.0697 over eight folds;
the latest supplies seven stable targets and still does not recover target
eight. Smaller coefficients fail the latest native mapping, and two also
trigger repeated acquisition in the enclosing stationary-target phase probe. The full reports
and failures are retained. Camera selection over the expanded grid still picks
the fixed-camera alternatives above. This supports material camera-model
sensitivity, but does not establish a unique physical lens calibration or show
that camera intrinsics alone explain the remaining failures.

### Reproduce

The camera overrides exist only in the native test harness; live camera defaults,
mounting selection and confidence thresholds are unchanged. Set the normal
recorded-calibration cache/report/eye/mount variables from the earlier recipe,
then add `BUTTERCUP_JOINT_CALIBRATION_INTRINSICS='[fx,fy,cx,cy]'` when running
`joint_gaze_live::tests::recorded_calibration_acquires_from_source_timed_live_worker_evidence`.
For the radial approximation also set `BUTTERCUP_JOINT_CALIBRATION_RADIAL_K1`.
Nonzero radial trials are explicitly labeled in their receipts. An unsuccessful
phase probe remains a failed test even if its complete evidence ledger can still
be scored against the fixed recorded stimulus. This probe never submits hidden
thumbnails or replays the recorded target transitions: it stays on an unpresented
first target. Its six failures across the camera grid do not establish that the
corresponding recorded calibration sequences fail. Conversely, an accepted
offline fit does not prove live nine-target progression and persistence succeed.

Run each bridge through the playback command, then compare a fixed list with
`--offline-calibration-camera-compare NEW_REPORT PLAYBACK_REPORT...`. The selector
maximizes available training targets and then minimizes native training residual;
held-target values cannot select the camera. Missing held estimates remain missing.
For RAW validation use `--offline-stereo-motion-export SOURCES NEW_MOTION CACHE`
and `--offline-calibration-geometry-compare BASE_BRIDGE CANDIDATE_BRIDGE MOTION NEW_REPORT`.
All preparation and evaluation commands use the native Rust implementations.

The final build passes 72 calibration, 17 joint source-tracking and 19 camera
cooperation tests (six optional corpus tests are ignored across the first two
groups). The nominal 4000-pixel trials reproduce the preceding baseline fits
exactly. Frozen binaries, source/model hashes, test logs, accepted/rejected
camera reports, RAW contact sheets and native 3D rays remain under the output
directory. No camera candidate has been promoted to the live viewer.

### Numerical integration budget control

Six additional native replays under `outputs/calibration-precision-20260914`
change only the maximum posterior integration budget. The native
`--offline-calibration-precision-compare BASE_PLAYBACK CANDIDATE_PLAYBACK NEW_REPORT`
command verifies identical source identities, measured supports, MAP eye geometry,
projected conics and optimizer cost on every row before comparing admission.
All six comparisons pass; the repeated 8192-budget nominal control is identical.

| Recording / camera | Maximum draws | Fresh qualified sources | Stable targets / fit | Mean actual draws per pair |
| --- | ---: | ---: | --- | ---: |
| Latest / nominal | 8192 | 188 | 7 / accepted | 6134 |
| Latest / nominal | 32768 | 204 | 7 / accepted | 14752 |
| Latest / nominal | 131072 | 215 | 7 / accepted | 39561 |
| Latest / 5488,5712 | 8192 | 293 | 8 / accepted | 5530 |
| Latest / 5488,5712 | 32768 | 323 | 8 / accepted | 10764 |
| Latest / 5488,5712 | 131072 | 334 | 8 / accepted | 20909 |
| Earlier / 5712,5488 | 8192 | 212 | 9 / accepted | 3851 |
| Earlier / 5712,5488 | 32768 | 216 | 9 / accepted | 4291 |

These counts apply the shared source qualification checks, beyond the raw Ready
status. The latest nominal 32768 run gains 26 qualified sources and loses 10;
131072 gains 35 and loses eight. Increased computation can remove apparent
support as well as recover it. Neither recovers another nominal-camera target.
The changed-camera latest held-target mean changes from 0.0762 (six evaluable
folds) to 0.0814 (seven) and 0.0832 (six); changing fold coverage prevents a clean
accuracy comparison. The earlier changed-camera mean changes from 0.0685 to
0.0682 over the same eight folds.

One latest source, 1151, illustrates numerical admission: its selected-eye
estimated mass inside 15 degrees is 0.9314, but the between-batch error lowers
the conservative bound to 0.8928, below the 0.9 requirement. Other rejected
sources have genuinely broad conditional distributions. More integration
cannot resolve an underlying ambiguity by itself.

Set `BUTTERCUP_JOINT_CALIBRATION_QUADRATURE_BUDGET` in the native test harness
to reproduce these trials. Early stopping remains enabled, so these are maximum
budgets, not independent fixed-budget reference integrations or convergence
proofs. Estimates remain conditional on the anatomy/camera model. Whole replay
wall times under concurrent bounded jobs do not establish live frame latency.
No integration default or live setting is changed. Target nine in the latest
recording was never presented; no experiment creates evidence for that target.

### Recorded-stimulus production collector replay

`outputs/calibration-collector-20260914` contains four replays which additionally
exercise `VirtualMouseMode::calibration_presented` and `observe_at_frame`, including
the production finalizer. The playback report's `native_collector` section is
separate from the earlier feature-summary fit. The recorded hidden-thumbnail
submission and its elapsed-time receipt reconstruct the target timer; no short
visit is extended to make the hold pass. The first complete fresh paired solve
owns one vote. Once the candidate advances, remaining sources in that recorded
visit are not fed to the next target.

| Recording / camera | Production collector completed targets | Production finalizer |
| --- | --- | --- |
| Earlier / 4000,4000 | All nine | Accepted, display-intersection mapping |
| Earlier / 5712,5488 | All nine | Accepted, display-intersection mapping |
| Latest / 4000,4000 | First seven | Not reached; target eight requests recovery |
| Latest / 5488,5712 | First eight, including both recorded eighth-target visits | Not reached; target nine absent |

This verifies more of the actual calibration path than computing an offline fit
alone, but the stimulus remains externally imposed. Each recorded stationary
visit initializes an already-acquired sign phase, preserves previous target
samples within its recorded authority, and resets the interrupted target on a
recorded recovery. It does not replay the moving recovery stimulus, inference
latency, or a counterfactual closed-loop schedule. The hidden-submit trace stamp
also follows the UI callback rather than measuring scanout. No calibration is
installed in live settings and save/reload is not exercised by this diagnostic.

The source/timer regression suite now passes 74 calibration tests, with five
optional corpus tests ignored. New tests reject changed timer receipts and
verify that duplicate exposures cannot add votes and a short visit cannot be
extended into completion. Both the nominal and camera-candidate corpus reports
remain available, including the incomplete latest sequence.

### Native persistence and cursor stream

The playback command now saves and reloads a calibration only when its production
collector completes every recorded target and the native finalizer accepts it.
It creates a new sibling `*.playback.reload-check` directory with the native
`gaze-calibration.json`, `mapped-sources.jsonl`, and a provenance receipt. This is
an isolated replay artifact; it never writes `outputs/settings/gaze-calibration.json`.
The receipt preserves the source session, cache, bridge, detector/model path,
mounting policy, camera diagnostic parameters and integration recipe. The native
calibration format itself does not encode intrinsics, so a camera-trial artifact
cannot be treated as a fit for arbitrary live camera assumptions.

Under `outputs/calibration-reload-20260914`, both earlier nine-target runs save
and reload successfully. Each maps 212 qualified native sources with no lost
cursor estimates. Comparing the reloaded output against the production
finalizer's predictions before serialization gives maximum normalized-screen
differences of 1.11e-16 for the original camera and 6.72e-16 for [5712,5488]. Both
latest-recording runs correctly skip persistence because their ninth target is
absent. The finalizer's source predictions and reloaded source stream are retained
for inspection; round-trip consistency is not independent accuracy validation.

`reloaded-cursor-paths.png` compares the source-matched cursor clouds with the
recorded cues. The candidate brings center and bottom-center closer to their
cues, but top-center overshoots and isolated outliers remain. These are training
cues, not an independent test set; the earlier held-target reports remain the
separate generalization diagnostic. The source-tree audit and 75 calibration
tests pass (five optional corpus tests ignored), including rejection of an
incomplete or evidence-free result for persistence.

### CPU replay at recorded cadence

`outputs/calibration-offered-20260914` records the latest 852-ROI clip offered
at its native sensor cadence through the production two-lane worker ingress.
Both lanes explicitly log `requested=cpu selected=Cpu`, using four CPU cores
(4-7), two OpenMP/MKL threads and the same pinned Butter Obelisk model. All 852
results complete, with zero dropped or replaced jobs. Worker result availability
lags scheduled source time by 70.19 ms median, 76.41 ms p95 and 83.96 ms maximum.
These timings include the worker pipeline and scheduling, not the subsequent
native joint solver or UI presentation.

Replaying those actual delayed completions through the native bridge preserves
exactly the completion-paced qualified source sets: 188 for [4000,4000] and
293 for [5488,5712]. Every common qualified gaze vector is also bit-identical.
Intermediate waiting/uncertain publications change, but no final qualified
source is gained or lost in either matched camera arm. The two native bridge
tests pass. This reduces the evidence for detector backlog as the cause of this
clip's failure; it does not establish end-to-end timing or target-window admission.

Reproduce the worker run with `BUTTERCUP_OBELISK_DEVICE=cpu` and
`BUTTERCUP_STEREO_LIVE_REPLAY=offered-combined`, then use the existing
`--offline-stereo-sam-export SOURCES NEW_CACHE 0 852 student` command. Student
corpus exports now reject auto/GPU selection before starting inference. The
initial accidentally auto-selected CUDA run is retained with `not-cpu` in its
filename and excluded from these CPU results.

The recorded-stimulus fit command currently rejects offered-load caches: its
collector must not mistake result completion for RAW arrival. Extending that
collector to account for actual availability and joint-solver service time is
still required before claiming a timing-valid accepted calibration. The source
receipts and both bridge ledgers are retained for that next check.

### Native service cost and explicit-pair deferral experiment

`outputs/calibration-native-service-20260914` instruments native proposal
observation, publication and admission on the same offered CPU cache. RAW decode,
JSON I/O, model inference and rendering are excluded from the measured service.
With [5488,5712], native work totals 16.410 seconds: proposal observation accounts
for 16.398 seconds and publication/admission for 0.012 seconds. The per-completion
total median is 17.35 ms, p95 35.63 ms and maximum 45.67 ms. This cost is material
on top of worker latency; it was measured separately from the worker run.

The prototype `BUTTERCUP_JOINT_CALIBRATION_DEFER_EXPLICIT_PAIRS=1` stores the first
ROI evidence of an explicitly paired request without performing a provisional
one-eye solve. Once the matching source arrives, the same native solver runs.
The normal independent-ROI path is preserved. The source floor advances while
waiting, so an older publication cannot become a fresh observation. No same-time
provisional result is allowed to initialize the paired solve.

| Camera | Baseline native work | Deferred native work | Qualified sources, both arms |
| --- | ---: | ---: | ---: |
| 4000,4000 | 16.581 s | 11.890 s | 188 |
| 5488,5712 | 16.410 s | 11.746 s | 293 |

Both comparisons preserve exactly all 416 complete paired publications: target,
eye centers, surface normals, gaze directions, projected conics, measured arcs,
cost and posterior are bit-identical. Native geometry comparisons also preserve
RAW support residuals, projection coverage and independently normalized area
steps. The retained source-matched RAW overlays therefore still describe these
paired solves; their reflection/pupil limitations remain unchanged.

The 18 source-history tests pass, including a new full-solution/posterior parity
test for both arrival orders, rejection of held-source publication while waiting,
and preservation of independent single-ROI operation. The optimization is still
test-only: missing/invalid paired-result presentation and fallback behavior need
validation before enabling it in the live bridge. The approximately 28% native
work reduction is not a measured end-to-end frame-latency improvement.

### Production optimization for already-buffered pairs

The viewer now passes its admitted proposal batch to
`Bridge::observe_admitted_batch`. It elides a provisional solve only for adjacent
opposite-eye results which both advertise a paired request, have the same source
timestamp and prompt generation, and independently pass exact RAW-journal identity,
buffer-size and current-source checks. Both results are already present; the
bridge never waits for an assumed future partner. It keeps the existing admission
order and does not combine interleaved exposures. Stereo-disabled operation returns
before these pairing checks.

This uses the same pair-wait solver primitive tested above. The broader
request-based deferral remains an experimental test option. Missing, malformed,
unverified, duplicate, legacy single-ROI and differently prompted results follow
the original path. Empty segmentation evidence still reaches the native solver
as an actual completed observation; it is not invented as a successful detection.

The buffered-pair regression compares full native publications against sequential
processing for ten cases, including both arrival orders, an empty partner, absent
and invalid partners, interleaved exposures, duplicate delivery and changed prompt.
The current build and receipts are under `outputs/calibration-buffered-pairs-20260914`.
Actual live savings depend on how often both results occur in one admitted batch;
the earlier 28% measurement applies to the broader replay experiment. The running
viewer has not been replaced by this build, and this change does not establish
end-to-end calibration timing or recover an unrecorded target.

### Coupled CPU processing and measured target-window admission

`outputs/calibration-coupled-20260914` runs the real CPU Obelisk workers while
the native bridge consumes their actual typed proposals. Bridge work blocks
subsequent pump dispatch/drain, and both worker lanes remain active during that
work. Sources are offered when their recorded host RAW arrivals become due; an
atomic pair waits for both recorded arrivals. The replay records native-result
availability after bridge processing on that same clock. This is materially
stronger evidence than adding timings from separate runs.

This scalar completion hook does not apply the new buffered-pair optimization.
Synchronous diagnostic export also adds load to the pump. Rendering, display
scanout and a counterfactual closed-loop target schedule are not simulated.
The enclosing stationary-target acquisition probe remains separate from the
recorded-stimulus fit; its success is not a nine-target UI completion claim.

| Recording / camera | Native proposal records | Result delay median / p95 / max | Late qualified results excluded | Native fit |
| --- | ---: | --- | ---: | --- |
| Latest / 5488,5712 above | 852 / 852 RAW ROIs | 129.09 / 179.01 / 250.09 ms | 12 | Eight stable targets, accepted |
| Earlier / 4000,4000 below | 459 / 460 RAW ROIs | 126.30 / 171.28 / 196.43 ms | 14 | Nine stable targets, accepted |

Both runs log explicit CPU selection and zero dropped/replaced worker jobs.
The earlier worker reports 460 completed jobs but emits 459 proposal records;
right-eye source sequence 223 (input 368) has no proposal record. That missing
result remains absent, never a fresh observation or a fabricated negative mask.
The geometry comparison retains 416/227 complete paired publications respectively.
Direct typed proposals produce small numerical differences from reconstructed
JSON-cache proposals: mean gaze-vector differences are about 2.2e-8 and 4.0e-9,
not bit identity. Native RAW residual, coverage and independent-area comparisons
are retained and show only subpixel numerical changes in these matched solves.

Use `BUTTERCUP_JOINT_CALIBRATION_NATIVE_INDEX=SOURCES` instead of a cache when
running the native bridge corpus test; set
`BUTTERCUP_JOINT_CALIBRATION_NATIVE_CACHE_REPORT=NEW_CACHE`,
`BUTTERCUP_OBELISK_DEVICE=cpu` and
`BUTTERCUP_STEREO_OFFERED_CLOCK=recorded-host-arrival`. The usual eye, mount,
intrinsics and report settings still apply. Then run
`--offline-calibration-availability SESSION CACHE BRIDGE NEW_REPORT`.
This command verifies the clock anchor against native RAW, matches each worker
receipt to its native processing receipt, and requires availability after RAW
arrival and before the recorded target exit. It retains the original hidden-
thumbnail and source-settle gates. A result for an earlier target cannot migrate
into the next target. Missing proposal records are reported as unavailable.

The availability-aware scorer now also drives the production `VirtualMouseMode`
collector with measured native-result availability. RAW arrival remains the
source-admission clock, so delayed preview frames cannot become eligible merely
by finishing after thumbnails disappear. Results arriving at or after target exit
are excluded. The recorded source frontier supplies the current-frame age check.
Recorded stationary visits are imposed externally; moving acquisition, rendering,
scanout and a counterfactual closed-loop target schedule remain untested.

`outputs/calibration-timed-collector-20260914` contains the completed collector
checks. The earlier nominal-camera recording completes all nine targets, passes
the native finalizer, and saves/reloads its display-intersection mapping. All 212
qualified source mappings survive reload with zero numerical difference. The
latest candidate-camera recording completes the eight presented targets but has
no ninth visit, so it does not finalize or save a completed calibration. No live
camera defaults or personal calibration files were changed.

The unchanged completion-paced control reproduces its features, fit and
held-target results exactly. Seventy-eight calibration tests pass (five optional
tests ignored), including late-result exclusion and delayed-preview rejection.
The source-tree audit also passes.

`native-result-availability.png` plots actual result times, target windows and
excluded late samples. The latest recording still contains no ninth target;
the accepted eight-target fit does not invent that missing visit.

### Paired camera trials with measured CPU availability

`outputs/calibration-timed-camera-pairs-20260914` fills the missing timed arms:
latest nominal and earlier candidate. Both use CPU 4–7, two Obelisk workers,
recorded host arrivals and synchronous native bridge processing. These are paired
recordings, not identical scheduling traces or independently unseen sessions.
The comparison retains actual completion times and missing proposals.

| Recording | Camera fx/fy | Native collector targets | Available accepted held-target predictions | Mean held error |
| --- | --- | ---: | ---: | ---: |
| Latest, above | 4000/4000 | 7 of 8 presented | 0 | unavailable |
| Latest, above | 5488/5712 | 8 of 8 presented | 2 | 0.1722 |
| Earlier, below | 4000/4000 | 9 of 9 | 5 | 0.1283 |
| Earlier, below | 5712/5488 | 9 of 9 | 8 | 0.0674 |

Errors are Euclidean distances in normalized screen coordinates, not degrees or
measured fixation error. The center holdout necessarily fails mandatory center
coverage; it is not a geometry regression. Missing/failed other folds remain
unavailable. On the five earlier targets shared by both settings, mean error
falls from 0.1283 to 0.0633. The candidate also saves/reloads all 212 qualified
source mappings (maximum numerical difference 5.7e-16).

The latest candidate recovers collection at target 8, but the held-target screen
fit for that target remains rejected. Its two available held predictions have
errors 0.2528 and 0.0916. This is evidence for improved collectability, not proof
of reliable edge accuracy. The ninth target was never presented, and there is
still no complete latest-session calibration to install.

The latest nominal export finishes 852 worker jobs with 851 proposal records and
zero dropped/replaced jobs. Its separate unpresented stationary-target probe
fails the repeated-acquisition assertion after export; that failure is retained
in `latest-nominal.log`. The actual recorded-target collector is evaluated
separately from those exported results, with no invented missing observation.

Native source/ROI/clock-matched geometry comparisons retain outer projection
coverage (latest 394/416 per eye, earlier 227/227). Mean outer residual to baseline
RAW supports increases slightly: latest 0.754/0.723 to 0.761/0.741 px; earlier
0.797/0.756 to 0.830/0.790 px. Independently scaled SN-FEIDA steps are mixed, not a
uniform improvement. There are no human contour labels for these comparisons.
The source-matched RAW contact sheet and the new `held-targets.png` were visually
inspected; the latter makes the remaining edge errors and unavailable folds
explicit. No intrinsics, mounting settings, live calibration or running viewer
were changed. The isolated saved mapping requires its companion intrinsics
provenance: the native personal calibration JSON alone does not restore the
experimental gaze basis.

### Localizing the remaining upper-right discrepancy

`outputs/calibration-plane-diagnostics-20260914` adds observational diagnostics
to the shared native display-plane fitter. The optional diagnostic records the
number of refined candidates and the strongest candidate before coverage
rejection, ranked by inlier count then robust cost, with every target residual.
It never supplies a rejected plane as a calibration and does not alter fitting,
coverage, the 0.10 inlier threshold, or final consensus polishing. Ordinary live
calls do not request or compute these extra diagnostics.

All four timed baseline/candidate reports reproduce their previous selected
features, fit outcomes, held-target outcomes and native collector results exactly
when the new diagnostic field is excluded. Seventy-nine calibration tests pass,
five optional tests remain ignored. A recorded-feature regression verifies that
the diagnostic candidate cannot promote a rejected fit.

For the latest candidate, every non-center failed holdout has six inliers and
one outlier: upper-right (target 2). Its residual ranges from 0.106 to 0.143 in
normalized screen distance. Holding out bottom-center (target 8) leaves target 2
at 0.132, so the failure is not absence of the bottom target's gaze estimate.
Holding out target 2 itself allows seven supported training targets to fit.

Source inspection further localizes the problem: all 16 admitted target-2
sources have contributing eyes `[false,true]`, despite complete two-ROI camera
source groups. Source completeness is not binocular geometric support.
`upper-right-raw-neighbors.png` shows native exposures 869–871 for both eyes;
`upper-right-native-rays.png` shows only the actually available left-eye rays.
The right iris remains visible in RAW, but its Obelisk cache at source 870 has
an empty candidate list. The left-eye pupil supports appear influenced by the
screen reflection; that visual assessment is not human contour ground truth.
The next investigation is upstream candidate extraction/rejection for the right
eye. No held eye, synthetic right-eye contour or missing ninth target was added.

### Recovering the missing right eye at upper-right

`outputs/calibration-right-eye-rejection-20260914/neighbors-debug.log` identifies
the actual rejection: right-eye native model masks for exposures 869–871 have
11,346, 11,412 and 11,447 component pixels. The shared 12% model-image area floor
requires 11,796; the left masks at 11,901–12,007 barely pass. This is a size gate,
not absence of an eye mask or a failed camera capture.

The shared cold mask-fit floor is now 10% (9,830 pixels), retaining all shape,
RAW support, source-time and calibration gates. A prior 5% experiment admitted
small fragments; that broader relaxation is not repeated. The small-iris
regression now covers an ellipse between the old 12% and new 10% limits. This
applies through the existing shared fitter rather than a calibration-only bypass.

Coupled CPU replays use the same fixed candidate camera per recording as the
12% baseline. All 16 previously admitted upper-right sources change from
left-only to both-eye geometric contributions. Native RAW overlays for exposures
869–871 and actual 3D rays were inspected: the right limbus is recovered, while
pupil/reflection ambiguity remains visible. `held-targets.png` displays the
paired leave-one-target-out results, including failures and missing targets.

| Recording | 12% / 10% accepted held predictions | 12% / 10% mean held error | Collector with 10% |
| --- | --- | --- | --- |
| Latest, 5488/5712 above | 2 / 7 | 0.1722 / 0.0732 | All eight presented targets complete |
| Earlier, 5712/5488 below | 8 / 8 | 0.06744 / 0.06787 | All nine complete; native finalizer accepts |

The latest held bottom-center error is now 0.0262; upper-right improves from
0.2528 to 0.1941 but remains poor. Means have different fold coverage in the
latest comparison, so the full per-target report matters. Errors are normalized
screen-coordinate distances to calibration cues, not measured fixation truth.
The center fold requires center coverage and the latest ninth cue is absent.

Source-matched outer projection coverage rises from 394/416 to 426/426 in the
latest clip, and 227/227 to 229/229 in the earlier one. On shared sources, latest
mean outer residual to baseline RAW supports is 0.7614/0.7406 versus
0.7625/0.7358 px; independently scaled SN-FEIDA changes are small and mixed.
Earlier common-source residual and independent area changes are numerical noise.
No human contour truth is available. Extra worker load and actual result timing
remain in these replays, rather than assuming restored masks are free.

Both candidate-camera acquisition probes pass. The 79 calibration tests and 67
outer-analysis tests pass (5 and 3 optional ignored respectively; suites may
overlap). The viewer builds and the source-tree audit passes. The rebuilt viewer
is preserved in this output directory; the running camera-owning viewer has not
been restarted. Camera intrinsics and personal calibration settings remain
unchanged. No training or model replacement was performed.

The additional 4000/4000 above-eyes replay with the 10% floor still completes
only seven targets; target 8 remains unsupported and no held-target prediction
is available under the unchanged coverage gates. Its acquisition probe passes.
This separates the mask-area dropout from the remaining intrinsics sensitivity:
the shared size fix alone does not make the latest recording calibrate with the
nominal camera. A recorded eight-target summary fit is not a completed nine-target
M calibration.

### Measured-pupil ablation: useful cue despite reflection failures

`outputs/calibration-pupil-ablation-20260914` tests
`BUTTERCUP_JOINT_CALIBRATION_OMIT_PUPIL=1` in the ignored native bridge corpus
probe. This is test-only: the measured pupil proposal is removed before shared
RAW boundary extraction; original typed worker output and cache remain intact.
Native reports attest `pupil_evidence_ablation`, and playback preserves it in
saved provenance. Unknown/mixed modes and claimed omissions with used pupil arcs
are rejected. Camera/precision comparisons cannot mix pupil-ablation modes.
No production pupil setting or weighting changed.

Both runs use the 10% outer-mask floor, existing CPU workers, recorded arrivals,
actual native result availability and the same candidate camera per recording.
Both acquisition probes pass. Only OuterLimbus arcs remain used in the omission
trial. Model-projected pupil ellipses may still appear in the visualizer; they
are prior-conditioned fitted shapes, not measured pupil observations.

| Recording | Measured pupil mean held error | Pupil omitted mean held error | Available held predictions |
| --- | ---: | ---: | ---: |
| Latest, 5488/5712 above | 0.0732 | 0.0909 | 7 in both |
| Earlier, 5712/5488 below | 0.0679 | 0.0652 | 8 in both |

Latest upper-right improves from 0.1941 to 0.1554, but upper-left worsens from
0.0625 to 0.1852, lower-left from 0.0752 to 0.1349, and bottom-center from 0.0262
to 0.0669. Target completion is unchanged: latest eight presented targets,
earlier all nine with accepted native finalizer. This rejects globally removing
pupil evidence as an improvement for the latest recording. Selective handling
of pupil/reflection ambiguity remains an experimental direction, not a validated
replacement or calibrated quality probability.

Source-matched outer projection coverage remains 426/426 and 229/229. Latest
mean baseline-RAW-support residual improves from 0.7676/0.7404 to 0.7424/0.6603 px;
mean independent SN-FEIDA steps improve from 0.01124/0.01227 to 0.01058/0.01153.
Those apparent contour/area gains accompany WORSE held gaze error. Earlier area
and outer residual changes are smaller and mixed. No human contour labels or
independent fixation truth are available; recorded cues remain the target proxy.
The neighboring RAW overlays, native rays and per-target comparison were produced;
RAW and native rays were visually inspected.

The original pupil-enabled control reproduces its features, native collector,
fit and held-target results exactly. Three malformed-ablation CLI checks pass,
79 calibration tests pass (five optional ignored), the viewer builds and the
audit passes. No running viewer, camera ownership, intrinsics, personal
calibration or model weights were changed. Latest target 9 remains absent.

### Half-weight pupil support sensitivity

`outputs/calibration-pupil-half-weight-20260915` tests
`BUTTERCUP_JOINT_PUPIL_HALF_WEIGHT=1` in test builds only. The native joint model
halves the pupil arc weight after computing its ordinary bounded support mass;
limbus weights, measured positions, uncertainty and calibration gates are not
changed. This is mutually exclusive with measured-pupil omission. Reports and
saved provenance identify `pupil_evidence_ablation: half-weight`; production
binaries do not apply this experimental weighting.

Exact current paired-source comparisons verify 2,946 pupil / 1,331 limbus arcs
in the latest clip and 1,756 / 618 in the earlier clip. Every matched pupil arc
has half its prior weight; every matched limbus weight is unchanged. Coordinates,
sigma and support lengths are identical on these matched arcs. There are eight
unmatched arcs per arm in the latest comparison and one per arm in the earlier,
retained explicitly rather than silently treating them as matches. Held earlier
publications must be excluded before making this source comparison.

| Recording | Full / half mean held error | Held predictions | Native collector |
| --- | --- | ---: | --- |
| Latest, 5488/5712 above | 0.0732 / 0.0631 | 7 in both | Eight presented targets complete |
| Earlier, 5712/5488 below | 0.06787 / 0.06691 | 8 in both | All nine complete; finalizer accepts |

The latest upper-right improves 0.1941 -> 0.0780 and right-center 0.0655 -> 0.0084.
Lower-left worsens 0.0752 -> 0.1263 and bottom-center 0.0262 -> 0.0705. The candidate
therefore improves aggregate error but is not a uniform fix. It remains a test
candidate rather than a promoted production weighting. These are exploratory
Rob-only recorded-cue errors under fixed candidate cameras, not independent gaze
truth or an unseen-session evaluation.

Outer projection coverage stays 426/426 and 229/229. Mean shared-RAW outer
residual improves slightly in both eyes of both clips; independent SN-FEIDA mean
steps also decrease slightly (latest 0.01124/0.01227 -> 0.01102/0.01176; earlier
0.01151/0.01461 -> 0.01139/0.01455). Human contour truth is still absent. Native
source-matched RAW neighbors and the per-target prediction comparison were
visually inspected. Both coupled CPU probes and scorers finish successfully;
79 calibration tests pass, five optional tests are ignored, and build/audit pass.
No running viewer, camera ownership, personal calibration or production pupil
weight was changed. The latest ninth target remains absent.

### Transfer to another complete calibration recording

`outputs/calibration-transfer-20260915` prepares and replays
`sam31-mouse-3d-1789347706-526705558`, the nearest additional complete nine-target
recording before the earlier evaluated clip. It originally completed collection
but rejected the 2D affine. The archive contains 468 native ROI frames / 234 pairs,
below-eye mounting, one clock lineage, and two initial frames without prior scale.
Camera and pupil-weight trial choices were written before replay in
`trial-plan.json`; none was tuned to this recording. It is the same user and a
nearby session, so this is transfer between recordings, not an independent setup
or population test.

All arms use the shared 10% mask floor and coupled CPU processing on CPUs 4–7.
All three complete the native nine-target collector, accept the native finalizer,
and save/reload their mappings. The original pupil-enabled camera setting is
4000/4000; the fixed transferred candidate is 5712/5488 with principal 4000/3000.

| Camera / pupil weight | Accepted held predictions | Mean held cue error | Reloaded source mappings |
| --- | ---: | ---: | ---: |
| Original / full | 8 | 0.08016 | 214 |
| Candidate / full | 8 | 0.05910 | 210 |
| Candidate / half | 8 | 0.06022 | 211 |

The center holdout still lacks required center coverage by design. Different
qualified mapping counts remain explicit; acceptance does not imply every frame
is supported. Reload mapping differences are below 1e-15. Target cues are not
independently measured fixation truth. The half-weight trial does not improve
this recording over the full-weight candidate, reinforcing the decision to
leave production pupil weighting unchanged.

Both eyes retain 233 outer projections in each arm. The fixed camera increases
mean residual to original-camera RAW supports from 0.802/0.786 to 0.870/0.843 px,
while independent SN-FEIDA mean steps change 0.01054/0.01115 -> 0.01049/0.01094.
Against its own full-weight baseline, half pupil weight slightly improves outer
residual and area stability but slightly worsens held gaze error. Per-comparison
RAW supports differ; do not combine residual baselines across comparisons.
Motion estimation excludes each current/previous nominal 2D limbus plus margin,
uses source-ordered native receipts, and never uses candidate radius as scale.
No human contour labels were added.

Native source-matched exposures 2071–2073 and their native 3D rays were inspected.
Both eyes contribute at upper-right; pupil/reflection ambiguity remains visible.
`held-targets.png` gives all per-target predictions and unavailable folds. The
three probes, scorers and two native geometry comparisons finish successfully.
This step reuses the previously tested frozen binaries; no source implementation,
live viewer, camera settings, model or personal calibration was changed. The
latest eight-target archive still cannot establish its absent ninth visit.

### Explicit binocular support in admitted playback samples

Playback now stores `contributing_eyes` and `direction_supported` on every
admitted source, taken from its first fully qualified exact paired publication.
Each visit also reports `admitted_geometry_support`: total admitted sources,
both-eye contributions, both directions supported, and one-eye contributions.
The flags are parsed as two explicit booleans; missing flags do not silently
become negative evidence. These counts describe the admitted source stream,
not just the final robust-cluster inliers or camera frame-pair completeness.
They do not change acceptance or claim calibrated probabilities/accuracy.

`outputs/calibration-binocular-audit-20260915` replays the prior latest baseline,
latest size fix, earlier complete session and transfer session. All four retain
exactly their previous target features, fit, held-target results and native
collector outcomes. The two complete nine-target recordings have both-eye
contributions AND both supported directions on every admitted source, for every
target. This strengthens the evidence that these are binocular calibrations,
rather than merely paired camera delivery with one-eye geometry.

Before the size fix, the latest upper-right visit admitted 16 one-eye sources.
After it, the visit admits 14 sources and all 14 have both-eye contributions and
both supported directions. The different admitted count remains explicit;
restoring both eyes does not guarantee the same uncertainty/admission decisions.
Latest lower-right has 33 of 42 admitted sources with both directions supported,
lower-left 34 of 40, and the first bottom-center visit 22 of 24. Both eyes
contribute on all of those sources; the remaining cases support the selected
eye's direction only. The repeated bottom-center visit has 13 of 13 with both
directions supported. The ninth target is still absent.

`binocular-support.png` was inspected and visualizes these distinctions.
Eighty calibration tests pass (five optional ignored), including the distinction
between paired contributions and two supported directions. Build and source-tree
audit pass. This is a reporting change only; the running viewer and personal
calibration settings remain unchanged.

### Using explicit intrinsics through the production stereo adapter

The viewer now accepts:

```sh
--joint-camera-intrinsics '[5488,5712,4000,3000]'
```

The equivalent environment variable is `BUTTERCUP_JOINT_CAMERA_INTRINSICS`.
Values are `[fx,fy,cx,cy]` in native full-sensor pixels, never ROI/model/display
pixels. This configures the shared joint stereo adapter, including its viewer
and calibration consumers. It does not assert a measured camera calibration or
modify legacy pipelines that do not use the joint solver. Default intrinsics
remain `[4000,4000,4000,3000]`; no candidate is silently selected from mounting.
The pinhole values are validated and frozen for the process at startup, before
camera cooperation/capture initialization. Restart to change them. Invalid/non-
finite values and nonpositive focal lengths are rejected.

Startup logs report the exact values. Recording scenes and configuration events
include `joint_camera_intrinsics`, and joint publications retain actual numeric
intrinsics with uncalibrated provenance. Existing personal gaze mappings still
reload as requested; use M to recalibrate after changing intrinsics. A standalone
saved gaze-calibration JSON is not an intrinsic calibration file; reproduce its
explicit startup configuration or retain the offline companion provenance.

`outputs/calibration-explicit-intrinsics-20260915` verifies the production path:
the latest coupled CPU replay sets only `BUTTERCUP_JOINT_CAMERA_INTRINSICS`, with
`BUTTERCUP_JOINT_CALIBRATION_INTRINSICS` absent. All 426 current paired native
publications reproduce the earlier test-only camera trial exactly in numeric
intrinsics, source keys, targets, eye centers, normals, gaze, projections, arcs,
cost and posterior. Only the descriptive provenance text changes. The recorded
collector still completes all eight presented targets with unchanged held-target
errors. No ninth target is invented and no new live calibration is installed.

CLI help/valid parsing and invalid CLI/environment startup checks pass. Eighty
calibration tests, one intrinsics parser test, seventeen recording tests and
nineteen camera-cooperation tests pass (suites may overlap; five optional
calibration tests ignored). The latter retain the socket-boundary and process-
fatal ownership checks. The viewer builds and the source-tree audit passes.
The running camera-owning viewer is unchanged; the rebuilt executable is
`outputs/calibration-explicit-intrinsics-20260915/calibration-viewer`.

### Recorded camera provenance in archive preparation and playback

`outputs/calibration-camera-provenance-20260915` verifies that preparation,
playback reports and reload companion provenance inventory camera declarations
from recording-start configuration and configuration-change events. They report
missing declarations, partial declarations and multiple distinct cameras
explicitly, separately from the intrinsics selected for an experimental replay.
Recorded values must use native sensor pixels and valid positive focal lengths.
This inventory does not automatically select or infer replay intrinsics.

The latest archive has 2,518 configuration records without intrinsic declarations:
its camera remains `not-recorded`, rather than being retroactively assigned the
current engineering default. Preparation and playback inventories agree. The
fit, leave-one-target-out results, visits, native collector and selected target
features exactly match the preceding production-camera replay. Eighty-two
calibration tests pass, five optional tests remain ignored, the release viewer
build succeeds and the source-tree audit passes. No live viewer, startup setting
or personal calibration was changed.

### Separate pupil sensitivity from calibration mapping selection

`outputs/calibration-mapping-basis-20260915` repeats the native measured-result-
availability collector on the existing full/half-pupil CPU streams for latest
1789432526-545550713, earlier 1789347789-185587461, and transfer
1789347706-526705558. No model inference or geometry was rerun or changed here.
The new mapping-basis diagnostics fit both bases using the same training-only
accepted plane, native robust affine and shared-support gate. Held cues affect
neither training nor selection; unsupported alternatives remain diagnostic only.
Before the selection change, all six native outcomes match their references,
excluding explicitly added reporting fields (see `baseline-parity.json`).

The old shared selector preferred an affine on projected direction whenever it
passed. For the upper-right holdout, both that mapping and the perspective-aware
display-intersection mapping pass, yet the former extrapolates worse. Changing
the shared selector to prefer the supported display-intersection mapping reduces
full-pupil latest upper-right error from 0.194121 to 0.124518, and earlier from
0.1269 to 0.0717. Direction mapping remains a gated fallback. Existing saved
mapping bases are preserved; this change applies when fitting new calibrations.

| Stream | Previous mean held error | Perspective preferred |
| --- | ---: | ---: |
| Latest, full pupil | 0.073212 | 0.063269 |
| Latest, half pupil | 0.063063 | 0.063063 |
| Earlier, full pupil | 0.067874 | 0.060972 |
| Earlier, half pupil | 0.066906 | 0.060240 |
| Transfer, full pupil | 0.059100 | 0.059100 |
| Transfer, half pupil | 0.060223 | 0.060223 |

Errors are Euclidean distance in independently normalized screen X/Y coordinates,
**not physical screen-diagonal fractions**. Latest means cover seven accepted
holdouts; the two complete recordings cover eight each. Center holdout still
fails mandatory coverage; latest target nine is absent. No available held-out
target regresses under the selection change. These are same-user, limited corpus
comparisons, not independent population accuracy or measured gaze truth.

This explains nearly all the previous mean benefit of half pupil weighting on
the latest recording. Normal pupil weight stays in production. Neighboring RAW
1018–1020 and native rays were visually inspected: right-eye pupil measurements
are absent in 1019–1020 while the plotted pupil curve remains prior-conditioned;
left-eye support is partial and adjacent to a strong screen reflection. Pictures
do not justify a target-specific pupil heuristic. `mapping-comparison.png` was
also inspected. The native geometry, RAW residual and independent-scale SN-FEIDA
results are inherited unchanged from the exact source streams; human localization
labels remain unavailable for these examples.

All six replays preserve target visits, selected geometry features and native
collector outcomes exactly. Both complete recordings still finish nine targets
and pass native save/reload, with maximum cursor round-trip difference below
1e-15. Latest completes eight presented targets and cannot finalize nine.
Eighty-three calibration tests pass (five optional ignored), including a recorded
both-bases-supported regression and a held-cue isolation check. Desktop mapping
checks and the source-tree audit pass; the release viewer builds. The frozen
new executable is `perspective-viewer` in this experiment directory. The running
camera-owning viewer and live personal calibration remain unchanged.
