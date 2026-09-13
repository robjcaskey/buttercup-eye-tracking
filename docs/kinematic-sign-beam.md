# Conventional source-timed sign beam

This is an isolated CPU-only experimental tracker and native corpus replay,
not a live gaze authority or a demonstrated fix for the calibration failures.
It consumes prior estimates, loads no learned model, performs no training, and
does not change the input limbus ellipse. Its ordinary inference path is std-only
Rust in `src/eye_scene_model/sign_kinematic_beam.rs`; the replay is
`src/bin/buttercup_kinematic_sign_replay.rs`. The upstream recorded estimates may
have learned or joint-binocular ancestry. Diagnostic consumption does not certify
those models for bootstrap training, and per-ROI replay does not make the
upstream joint solver a monocular pipeline.

## State and observability

One instance owns one physical ROI. Source identity includes a stream and frame
counter plus an exact integer nanosecond source timestamp. Missing clocks, held
results, repeated identities, repeated timestamps and late results cannot refresh
support. A new stream resets the history; invalid/explicit missing geometry
clears it. Wall-clock delivery is never substituted for exposure time. The adapter
counts metadata snapshots without an asynchronous result separately: these are
not source-bound blinks and must not advance a watermark beyond pending answers.

Each current conic has two normal/pivot hypotheses. The tracker retains at most
four states per sign, including distinct predecessor histories and a fresh
zero-derivative restart on every admitted observation. It explores both current
signs from both preceding signs and fixation/saccade modes, at most 32 transitions.
Historical cost decays by 0.65 per admitted update. A twelve-entry, 1.25-second
support history and 0.75-second maximum interval bound memory and stale evidence.
These are engineering choices, not fitted physiological distributions.

Independent source-matched transport compares the hypothesized effective pivots
in sensor pixels. The pivot is allowed to move: both endpoint uncertainties,
transport uncertainty and an explicit nonrigid allowance enter its residual.
The input image similarity is not a 3-D head rotation. Only an independently
supplied camera-frame rotation vector removes head rotation from normal motion.
When it is absent, the angular derivative is explicitly camera-relative and
cannot distinguish head rotation from eye rotation. Torsion about the normal is
unobserved. ROI translation disappears by expressing points in sensor coordinates.

The selected direction is the current input normal, never a temporal average.
There is no branch transition penalty or absolute sign lock. A current,
independently discriminating pivot/anchor residual and recent supporting source
intervals authorize a sign; the bounded beam must also favor it. Smoothness alone
cannot authorize reflection-symmetric alternatives. A frontal separation below
0.05 radians abstains. Support counts and margins are heuristic, not independent
trials or calibrated probabilities; neighboring intervals share measurements.

## Angular priors and physical limits

Normals use camera x-right, y-down and z-toward-camera. Angular velocity is the
minimal geodesic rotation of the normal, in radians/second. Acceleration and jerk
use actual source-interval midpoint separations; previous derivative vectors are
rotated into the current camera basis when independent head rotation exists.
Degrees appear only in reported comparisons and the documented default onset.
Rodrigues rotation uses the component cross-product convention: positive Z
rotates +right toward +down. A pose-provider adapter must convert handedness
and rotation direction explicitly.

The default velocity penalty starts softly at **1200 degrees/second**
(20.94395 radians/second), with a low-weight saccade mode. This is a deliberately
loose engineering envelope, not a measured human maximum or a rejection bound.
In a primary human whole-body gaze-reorientation study, eye peak velocity was
448 ± 135 degrees/second and head acceleration varied substantially; those task
averages do not provide population-wide hard limits.
[Anastasopoulos et al., 2015](https://link.springer.com/article/10.1007/s00221-015-4238-4).
Another primary human study reports mean eye maxima near 292 degrees/second for
10-degree shifts and 398 for 20-degree shifts, with head contribution depending
on target angle. This supports separating eye/head motion and retaining a wide
envelope, not imposing those averages as limits.
[Eye-head coordination during lateral gaze](https://pubmed.ncbi.nlm.nih.gov/7468181/).

Acceleration and jerk penalties are available but **disabled by default**.
Primary human trajectory measurements rejected minimum-jerk and several other
minimum-derivative shapes as general saccade models; we therefore do not force
minimum jerk or claim that a sharp saccade must be an error.
[Harwood, Mezey and Harris, 1999](https://pubmed.ncbi.nlm.nih.gov/10516327/).
Sparse approximately 100ms recorded estimates cannot recover a true saccadic peak:
sampling methodology materially changes measured kinematics.
[Gibaldi and Sabatini, 2021](https://pubmed.ncbi.nlm.nih.gov/32643061/).
Mass alone supplies no acceleration bound. No individual torque/inertia estimates
are available, so the implementation has no mass-derived acceleration prior.

The observed 51–56 degree candidate jumps over approximately 100–300ms in recent
recordings are not automatically biomechanically impossible. Smoothness cannot
identify their physical sign without stronger observations.

## Reproducible corpus evaluation

Build and test without optional GPU dependencies:

```sh
cargo test --offline --no-default-features --bin buttercup_kinematic_sign_replay -j 2
cargo build --offline --profile live --no-default-features --bin buttercup_kinematic_sign_replay -j 2
```

Announce shared low-priority CPU/block claims with the host `agent-coord` protocol
before builds/replays. Use fresh output names; replay output uses `create_new`.
Independent corpus groups can run concurrently with one CPU process each:

```sh
data/target/live/buttercup_kinematic_sign_replay outputs/sign-historical.json \
  --labels data/labeled-corpus --labels outputs/labeling \
  outputs/cross-video-corpus-v3/extracted
data/target/live/buttercup_kinematic_sign_replay outputs/sign-calibration.json \
  --motion-sidecar archived-features-only.jsonl \
  outputs/student-calibration-diagnosis.1MjGcu \
  outputs/sign-acquisition-20260907.mhqqmo outputs/sign-audit-20260907/capture
```

The archival-feature baseline above intentionally names an absent native-motion
sidecar. Confirm that `archived-features-only.jsonl` is absent in the selected
captures; leaving out that option now consumes the subsequently added native RAW
sidecars and reproduces the RAW follow-up instead.

The native tool discovers extracted `frames.jsonl`/`predictions.jsonl` pairs,
preserves file hashes, checks source-index correspondence, excludes ambiguous
same-ROI/timestamp keys rather than overwriting them, and resets adjacency at
stream changes. Exact index-and-prediction copies are deduplicated; partially
overlapping captures are not claimed to be RAW-deduplicated. Older missing epoch
metadata uses isolated archive-local namespaces and remains explicitly unattested.

Native per-frame semantic ellipses become conventional weak-perspective circular
limbus hypotheses. Recorded contact geometry is reconstructed by the existing
circle convention, then both transverse signs are exposed. These counterfactuals
are **not necessarily the two exact off-axis pinhole solutions** of the joint
solver. The established 1.83 globe/limbus ratio gives an uncertain effective depth;
the replay supplies pivot sigma `1 + 0.04*depth` pixels. This is not a personalized
anatomical measurement. Old semantic snapshots have no separate measurement
freshness attestation: distinct timestamps do not prove a newly measured ellipse.
For joint-conic contact exports specifically, area plus the chosen perspective
normal does **not** encode the actual observed conic. Setting minor/major to
normal.z and inferring the axis from transverse normal components constructs a
surrogate; it must not be described as recovering the actual frozen joint ellipse.
The exterior exclusion based on that enlarged surrogate has the same limitation.

Six matched arms use unchanged geometry: surrounding-tissue transport; no
transport; an intentionally iris-contaminated negative control; and tissue
transport at strides 2, 4 and 8. The latter directly match the preceding admitted
source interval, do not sum overlapping weak votes, and explicitly report
intentional source skips. They trade publication cadence/latency for longer
displacement baselines and still reject intervals over 750ms.

Archived native general-layer track IDs supply bounded correspondences (at most
64 pairs, at least eight robustly consistent points, spatial extent required).
The tissue arm excludes both endpoints inside a 1.3-times expanded iris ellipse
plus ten pixels. Native skin/background motion remains a defeasible transport
proxy; candidate independence does not establish head rigidity. The negative
control deliberately removes this exclusion. Neither arm invents 3-D head pose.

An optional `exterior-motion.jsonl` beneath each capture can supply independently
computed native RAW exterior transport. Required fields are `roi`,
`from_source_ns`, `to_source_ns`, `from_sequence`, `to_sequence`, `stream_epoch`,
`center_sensor_px`, `translation_px`, `scale`, `angle_rad`, `sigma_px`, `support`,
and `reliable`. The map is `p' = scale*R(angle)*(p-center)+center+translation`.
Only exact contiguous endpoints in the same epoch compose, at most 96 links and
750ms. Uncertainty adds conservatively after scale transport; missing links are
unknown, never identity. The sidecar hash is retained. This allows the existing
native RAW exterior tracker to supply measurements without changing live state.
Use `--motion-sidecar exterior-motion-circle.jsonl` for a separately retained
enclosing-circle control, or another explicitly named local sidecar basename.

Human-label comparison verifies label RAW bytes against the exact indexed capture
stream, ROI, sequence and sensor origin. Reviewed visible edge/midpoint points
are scored against 3600 samples of the unchanged ellipse rim, with the
discretization-error bound reported. Existing historical labeler provenance is
preserved; assistant-visual and backup labels are excluded. This is localization
evidence, not a signed gaze label. Missing raw/labels and unmatchable annotations
are not filled in.

SN-FEIDA follows [the canonical definition](flat-tire-area-and-motion.md). Only
supported surrounding-feature scale intervals contribute an area-change diagnostic:
`delta log(area) = 2*log(a_current/a_previous) - 2*log(scale_transport)`.
No candidate-radius normalization is used. Since all arms consume the same
ellipse, localization and SN-FEIDA differences are exactly zero by construction.
Neither a constant area nor an abstaining tracker proves good gaze.

## Development results and failures

The inspected six-arm run is under `outputs/kinematic-sign-corpus.veixuV`, reports
`historical-v3.json` and `calibration-v3.json`, with per-source `.cases.jsonl`
and resource-claim logs. It covers **31 capture directories, 20,685 native frame
index rows**: 23 historical extracted clips (2,932 rows), four previous sign
acquisition clips plus one longer sign-audit clip and three recent calibration
clips (17,753 rows). This is a broad existing single-user diagnostic subset, not
every archive on disk and not a held-out human sign-accuracy benchmark.

The historical group includes the ROI-jump `1788101004`, limbus-band/night
`1788101016`, manual correction `1788259773`, eight low-light clips from
`1788262955` through `1788262970`, and earlier/later failure or motion clips.
The report lists every exact directory, hashes and per-eye exclusions. Both eyes
exist in the corpus, but historical usable semantic-ellipse snapshots are only
from ROI1; ROI2 absence is reported. Both eyes provide recent contact estimates.
The low-light/glare/occlusion labels identify difficult examples, not controlled
illumination or focus interventions. Abrupt lighting changes and anatomical
head motion are not independently annotated throughout this subset.

| Matched source set | Full-rate tissue | No transport | Contaminated control | Tissue stride 2 / 4 / 8 |
| --- | ---: | ---: | ---: | ---: |
| Historical, 1,393 distinct-source geometry snapshots | 2 resolved | 0 | 2 | 0 / 0 / 0 |
| Calibration/motion, 7,528 distinct first-publication contact sources | 0 | 0 | 0 | 0 / 0 / 0 |

Historical full-rate rejection/support breakdown: 94 late source rows; 802 without
transport; 443 with insufficient residual separation at the stated uncertainty;
34 with poor pivot support; 18 with insufficient sustained support; two resolved.
Those two isolated ROI1 decisions are sequences 144 and 204 in
`1788335131-540962832`; no labels establish that either sign is right, and there
are no adjacent resolved pairs. Their zero reported flips is **not improvement**.
The three sparse arms receive 701/354/182 sources, intentionally skipping
692/1039/1211 respectively, with 269/123/4 supported transport intervals.
They regress resolved coverage to zero. At roughly 100ms source cadence, stride8
usually exceeds the 750ms interval allowance; sparse scheduling is not a generic
solution to low sample rates.

The calibration/motion group has **zero archived feature-transport intervals**
even in the contaminated control. Its 6,138 held publication repeats are rejected;
4,059 raw metadata snapshots lack usable geometry and 28 rows lack an eligible
archived source. The recorded baseline marks 6,120 first-publication contact
sources resolved. Replacing it with this independent-evidence-only tracker would
therefore destroy coverage. No automatic live integration is justified.

For the latest `1789227354-277611143` left eye, sequences
2706→2708→2711 and 2857→2858→2860 are all `Ambiguous`, with zero independent support
and equal beam costs for the two reflections. These match recorded large
candidate reversals, all already unresolved by the joint posterior. Native RAW
inspection found stronger upper-lid occlusion at 2708 and nearly unchanged
visible eye geometry around 2858. These are useful failure fixtures, not verified
physical meridian crossings. The recent3 per-capture source-to-publication
95th-percentile lags are approximately 388/292/292ms. The upstream joint solver
remains a coupled provider, and its `sign_epoch` is not a branch-flip counter.

The label comparison matched **16 reviewed RAW-verified label files / 205 visible points**, including
canonical temporal paired-boundary labels and separately reported historical
eye-tagging labels. Old semantic geometry is often badly localized: per-frame
median distances range from 3.66px to 123.46px; the maximum visible-point distance
reaches 157.20px. These are unchanged for every sign arm and must not be hidden
by reporting stable area or a favored branch. No new fitting/localization claim
is made. Recent signed 3-D/fixation labels, calibrated scale and independent
3-D head pose are absent.

The full-rate tracker averages about 1.65 microseconds per historical call and
0.85 microseconds per calibration/motion call on shared resources, excluding JSON
IO and feature fitting. All arms stay within eight states and 32 transition
evaluations. This demonstrates a small bounded CPU step, not isolated benchmark
latency or a complete real-time pipeline budget.
The final combined native test suite passes 23 tests. Source-tree audit and the
optimized CPU-only build pass. `historical-final.json` repeats the historical
results with embedded compiled tracker/replay source SHA-256 identities;
`final-control-tests.log` preserves the final test receipt.

Synthetic validation now includes 1,152 recovery scenarios / 27,648 observations,
32 motion-only crossing scenarios / 992 observations, and 48 slow-motion/control
scenarios / 752 admitted observations. Explicit physical-normal checks cover x/y,
both signs, source intervals 8/20/100/300ms, wrong initial support, permutations,
ROI changes, missing/uncertain head evidence, saccades and nonrigid nuisance.
With 20ms sensor cadence and 4px pivot uncertainty, coherent small motion is
unresolved at adjacent intervals but resolves using direct80ms or160ms intervals
after 240ms or480ms, respectively. Head-only motion, uncertain drift and no-head
reflection controls remain ambiguous. These constructed successes explain the
longer-baseline idea; the actual sparse corpus arms did not reproduce a benefit.

The next necessary observation is source-matched exterior RAW motion or another
independent sign cue with explicit uncertainty, followed by the same frozen-input
comparisons. The available native exterior tracker can provide that measurement
through the optional sidecar. Lowering a residual gate to force these unlabeled
clips to produce more decisions would not establish correctness.

## Native RAW exterior follow-up

`recent-raw-exterior-v4.json` adds native exterior RAW motion to all three recent
clips, using the existing `--offline-stereo-motion-export` implementation and
first-publication frozen estimate-source inputs. The source-index/exclusion
receipts, full source keys, native RAW hashes and exported correspondence records
are retained beneath each recent capture. No SAM inference or target labels enter
this exterior matcher. Reliable direct intervals are 194/431, 357/378 and423/441
respectively; all source intervals stay below 750ms. This is 2-D skin/background
support outside the declared enlarged surrogate, not independently measured
rigid head pose or exact observed-limbus exclusion.

The beam consumed all 974 reliable intervals across 1,256 distinct contact sources.
It still resolved **zero**, against 265 source-matched baseline-resolved contacts.
Full-rate support failures were 282 missing intervals, 950 insufficient residual
separation at the stated allowance, two poor-pivot cases with a separating
residual, and 22 insufficient sustained-support cases. The 950 nondiscriminating
cases include saturated failures in which neither pivot fits; they are not 950
instances of successful low-error ambiguity. Stride 2/4 received 423/179 supported
composed intervals and still resolved zero; stride8 exceeded the maximum gap.
This rejects both a missing-measurement-only explanation and a claim that sparse
sampling already repairs the problem.

At 2706→2708→2711, exterior matching is unreliable around the lid occlusion and
cannot vote identity. At 2857→2858, exterior motion has 15 inliers and about
−0.074px vertical translation; the next interval has 24 inliers and +0.059px
vertical translation. Despite this extra observation, both reconstructed-pivot
costs saturate at 4.0 on 2858 and 2860. The tracker correctly reports unsupported
surrogate geometry instead of pretending that angular smoothness identifies a
valid branch. A complete perspective-conic candidate pair with geometry support
is needed to investigate this failure; reflecting the selected ray alone cannot
provide it. Existing `circle_pose_hypotheses` and `circle_normal_hypotheses` in
`src/conic_solver/joint.rs` already provide the conventional perspective
decomposition; a future adapter should use those outputs with their actual
source conic instead of reimplementing optical-axis reflection.

The added measurements therefore do not justify live promotion. Their measured
native extraction cost on the latest clip averaged 6.89ms/source frame, with a
9.32ms maximum under shared CPU use; the beam's roughly 1.90µs update is only one
small part of the total diagnostic pipeline. Calibration maps, viewport/camera
state and all model defaults were untouched.

## Enclosing-circle exclusion control

`recent-circle-final.json` repeats the same 1,256 source contacts and exact native
RAW inputs using `exterior-motion-circle.jsonl`. The exclusion uses the same
center and major radius but sets minor=major, so selecting a transverse sign,
tilt or in-plane angle no longer changes which pixels are eligible for exterior
tracking. It remains an enclosing-circle image exclusion, not a true anatomical
head measurement. Original surrogate-exclusion files are retained separately.

Reliable intervals become 160/431, 354/378 and 419/441, or 933 total—41 fewer than
the surrogate-ellipse exclusion. All six sign arms still resolve zero. Stride2/4
retain 405/176 intervals; stride8 retains none. In the clean reversal episode,
both pivot costs still saturate at 4.0 on 2858 and 2860. Thus the failure is not
removed by making the exterior pixel selection independent of normal and angle.
The control report embeds the exact tracker/replay source hashes and sidecar
hashes and reports no replay errors.

```sh
data/target/live/buttercup_kinematic_sign_replay outputs/sign-circle-control.json \
  --motion-sidecar exterior-motion-circle.jsonl \
  outputs/student-calibration-diagnosis.1MjGcu
```

The implemented tracker is tested and bounded, but these results reject its
promotion as a working calibration sign correction. A future experiment needs
the source-attested actual conic and its complete perspective candidate pair,
with nuisance/geometry uncertainty; the current selected-ray surrogate cannot
stand in for that information.
