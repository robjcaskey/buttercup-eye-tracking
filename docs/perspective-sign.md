# Perspective sign shadow experiment

This isolated CPU Rust experiment does not establish a deployable sign fix.
`buttercup-perspective-sign-replay` evaluates exact perspective alternatives on
recorded inputs without loading a learned model, training, changing the source
ellipse, or changing live viewer behavior. A fresh preferred candidate is not
verified gaze coverage. Conditional motion support is not signed 3-D truth.

## Geometry and source contract

Recent recordings preserve joint circle center, surface normal and camera
intrinsics. Their contact `rectified_area_px2` preserves the fitted ellipse's
major radius. A bounded 64-step inversion of the existing `ProjectedCircle`
forward model recovers the radius conditional on that saved pose and intrinsics.
The resulting sensor-space ellipse center must agree within 0.05px with the
separate saved near-point minus bucketed-radius times surface normal. This is
not recovery from area and selected normal alone, nor an optical-axis reflection.
The existing `circle_pose_hypotheses` supplies both exact normals and their
different circle centers per unit radius. Both forward-project to the same conic.

Archived joint source sequence, ROI, integer nanosecond timestamp, and hashed
clock lineage must agree with the native source index. Published/held repeats do
not refresh the tracker. Ambiguous index entries are excluded. Missing archived
geometry is not used to advance an asynchronous source watermark. Old semantic
ellipses are evaluated with explicitly assumed focal length 4000px, principal
point (4000,3000) and nominal radius 6mm; the pivot ratio formulation is invariant
to this arbitrary metric radius. Old contact-only archives lacking a saved joint
pose cannot provide the true conic and are excluded.

Exterior native RAW motion requires complete endpoint source keys, including
viewer session and ROI generation, plus matching native RAW SHA-256 hashes.
Missing/unreliable intervals are not identity transforms. Only direct matched
endpoint intervals are used. `--source-membership BASENAME` freezes the exact
native index membership and verifies every member's source key and RAW bytes.
`--motion-sidecar BASENAME` selects its independently computed motion. Optional
`--source-stride N --source-phase P` provide ordinal diagnostic scheduling; fixed
membership is preferred for matched native controls. All baselines use exactly
the same retained source subset as the candidate.

## Bounded inference and limitations

Each ROI retains both physical branches at five shared pivot-depth/limbus-radius
ratios: 1, 1.25, 1.5, 1.75 and 2. The ratio is shared across a retained history;
rescaling the upstream conditional circle center and radius changes no projected
pivot or score. It is not measured anatomical scale. The v5 dynamic program
uses a **common source window** for both alternatives, with no shorter restart
competing against a longer history. Five depths times eight last-three-branch
suffixes give at most 40 cells and 860 edge evaluations per recomputation.
Each cell carries separate residual-only and residual-plus-kinematics recurrences.
Suffixes suffice for velocity, acceleration and jerk edge costs. An exhaustive
enumeration test checks both additive optima, including a missing transport edge.

History is limited to 13 sources / 12 intervals and 1.25 seconds measured from
the first source, with a 750ms maximum source gap. Emitted derivative history
obeys the same time horizon. Image endpoint allowance is swept at 2, 4 and 8px; exterior uncertainty,
0.5px nonrigid slack and 2px/s source-time drift allowance are added. A local
secant allowance propagates a shared half-grid ratio cell through the complete
transition. This cancels common stationary depth error instead of paying the
entire depth range independently at both endpoints. It is a local approximation,
not a certified nonlinear bound or proof that off-grid nuisance is exhausted.

The independent objective is the mean Huber cost of normalized pivot residuals:
`rho(r)=r²` for `r<=1`, otherwise `2r-1`. Missing transports are masked identically
in both recurrences; no probability product or independence assumption is used.
The nuisance and branch history minimize this common-window mean. Maximum
residual on that selected path remains a diagnostic, not a minimax certificate.
Conditional support requires current transport, at least three supported window
intervals, current twin separation above 0.05rad, winning mean at most one,
absolute mean gap at least 0.15, and losing mean at least `2*winning_mean+0.15`.
These explicitly heuristic rank margins allow consistent 2px versus 6px histories
to differ even when both satisfy broad endpoint bounds. They are not calibrated
probabilities or exhaustive continuous-nuisance physical identification.

Geodesic angular velocity is in rad/s. Acceleration and jerk use successive
source-interval midpoint times, in rad/s² and rad/s³. Optional independently
provided source-matched head rotation transports normals and derivative vectors
to the current camera basis; a known/unknown head-reference transition resets
derivative order. Unknown head motion leaves camera-relative normal motion,
not identified eye-in-head motion. The component Rodrigues convention is positive
Z rotating right toward down; under a basis reflection `M`, an axial rotation
vector transforms as `det(M)*M`, unlike a polar point. Image similarity never
supplies this 3-D head rotation. No real corpus head-rotation measurements exist.
The optional vector is currently an exact input/oracle contract; it does not
represent uncertain 3-D head-pose support or establish anatomical eye/head separation.

The motion objective has a soft 1200deg/s onset and a low-order acceleration/jerk
cost with engineering scales 40,000deg/s² and 8,000,000deg/s³. A competing bounded
free-saccade mode caps the latter cost at 0.02. These are not measured population
limits or mass-derived bounds; sparse finite differences do not measure saccadic
peaks. In particular, primary trajectory measurements do not support forcing
minimum jerk as a universal saccade shape; see the primary sources and limitations
in [kinematic-sign-beam.md](kinematic-sign-beam.md#angular-priors-and-physical-limits).

A sufficient independent margin selects its winner. Otherwise provisional
readout uses a bounded joint-window score contribution, actual emitted-history
motion-order cost and a small geodesic continuity cost. Motion order therefore
can change an emitted candidate, but cannot create conditional sign support.
The selected normal is always an unchanged current perspective twin. JSON reports
both `fitted_path_kinematics` and `emitted_kinematics`; those histories can differ.
`motion_supported_under_model` remains separate from always-unverified `identified`.

## Validation interpretation

The initial marginal-cost readout regressed badly: 194 preferred steps above
30 degrees across the recent three clips versus 38 recorded baseline steps.
Adding a soft continuity readout reduced this to 13, but yielded no independent
sign evidence. Shared ratio history then supplied some conditional motion support
while regressing the noisy first clip. These rejected iterations are preserved
as `outputs/perspective-sign-recent-v1`, `v2` and `v3` JSON/JSONL receipts.

At left-eye source 2857→2858→2860, exact perspective self-history pivot residuals
are approximately 2–6px and opposite-history transitions about 94–100px. Both
coherent physical histories remain possible; suppressing the recorded reversal
does not establish which is anatomically correct. The old surrogate's saturation
was not a faithful representation of these true perspective alternatives.

Tests include off-axis conic roundtrips, both exact pose reprojections, metric
gauge invariance, candidate permutation, wrong initial branch recovery, meridian
crossing, a legitimate approximately 50-degree saccade, source freshness and
malformed source/RAW evidence. The broader synthetic sweep varies both axes,
signs, 8/20/100/300ms source cadence, 2/4/8px image allowances and an off-grid,
slowly moving pivot ratio. Its partial recovery must not be called full success.
The head-rotation negative control uses oracle projected-pivot transport and
wide nuisance support; it is not validation of real exterior matching under
head rotation. The narrow recovery fixture uses 0.2px endpoint support and is
more optimistic than the corpus.

Scale-normalized frontal-equivalent iris disk area (SN-FEIDA) uses unchanged
conic major radius and direct independently measured exterior similarity scale.
The scale is never the candidate's own radius. These are source-pair relative
area changes, not absolute anatomical area or a persistent calibrated scale.
Sign choice cannot improve the unchanged input boundary localization or SN-FEIDA.
Recent signed 3-D labels and calibrated head pose are absent. Historical semantic
localization failures documented in `kinematic-sign-beam.md` remain relevant;
the present replay does not rerun human-label scoring or claim hidden-rim accuracy.

Build/tests/replays use CPU-only no-default-feature Cargo targets, two build jobs,
the checked bulk-data target directory and announced shared low-priority resource
claims. Per-step timing excludes parsing, radius inversion, source hashing and
native motion extraction and is not an isolated real-time pipeline benchmark.

## Matched v4 results

`outputs/perspective-sign-recent-v4.json` and its JSONL contain all 1,256 recent
contact sources, 933 reliable direct RAW intervals and 265 recorded resolved
sources. Maximum reconstructed center error is 2.89e-12px and saved-normal to
exact-twin angular difference is 2.11e-8 radians. Thus the archival forward
reconstruction passes its independent saved-center cross-check on this subset.

All 18 native longer-interval jobs in `outputs/direct-native-motion.uF9VAM`
were replayed, including every phase of strides two and four for all three
captures. `outputs/perspective-sign-direct-*-v4.json` preserves the matched
membership and motion hashes. Each stride's phases partition all 1,256 sources;
every native membership source was admitted. These are direct RAW endpoint
matches, not sums of adjacent transforms. Longer gaps still reset the tracker.

| Schedule, all phases | Matched sources | RAW intervals | Model support, 2/4/8px | Preferred steps >30°, 2/4/8px | Baseline steps >30° |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full rate | 1,256 | 933 | 30 / 6 / 0 | 5 / 3 / 3 | 38 |
| Direct stride 2 | 1,256 | 850 | 39 / 22 / 0 | 6 / 3 / 4 | 37 |
| Direct stride 4 | 1,256 | 728 | 30 / 23 / 0 | 3 / 3 / 4 | 34 |

No-motion control preferred steps are 2/3/3 respectively and its model support
is zero. Therefore reduced jumping alone is not evidence that exterior motion
identified the physical sign. Model support vanishes at the 8px allowance; the
experiment is sensitive to uncertain geometry and supplies only limited support
at the narrower settings. Full-rate 2px support is 0/19/11 for the three captures,
and preferred steps are 4/1/0 versus recorded 4/26/8. The noisy first capture
remains unsupported. All absolute sign identifications remain unverified.

The final full standalone suite passes 119 tests with one ignored corpus test,
including all 11 dedicated perspective tests. In the 864-observation off-grid synthetic
sweep, 405 preferred normals are correct and 354 model-supported normals are
correct, with no wrong model-supported outputs. The other 459 preferred normals
are wrong; this is a substantial cold-start/weak-evidence failure, not merely
missing output. The separate narrow-support fixture recovers from the wrong
seed, but does not establish general recovery at corpus uncertainty. First-arm
source update p95 is approximately 11–14µs under shared CPU use, with at most 40
states and 80 transitions. The run claim is
`l18d4a035cd490cca-3e5d9f`; shared coexistence is not exclusive benchmarking.

The final broader v4 replay admitted 1,299 historical semantic sources after
94 late-source rejections; older contact-only archives lacked reconstructable
joint poses. Its historical semantic timestamps do not attest that every ellipse
was independently remeasured. No native exterior intervals are available there.
This is an availability/conditional geometry check, not broad
signed-gaze validation. It has not established improved human-label localization.

The v4 implementation lacked angular derivative history and an independent
head-rotation input. V5 adds these below, but neither iteration justifies live
promotion or a claim that the requested general sign-recovery problem is solved.

Per-phase results are not interchangeable. Stride-two phase 0/1 has 629/627
sources and 26/13 model-supported outputs at 2px. Stride-four phase 0/1/2/3 has
315/315/314/312 sources and 14/6/6/4 such outputs. All phases are included above;
none was selected after inspecting its result.

Reproduction, under the documented `agent-coord` shared CPU/block claim:

```sh
export CARGO_TARGET_DIR=/mnt/bulk_data/buttercup-eye-tracking/target
export CARGO_BUILD_JOBS=2
cargo test --offline --no-default-features --bin buttercup-perspective-sign-replay -j 2
cargo build --offline --profile live --no-default-features --bin buttercup-perspective-sign-replay -j 2
data/target/live/buttercup-perspective-sign-replay outputs/perspective-new.json \
  outputs/student-calibration-diagnosis.1MjGcu
data/target/live/buttercup-perspective-sign-replay outputs/perspective-phase-new.json \
  --source-membership native-motion-index-circle-direct2-phase0-uF9VAM.jsonl \
  --motion-sidecar exterior-motion-circle-direct2-phase0-uF9VAM.jsonl \
  outputs/student-calibration-diagnosis.1MjGcu/1789227222-496864776
```

Repeat the last form for every row in `outputs/direct-native-motion.uF9VAM/jobs.json`.
Outputs use `create_new` and preserve prior receipts. Reports embed tracker/replay
source SHA-256 and input index, prediction, membership and motion hashes. No
captures, reports or compiled artifacts are stored as repository source files.

## V5 common-window results and remaining failure

The v5 fixed-window model materially improves the constructed wrong-seed test,
but still fails the captured calibration target diagnostic. It is an offline
experiment, not an enabled correction. `outputs/perspective-sign-recent-v5-final.json`
and all 18 `outputs/perspective-sign-direct-*-v5-final.json` reports retain matched
source, motion and geometry hashes. Historical v4 receipts remain untouched.

All 1,256 recent sources and 933 reliable direct intervals remain admitted;
the perspective center/normal reconstruction errors are unchanged. Every stride
phase is included and partitions the same sources, not a new independent cohort.

| Schedule | Sources | RAW intervals | Support, 2/4/8px | Preferred steps >30°, 2/4/8px | Recorded steps >30° |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full rate | 1,256 | 933 | 67 / 27 / 3 | 8 / 6 / 3 | 38 |
| Direct stride 2, both phases | 1,256 | 850 | 101 / 77 / 31 | 23 / 17 / 8 | 37 |
| Direct stride 4, all phases | 1,256 | 728 | 0 / 0 / 0 | 37 / 28 / 16 | 34 |

No-motion control support stays zero, with 2/3/3 large steps respectively.
Full-rate 2px support per clip is 6/29/32; large steps are 5/1/2 versus v4's
4/1/0 and recorded 4/26/8. The first clip therefore regresses even against the
recorded step count. Stride-four 2px also regresses against its recorded baseline.
Its true 1.25-second common window cannot retain three supported intervals;
v4's residual-endpoint history could include an interval beginning earlier than
that horizon. Increasing abstention is not success. Stride-two phases 0/1 supply
55/46 supports at 2px, with 14/9 large steps; neither phase was selected afterward.

Only 11 of the 67 full-rate 2px supported sources were baseline-resolved;
50 agree with the baseline branch and 17 disagree. These are not accuracy labels.
The three 8px supports are first-clip ROI2 sequence1267 and second-clip ROI1
sequences2247/2249. All were baseline-resolved and agree with that branch;
their robust-score margins are approximately 0.572/0.243/0.209. The clean latest
ROI2 sequence2857→2858→2860 remains unsupported. The 2/4px arms consistently
prefer one physical history, while the 8px/no-motion arms prefer its opposite.
Smoothness on this episode still does not identify absolute physical sign.

The independent target-window check is
`outputs/sign-target-window-check.vCbvRq/v5-summary.json`, alongside its v4
comparison. On the completed nine-target first clip, all nine examined arms
fail combined affine and shared-monitor-plane acceptance at 0ms and 500ms settle.
The recorded-resolved baseline separately passes the plane check and fails the
affine check; it does not fail both checks individually.
Preferred arms cover nine target windows; conditional-support arms supply no
stable selected-eye target estimates. Passing the replay's execution/source
tests is not a calibration success. Target coordinates were held out of model
development and are not independently verified fixation or signed-3-D labels.

The unchanged 864-observation off-grid sweep now has 816 correct preferences
and 690 correct supported decisions, with no wrong supported decision. Per-trial
assertions verify that the only 48 wrong preferences are the initial wrong seeds;
all 48 trials recover after one source interval, 8/20/100/300ms. Maximum later
angular error is 1.49e-8rad. Every window's actual angular error, twin separation,
source span and score margin is retained in `perspective-sign-v5-sweep.log`.
This is exact constructed geometry with declared uncertainty, not injected noise.

An additional 288 trials / 5,184 observations perturb actual perspective-conic
centers within 30% or 100% of the 2/4/8px image allowance, add bounded exterior
translation error, physically move the effective pivot, and drift its off-grid
depth ratio. Both axes, signs and four cadences are included. Moving-eye trials
select the closest true normal on 1,604/1,728 observations and support 1,204 with
zero wrong supports; maximum supported angular error is 0.002565rad (~0.147°).
There are 28 additional post-seed preference failures at the 8px full-noise level,
all delayed recovery rather than relapse. All 96 moving-eye trials recover,
with first recovery ranging from 8ms to 1.8s; maximum post-seed angular error
while waiting is 1.627rad. Per-trial recovery times and errors are retained.
All 1,728 head-translation-only and 1,728 unknown-transport observations have
zero support. Their wrong initial preferences may remain wrong: abstention is
not recovered orientation. These deterministic perturbations do not cover
arbitrary lid-induced conic bias, torsion, or real head rigidity failures.

A separate 288-trial / 5,184-observation shape test perturbs center, both
semiaxes and orientation, scaling orientation uncertainty by ellipse anisotropy.
Axes stay positive and correctly ordered. An operator-norm boundary-displacement
upper bound stays within the declared 2/4/8px. The true projected-pivot error
nevertheless reaches **41.57px**, with maximum amplification **9.664 times**
the declared boundary allowance. Thus boundary error cannot simply be reused as
the pivot endpoint allowance in this model. This is a substantial uncertainty
propagation failure, not evidence that the larger pivot deviations are impossible.

Shape-perturbed moving-eye trials select the closest true normal on 1,517/1,728
observations and support 763, with no supported choice of the farther twin.
All 96 trials first recover after one interval, but there are 115 later errors.
Even a supported closest twin has up to **0.20953rad (12.0°)** actual normal error;
the worst post-seed preference error is 1.67351rad. Both shape-perturbed negative
controls still have zero support. A correct branch index is therefore insufficient
to certify gaze precision. The exact receipt is
`outputs/perspective-sign-v5-shape-tests.log`; the perturbations and zero-false-
branch-support assertions were preserved without shrinking jitter or tuning gates.

New tests also compare the DP against exhaustive enumeration, verify irregular
source-midpoint derivatives, head-reference resets and reflected axial-vector
conversion, and demonstrate actual output changes when derivative order is
ablated. A true perspective x/y, both-sign near-frontal crossing continues through
the meridian with motion order while its ablation bounces; both stay unsupported
without transport. The original approximately 50° saccade remains admitted.

The broader 28-capture replay still admits 1,299 historical semantic sources,
rejects 94 late sources and has no usable native exterior intervals. It does not
rerun old human labels or remedy their documented localization failures.
All sign arms preserve the source ellipse, so localization and relative
independent-scale SN-FEIDA changes remain exactly zero.
SN-FEIDA direct-scale absolute-log-step p95 is 0.11274/0.02629/0.03192 over
160/354/419 independently supported pairs in the three recent clips.
Typical full-rate update p95 is 79–84µs under shared CPU use, excluding
parsing/reconstruction/RAW hashing.
The bound is 40 cells and 860 edges; no exclusive latency benchmark is claimed.

Final validation is `outputs/perspective-sign-v5-complete-tests.log`:
126 tests pass and one unrelated corpus test remains ignored. The source-tree
audit passes. After all source/test changes, all 20 reports were regenerated as
`*-v5-final.json` / JSONL: recent, 18 direct phases, and broader historical.
`outputs/perspective-sign-v5-final-replays.log` verifies each report's embedded
tracker/replay SHA-256 against the checkout, complete native membership admission,
and byte-identical per-source JSONL to the earlier v5 reports used by the target
diagnostic. Test success does not erase the target-fit, uncertainty and cadence
failures above. No live correction, training, calibration or camera action occurred.
The next bounded experiment should propagate conic boundary uncertainty through
both perspective pose/pivot solutions before comparing motion histories, then
repeat the same synthetic perturbations, fixed memberships and held-out target
diagnostic. More support or smoother normals alone is not an acceptance criterion.

## V6 conic uncertainty: bounded experiment, failed recovery

V6 replaces the invalid boundary-to-pivot allowance shortcut with a finite
branchwise propagation experiment. It **fails the recovery objective** and is
not suitable for live promotion. Exact current normals remain unchanged nominal
twins; nuisance poses only enter projected-pivot scoring. First and final recent
replays have identical source keys, normals, preferences, support and scores.

Ellipse shape is represented by the symmetric positive matrix
`S = t I + [[u,v],[v,-u]]`, avoiding a major-axis angle gauge at a circle.
`|delta t| + hypot(delta u,delta v)` is its operator norm, so adding center
displacement gives an image-boundary Hausdorff upper bound. The main candidate
predeclares half the 2/4/8px budget as a shared additive sensor-coordinate shape
bias and half as per-frame innovation. Seven shared states are zero and signed
isotropic, diagonal-traceless and off-diagonal perturbations. The same state is
profiled across the entire source window, together with the five depth ratios.
This is a conditional sensor-space fitting-bias model; it is not an anatomical
invariance under changing image scale or rotation. The seven-point stencil is
not covariant under arbitrary image-coordinate rotations.

For each shared state, 26 operator-normalized Cartesian shape directions,
16 traceless angular directions, an interior isotropization probe and eight
center directions are decomposed through both exact perspective solutions.
Pair association minimizes total angular distance over both assignments; both
different centers and all five projected pivot depths are retained. Separate
shape and center sampled maxima are summed, explicitly relaxing their joint
budget rather than making a statistical independence claim. These are finite
sample envelopes, **not certified continuous enclosures**. There is no post-test
inflation factor. Invalid decompositions are counted, and prevent support.

A separate full-budget envelope reports normal angular spread and pivot spread;
overlapping angular cones prevent conditional support but leave provisional
preferences visible. A fifth corpus arm uses the full 4px independent-innovation
budget with no shared bias. Its contrast with the mixed 4px arm is a sensitivity
check, not a fitted estimate of the actual bias/noise split. JSON records the
residual and joint argmin shared states; residual argmins include depth ratios.
The legacy `pivot_bounds_px` diagnostic is explicitly marked as zero shared bias
at depth ratio 1.5, not the profiled winning bound. `pose_allowances_px` is the
legacy nominal boundary budget, not a propagated endpoint bound.

The DP retains at most 280 suffix/nuisance cells and 6,020 edges. An eight-source
exhaustive enumeration now compares both objectives across all 35 nuisance
states with distinct shared-shape profiles, including a missing transport.
Kinematics still uses nominal normals, while the pivot residual profiles latent
shape poses. Thus the joint provisional ranking is not one fully consistent
latent-conic trajectory objective. Independent residual support does not use
that kinematic approximation. No motion or uncertainty thresholds were tuned
after examining v6 results.

There is a structural provisional-readout weakness: the normalized whole-window
contribution is below 0.01, while the continuity term is `0.03 * angle` and can
exceed that for widely separated twins. When propagated uncertainty reduces the
residual margin below the unchanged 0.15 gate, a better independent profile may
therefore fail to overcome the wrongly emitted branch. This helps explain the
observed seed locking; lowering the support gate would not establish physical
identification. Consistent latent-shape normals and trajectory-level readout
remain unimplemented, rather than being claimed fixed by this experiment.

Independent envelope validation probes off-stencil directions and interior
radii, both physical branches, all five depths, both axes/signs, centered and
off-axis geometry, and near-frontal tilts. All 46,080 pivot comparisons and
9,216 angular comparisons fit their finite envelopes, with zero measured excess.
The original noisy controls also cover every closest-true normal angular error
and every conditionally supported angular error. This is empirical coverage of
specified perturbations, not calibrated precision or coverage of arbitrary
occlusion bias, optics or intrinsics error.

The original tests retain their geometry, perturbation magnitudes and source
cadences. The noiseless sweep now collects every trial before applying its
unchanged recovery acceptance, so the failed assertion does not hide later data.

| Constructed moving-eye sources | V5 correct preferences / supported | V6 correct preferences / supported | V6 remaining failures |
| --- | ---: | ---: | --- |
| Noiseless, 864 | 816 / 690 | 544 / 141 | 272 post-seed failures; 16 of 48 trials never recover |
| Center noise, 1,728 | 1,604 / 1,204 | 1,156 / 276 | 28 of 96 trials never recover |
| Center and shape noise, 1,728 | 1,517 / 763 | 1,247 / 269 | 10 of 96 never recover; 147 post-recovery errors |

Every supported choice remains the closest twin in these constructed cases,
but this alone does not establish accuracy. V6 maximum supported normal error
is 0.000609rad for center noise and 0.053609rad (3.07 degrees) for shape noise.
Shape-noise first recovery can take 2.7 seconds in trials that recover at all.
Both 1,728-source head-translation-only and unknown-transport negative controls
have zero support for both noise families. Their wrong preferences can persist.
The original near-frontal continuation, legitimate 50-degree saccade, freshness,
head-reference and axial-basis tests pass; they do not repair the recovery failure.

All sixteen never-recovered noiseless trials use the 8px allowance, spanning
both axes, signs and all four source cadences. The other thirty-two recover
after one source interval; these are persistent seed locks, not delayed recovery.

A separate off-stencil shape-bias sequence comparison uses actual shared budget
fractions zero, one half and one, each with 48 trials / 864 sources. Mixed versus
fully independent inference gives correct preferences 413/429, 498/445 and
477/421 respectively, and supports 35/24, 98/50 and 172/71. All supported angular
errors are inside the reported full-budget sample envelope, with zero wrong
supported branches. Scores differ on 2,448 sources, demonstrating that shared
profiling is active. Neither formulation supplies general recovery or a precise
normal estimate merely by choosing a shared-state argmin.

All 20 final matched reports are `outputs/perspective-sign-*-v6-final.json`
and corresponding JSONL: recent, the same 28 broader clips, and all 18 direct
native phase jobs. The phases partition the same 1,256 sources per stride;
they are not independent cohorts. All source memberships are admitted and all
report tracker/replay hashes match the final Rust sources.

| Schedule | Sources / RAW intervals | Mixed support 2/4/8px | Mixed steps >30 degrees 2/4/8px | Baseline steps |
| --- | ---: | ---: | ---: | ---: |
| Full rate | 1,256 / 933 | 13 / 1 / 0 | 3 / 3 / 2 | 38 |
| Direct stride 2, both phases | 1,256 / 850 | 40 / 4 / 0 | 9 / 4 / 3 | 37 |
| Direct stride 4, all phases | 1,256 / 728 | 0 / 0 / 0 | 24 / 5 / 4 | 34 |

Full-independent 4px support is zero for all schedules, with 2/3/4 large steps;
the no-transport arm also has zero support, with 2/3/3 steps. Mixed full-rate
support per clip is 3/7/3 at 2px and 1/0/0 at 4px. This is a coverage regression
against v5's 67/27/3 full-rate and 101/77/31 stride-two supports, regardless of
the smaller jump counts. Angular cones overlap on 6/185/1,248 recent sources at
2/4/8px; no uncertainty samples fail decomposition on the recent set.

Of the thirteen 2px supports, eight were baseline-resolved; eleven agree with
the baseline and two disagree. The five newly supported sources have no signed
truth. Their supported angular radii span 0.10965–0.19254rad (6.3–11.0 degrees),
with median 0.11909 and p95 0.19130rad. These are wide conditional supports,
not thirteen demonstrations of useful gaze precision. The clean latest reversal
episode remains unsupported. Broad replay admits 1,299 semantic sources, rejects
94 late sources and has no native exterior transport. Its historical label
limitations and the unchanged SN-FEIDA/localization diagnostics remain as above.

There are at most 408 perturbed conic decompositions plus eight nominal/shared
decompositions per source/arm. Recent uncertainty-adapter p95 is 0.239–0.248ms;
first-arm DP p95 is 0.622–0.824ms under shared CPU use. These exclude parsing,
saved-radius inversion and RAW hashing; summed percentile values are not an
end-to-end percentile. The shared CPU/block lease was
`l18d4a1c3d4e3afab-3ec4cd`, priority 10, low sensitivity, with coexistence rather
than exclusive timing. No GPU, learned inference or live action was used.

`outputs/perspective-sign-v6-complete-tests.log` records 129 passed, one failed
(the retained noiseless recovery acceptance), and two unrelated ignored tests.
The first failed receipt remains `perspective-sign-v6-first-tests.log`.
Source-tree audit and `git diff --check` pass. V6 establishes a tested finite
uncertainty propagation approximation while failing the intended general sign
recovery task; abstention and smoother output are not a replacement outcome.
`outputs/perspective-sign-v6-final-validation.log` verifies all twenty final
source hashes, unchanged v5 input/motion/membership/scale measurements, and
first/final recent output equivalence. Root's independent final native target
check is `outputs/sign-target-window-v6-final.rksC0X/summary.json`, with eleven
source-checked native test executions. For the selected ROI2 (eye index one),
every arm fails affine and combined/shared acceptance at both 0ms and 500ms
settle; preferred arms zero through four each cover nine target windows, while
supported arms zero, one, two and four have zero stable target estimates.
The recorded-resolved baseline alone passes the separate plane check but fails
affine acceptance. Every selected-eye source-prefix shared check also fails.
These are surface-axis feature diagnostics, not verified fixation/gaze labels;
passing execution/source checks does not make the calibration fit successful.

## V7 consistent trajectory experiment (validation incomplete)

This single bounded candidate retains v6's uncertainty samples, budget split,
conditional support gates and residual-only recurrence. Each shared shape state
now stores both associated conic normals alongside its projected pivots. Its
source-timed velocity, acceleration and jerk use those same normals throughout
the path, including the existing independent head-rotation transport convention.
The five depth ratios share the same normal for a given shape state.

The existing `0.03 * angle` regularizer now applies to every latent trajectory
edge, after any known head transport. It is an arc-length preference, not a rate
bound. Provisional readout takes the full joint-window optimum; the former
saturating score and separate emitted-history motion/continuity penalty are
removed. Only numerical ties within `1e-10` use previous nominal-output proximity.
The existing meaningful residual winner still overrides provisional ranking.
Residual support must therefore match v6 exactly on matched inputs.

Output remains the unchanged current nominal twin. JSON separately records both
joint argmin states, their depth ratios and current latent normals, plus the
selected latent normal. Fitted derivatives describe that latent path; emitted
derivatives describe nominal outputs and provide no objective feedback. Revising
the preferred branch can change output without proving that the physical eye
jumped; all emitted jumps must nevertheless count in the regression comparison.

The first test receipt, `outputs/perspective-sign-v7-first-tests.log`, restores
all 48 noiseless wrong-seed trials to one-interval recovery: 816/864 preferences
correct, zero post-seed errors, and the unchanged 141 supported outputs. Center
noise improves to 1,632/1,728 correct preferences with the unchanged 276 supports
and zero wrong supports. All 96 center-noise trials recover with no later errors.
Shape noise gives 1,517/1,728 correct preferences, 269 supports, zero wrong
supports, all 96 trials recovering, and 115 later errors. Maximum supported
normal errors remain 0.000609rad and 0.053609rad respectively. Head-translation-only
and unknown-transport controls retain zero support in both noise families.
However, the retained no-transport near-frontal meridian
test fails at axis x, negative sign, fourth source. Recovery gains do not excuse
that regression. No thresholds, perturbations or assertions were weakened.
The first suite finishes with 129 passes, this one failure and two unrelated
ignored tests. A subsequently added analytic test isolates shared-state angular
derivatives from stationary nominal output; it awaits the next authorized test
run. The existing exhaustive DP comparison already passes with distinct latent
normals in all seven states and the same missing-transport edge.

Corpus replay and further builds are paused for the user's live recording.
The optimized offline binary built successfully before that pause; no v7 corpus
or held-out target results exist yet. This is incomplete validation, not a sign
fix or permission to promote live behavior.
