# Geometry module boundaries and type inventory

This remains an incremental geometry architecture, not a calibrated
probabilistic eye model. A sparse shared-target conic solver and source-keyed
live adapter now exist; their corpus validation is in progress. See
[joint conic solving](joint-conic-solver.md) for the objective, assumptions,
live routing, evaluated scope and known failures. The default single-ROI
route retains the established monocular surface tracker.

## Responsibilities

| Module | Existing functionality now owned here | Still missing |
| --- | --- | --- |
| `roi_evidence` | RAW model-frame record; image-motion records; source-exposure-aligned whole-ROI motion timeline | Uniform evidence packets from every detector, including correlated arc alternatives |
| `outline_conic_segments` | Ordered contours, flat-tire exclusion, tangent support; owned source-keyed retained SAM arcs and bounded RAW gradient alternatives | Unfitted multi-query arc admission and iterative resegmentation |
| `conic_solver` | Legacy RANSAC/refit plus exact projected-circle mixed-boundary/multi-ROI shared-target hypotheses and bounded robust refinement | Calibrated uncertainty, stronger boundary-selection/observability accounting |
| `eye_scene_model` | Relative gaze, surface/sign and effective-pivot histories, convex-contact proof, coarse scale; shared limbus/pupil state; coarse binocular metric support | Calibrated metric camera/eye pose, learned mobile-pivot feedback and a joint anatomical motion model |
| `binocular_coordinator` | Same-source-clock policy and bounded per-ROI exact-exposure pairing | Per-eye settling histories, measured rolling-row clock bounds, learned vergence/IPD factors |
| `gaze_target_solver` | Monocular display mapping/calibration plus joint shared-fixation inference and asynchronous source-keyed orchestration | Validated optical-to-visual-axis calibration and a metric transform to the legacy display frame |
| `geometry` | Shared ellipse shape, image-axis helpers and small numerical/vector routines | Further coordinate-safe primitives as callers migrate |

The live SAM route now calls the extracted outline and conic modules, then
the scene/surface and target geometry. SAM still owns inference, model/native
coordinate conversion, semantic candidate selection and RAW publication gates.
The native RAW detector and motion extractor remain observation backends.
The viewer owns mode authority, calibration admission/state, held overlays,
rendering and camera-service interaction.

The second extraction splits the scene model's shared radius/center reasoning
into four child modules, without adding new high-level solver layers:

| Scene child | Inputs / responsibilities | Deliberately not owned here |
| --- | --- | --- |
| `limbus_scale` | Frozen candidate-independent apparent-radius support, coarse/fine scale transport, strong-observation log-radius history and late-result rejection | Pixel extraction, semantic anatomy authority, metric eye radius |
| `pupil_projection` | Shared limbus provenance, typed radius rectification and native-ROI ↔ canonical-limbus point conversion | Center tracking, optical confidence, display guides |
| `pupil_center` | Borrowed whole-ROI/pupil motion evidence, sensor/limbus transport, fresh-ring admission, bounded relocation and pursuit | Image buffers, rendered trails, acquisition-mode selection |
| `pupil_size` | Hard radius support, soft temporal preferences, optical/detail conditioning, admitted pupil/limbus ratio history and radius continuity | RAW arc sampling, final frame-admission orchestration, rendering |

The center and size trackers share projection geometry rather than depending on
each other's state. Mutable histories remain private. The RAW motion backend
adapts its overlay to `PupilCenterMotionEvidence` by borrowing four existing
records; the center tracker cannot read image buffers or reflection-layer tracks.
The native detector retains compatibility reexports for the moved limbus types.
These legacy trackers still use caller-supplied host `Instant`; this port does
not pretend that they already use the new source-clock contracts.

The **joint target solver is separate from the binocular coordinator**.
Coordination supplies temporal/vergence factors, not a winning eye or final
cursor. A missing or defocused eye must remain optional. A geometrically
useful weak arc must be able to disambiguate a strong but ambiguous single-eye
fit; an incompatible second eye must not degrade a good one by forced averaging.

The sparse request contracts borrow native-ROI arcs and supported conics rather
than cloning frames. They preserve boundary kind, evidence-group correlation,
ROI identity and exposure. Joint conic requests have explicit hypothesis and
refinement budgets, enforced by the joint optimizer. The migrated legacy
fitter still uses its original bounded sampling/refit loops. The viewer adapter
is `joint_gaze_live`: no window state or calibrated screen target enters the
joint segment objective.

## Coordinate and uncertainty rules

### Required two-way arc / eye-scene constraint

Outer/inner limbus bands and pupillary-boundary arcs must jointly support a
plausible nested iris/pupil surface around an uncertain pupil-axis region.
This is evaluated through the shared 3D or rectified iris-plane model, not by
forcing raw-image ellipse normals to intersect the image pupil center. Allow
bounded decentration and model mismatch rather than requiring exact concentric
circles. The pupil center, iris-plane axis and effective rotation pivot remain
distinct latent variables.

- Forward: `eye_scene_model` supplies uncertain camera/scale, surface pose,
  pupil-axis and effective-pivot predictions. `conic_solver` uses them to rank
  or cull arc combinations; `outline_conic_segments` uses projected boundaries
  and their localization bands to refine sparse segmentation.
- Feedback: joint arc residuals constrain the same scene variables. Across
  source-timed frames, surface motion can adjust the approximate pivot, which
  changes the next predicted conic family. A single projected ellipse must not
  falsely make pivot depth or rotation-center location fully observable.
- Refinement: retain bounded alternative hypotheses and relinearize the same
  evidence factors for a limited number of iterations. Feedback is not a new
  observation: preserve exposure keys and correlated evidence groups so a fit
  cannot become more confident merely by being fed around the loop. External
  hard geometry/convexity limits stay fixed during this refinement; the
  approximate pivot is a soft constraint, not an immutable anatomical hinge.

The joint solver implements the same-frame coupled nested-circle model and
optional independent mobile-pivot support. Persistent anatomical learning,
iterative resegmentation and calibrated uncertainty remain incomplete. The
current live prior does not invent an independent pivot from its own candidate.

### Executable constraint examples and conditional fidelity

`conic_solver/constraint_tests.rs` supplies small numerical examples for this
contract, not a replacement live solver. An outer projected circle and coplanar
concentric inner circles leave opposite camera-facing tilt signs equally valid.
A partial pupil arc breaks that tie only under explicit relative-depth and
bounded-decentration assumptions; a free decentration restores the ambiguity.
Both mirrored examples are tested. A soft, explicitly synchronized/settled
cross-eye factor can disambiguate weak evidence, while excellent near-frontal
support contributes negligible *sign* information. Normalized profile-likelihood
weights in these examples are not calibrated probabilities.

`conic_solver::fidelity` now estimates **conditional translation fidelity** from
sparse conic normals, optical detail, one information budget per independent arc
group, and a declared model-error floor. It returns a 2×2 covariance proxy in
native px² and a worst-axis sigma in pixels. This conditions on the conic shapes
and boundary associations; it is not free-ellipse, 3D-target, or sign uncertainty.
Dense contour resampling and repeated fitted conics do not increase information;
unresolved correlated alternatives and singular tangent support abstain. Blur
increases a configurable localization-noise heuristic without discarding all
defocused evidence. These engineering sigmas need empirical corpus calibration.
The existing live publication and mouse calibration do not consume this helper.

`roi_evidence::timing` provides immutable sensor-read and ROI-buffer provenance
for future observation adapters. Derived images share their buffer timing;
different reads propagate signed lower/upper relative-time bounds only when
explicitly attested. The same sensor read cancels common reference-time error,
but different rolling-shutter rows still need an exposure-offset model. Unknown
clock epochs, accuracy or row phase remain unknown. The current TCP packet has
a source timestamp, **not these hardware uncertainty bounds**; no live producer
has been silently upgraded to exact synchronization. Tests show how source-time
bands improve motion-hypothesis rejection and how wider bands reduce certainty.

For real-data development, use the definition and matched corpus workflow in
[Scale-normalized frontal-equivalent iris disk area](flat-tire-area-and-motion.md).
Area consistency, conditional localization fidelity and anatomical validity
are complementary diagnostics, never substitutes for one another.
For crop-motion transport, the enabled sparse overlap sampler and the disabled
SAM identity-carry experiment, see [ROI reframe continuity](roi-reframe-continuity.md).

### Shared conventions

- Image coordinates are pixel centers, +x right and +y down. Model, native ROI,
  full sensor and screen coordinates are different spaces.
- A bare `Ellipse` has no source clock, anatomical identity or confidence.
  Its radii are semi-axes, not diameters. Its angle is radians along the first
  axis. Public construction does not itself establish canonical axis order.
- The legacy SAM fit domain remains 384×256. Its pixel limits are named
  `LEGACY_FIT_WIDTH/HEIGHT`, not universal sensor or biological constraints.
  Native/model conversion remains upstream and preserves pixel-center alignment.
- `RelativeGazeVector` is unitless: +x sensor-right, +y sensor-down, **+z toward
  the camera**. Do not silently reinterpret it using the common +z-away camera
  convention. Current display geometry is eye-relative **inches**.
- Contact sphere/pivot depth is currently in image-scale units relative to the
  limbus slice. It is not metric camera distance. The joint target contract
  names millimeters explicitly, preserves its metric-prior provenance, and
  leaves uncalibrated covariance absent.
- Effective rotation centers are movable approximations, not rigid anatomical
  hinges. Muscle-driven translation and model mismatch need nuisance/uncertainty
  terms. Motion history supplies defeasible support; the visible convex surface
  is a separate hard constraint.
- A presentation hold is not a new observation. Repeated SAM results retain the
  source exposure; clocks must never become inference-completion timestamps.
- A score, heuristic sigma, support interval and calibrated probability are
  different quantities. Do not rename one to another to make an API uniform.

## Single-eye temporal sign correction (September 7)

`eye_scene_model::sign_motion` compares the two camera-facing tilt hypotheses
using only one eye's source-timed geometry and independently extracted RAW
image motion. Neither the other eye nor monitor calibration is an input.
The earlier one-interval temporal path could seed a sign, but deliberately
could not revise an established sign. Its later kinematic check compared an
alternative against already-signed history, potentially penalizing correction
of a wrong initial choice as an abrupt physical saccade.

The bounded window keeps up to 12 interval scores over 1.25 seconds. Each
branch has its own implied-pivot trajectory. Each previous pivot is transported
into the current sensor frame using source-aligned whole-ROI similarity,
including its actual center, scale and rotation, and compared to the current
pivot of that SAME branch. At least four fresh, discriminating intervals must
support the correction, with limited contrary evidence and a sustained bounded
cost margin. Re-scoring one old fit or a neutral current frame cannot vote.
This intentionally abstains when movement is too small relative to fit and
transport error; it does not claim every non-frontal motion is identifiable.

The pivot uses the existing projected contact-depth approximation, not a
fixed, measured anatomical rotation center. Its error allowance includes a
0.5-pixel floor, 1.5% of apparent limbus radius, and twice the current-interval
RAW transport residual. These are defeasible engineering margins, not calibrated confidence
intervals or measured physiological limits. Missing/untrusted transport breaks
the chain; it never implies a stationary head. Near-frontal projected normals
below 0.08, invalid geometry, stale intervals, and scale-family resets discard
the relevant motion evidence. Repeated/out-of-order source results cannot vote.

A supported correction changes the persistent branch and advances the sign
epoch once, clears incompatible signed velocity/smoothing state, and publishes
the current source direction immediately. Existing calibration-basis guards
therefore invalidate an affine trained on the old sign. The physical monitor
pose is not adjusted to conceal a sign correction. The convex/camera-facing
proof remains mandatory; this change does not manufacture independently
reflected X-only or Y-only normals.

Validation uses mirrored upward/downward, horizontal and oblique synthetic
sweeps, approaching/receding tilt, initially correct and incorrectly resolved
seeds, several scales, head translation, ROI nudges, equivalent ellipse angles,
small pivot drift, weak/uncertain motion, jitter, single-frame
outliers, missing/uncertain motion, and stale/duplicate clocks. The offline
`--offline-contact-sign-eval OUTPUT.json CAPTURE_DIR subject-right|subject-left
[START] [COUNT]` command runs matched old/new policies on frozen recorded SAM
ellipse geometry while recomputing RAW transport. Its baseline disables only
the new window; it is not a byte-identical reproduction of live inference
scheduling. All captures/reports remain below `outputs`.

This is sign-policy validation, not a new limbus fitter or a complete 3D eye
model. Sign alternatives preserve the fitted ellipse and frontal-equivalent
disk area. SN-FEIDA therefore cannot determine which sign is right. Human sign
labels and independently measured scale are absent in the current recording;
cross-eye disagreement may be measured afterward as a diagnostic but is never
used to choose either eye's sign or claimed to be ground-truth gaze accuracy.

A wider, median-centered whole-trajectory scatter alternative was tested and
rejected: in the same recording it increased strongly opposite vertical pairs
from 494/1,220 to 518/1,220 and made 41 motion corrections, versus one correction
and 408/1,220 opposite pairs for the first conservative candidate. These are
matched offline rollouts seeded from the recorded sign, not the original live
publication counts. Both alternatives preserved ellipse-area and fit coverage.
The rejected source and reports are retained under
`outputs/sign-audit-20260907`; its permissive trajectory scoring is not enabled.
This experiment does not establish which eye is correct, and unresolved
disagreements remain.

## Existing custom types

Related types are grouped below, including types deliberately left with their
current backends. This inventory is of geometry/evidence/state contracts, not
every UI enum or private numerical scratch record.

### Shapes, arcs and observations

| Types | Space and meaning | Normalization decision |
| --- | --- | --- |
| `geometry::Ellipse` (formerly `sam31_outer::Ellipse`) | Caller-declared image pixels, semi-axes, radians | Canonical shared shape; old SAM path reexports the exact same type |
| `ContourFitEvidence` (formerly `OuterMaskFitReview`) | Ellipse plus retained/excluded contour samples and component area; model pixels during fitting, native ROI pixels after conversion | Owned by outline module; old name remains a compatibility reexport |
| `ConicArcConstraints`, `EllipseSupportSummary`, `RobustFit` | Tangents, phase coverage, local conditioning and pixel residuals | Solver internals; not anatomical observations or calibrated covariance |
| `BoundaryEdge`, `FlatTireRun`, `FlatTireSide` | Ordered raster topology and excluded run indices | Outline internals; retain contour order |
| `OuterIrisBoundary`, `OuterIrisPoint` | Native-detector ROI shape plus direct, occluded and projected samples | Keep evidence-bearing record distinct from a bare ellipse |
| `InnerIrisBoundary`, `InnerIrisPoint`, `InnerIrisRadialCandidate` | Pupil-margin shape, ordered samples, alternative prior-free radial peaks and RAW scores | Preserve pupil/limbus distinction and alternatives |
| `BorderFocus`, `BorderPoint` | Coarse native eye-basin geometry with focus/contrast and pupil hints | A hint is not an admitted limbus |
| `RoiTruncatedLimbusObservation` | ROI-censored conic, visible/support fractions, clipped-side flags and desired sensor reframe | Outside-ROI geometry remains prediction |
| `EyelidNautilusScene`, `EyelidObservationStatus` | Current-frame lid/fold/lash geometry and observed/clipped/unknown states | Missing evidence is not equivalent to an observed occluder |
| `LimbusPerimeterDriveSample`, `LimbusPerimeterStrip`, `LimbusLateralOrderEvidence` | Native sampled boundary profiles and lateral material ordering | Keep with RAW extraction until a lossless arc adapter exists |
| `IrisEllipseSeed`, `EdgeEllipse` | RAW-motion proposal shapes; internal angle normalized to [0, π) | Do not blindly alias to SAM's differently normalized fit |
| `CannyEllipseProposal`, `FeatureClusterIrisHypothesis`, `EllipseEvidence` | ROI conic proposals with edge support, conditioning or motion-layer identity | Geometry alone must not transfer their publication authority |
| `RadialLimbusProbe`, `NestedEyeBoundaryPair` | Normal-flow evidence and ordered pupil/limbus interpretation | Normal flow is not tangential feature identity |
| `ProjectedIrisGeometry` | Absolute sensor conic, heuristic confidence and anatomy authorization | Retain evidence/authority wrapper; now in the scene-owned kinematics module |
| `EyeCenter`, `Detection` in `native_mediapipe` | Normalized coarse sensor landmarks and apparent radius | Convert explicitly at the MediaPipe adapter; not metric pose |

### Size, scale and feasibility

| Types | Units and interpretation | Owner / remaining work |
| --- | --- | --- |
| `CentralCameraLimbusProjectionEnvelope`, `LimbusProjectionAssessment` | Focal bounds in pixels, angles in radians, anisotropy/axis ratios | Moved to conic solver with RAW-detector compatibility reexports; provisional policy, not calibrated anatomy |
| `OuterContourScaleContext` | Frozen fronto-parallel radius support in the fitting frame's pixels | Conic solver; excluded contour points cannot widen their own interval |
| `FrontoParallelLimbusRadiusPrior`, `FrontoParallelLimbusRadiusPriorSource` | Apparent fronto-parallel pixel-radius support plus provenance | `eye_scene_model::limbus_scale`, with RAW-detector reexports; the legacy “Prior” name does not make hard support a confidence interval |
| `FrontoParallelLimbusScalePrediction`, `FrontoParallelLimbusRadiusTracker` | Dimensionless image-scale transport/uncertainty and robust log-radius state | `eye_scene_model::limbus_scale`; candidate-independent frame freezing, cold-start consensus and late-result rejection preserved |
| `InnerIrisRadiusPrior` | Soft preferred area-equivalent pupil radius in pixels | Must remain defeasible by strong current RAW evidence |
| `InnerIrisRadiusEnvelope` | Hard current-frame area-equivalent radius search bounds | Not interchangeable with the soft prior above |
| `InnerIrisEvidenceCondition`, `PupilEvidenceCondition`, `OpticalFocusClass` | Optical/detail reliability and resolution, not physical pupil size | RAW detector / `eye_scene_model::pupil_size`; explicit adapter keeps optical reliability separate from geometric score |
| `ProjectedAreaEquivalentRadiusPx`, `AffineForeshortening`, `FrontoParallelCircleRadiusPx`, `PhysicalRadiusRatio` | Distinct private-constructor radius/ratio units | Moved intact to `eye_scene_model::pupil_radius_units`; do not replace with bare f64 |
| `CentimeterScaleEstimate`, `CoarseEyeScaleSeed` | Sensor pixels per centimeter and a movement-adjusted heuristic interval; seed is image pixels | Scene model with a thin MediaPipe adapter; **presentation-only**, not wired to SAM search bounds |
| `RadiusKinematicSupport`, `RadiusRateLimiter` | Adjacent-publication radius/ratio trajectory bounds in the caller's scalar space | `eye_scene_model::pupil_size`; candidates and admitted updates have distinct mutation paths; not yet a universally unit-typed scalar limiter |
| `PupilProjectionSource`, `PupilProjectionReference` | Provenance and weak-perspective pupil/limbus rectification | `eye_scene_model::pupil_projection`; center and size share the same typed radius conversions |
| `FrontoParallelScaleBucket`, `ScaleConditionedFocusReference`, `PupilEvidenceConditionTracker` | Resolution buckets and focus-conditioned reliability history | `eye_scene_model::pupil_size`; bucket construction is validated and its index is read-only; not anatomical pupil-size estimates |
| `PupilSizeGeometry`, `PupilSizeTracker`, `PupilSizeObservationAdmission` | Cached projection/ratio support, bounded histories and admission-result metadata | `eye_scene_model::pupil_size`; cached geometry survives a missing rough center; final RAW frame-admission orchestration remains in the viewer |
| `PupilSizeSupport` (formerly `PupilSizeReticle`) | Frozen per-frame solver support: hard limits, soft preferences, projection provenance and temporal metadata | Renamed and moved to `eye_scene_model::pupil_size`; this was never merely a display record or an independent observation |
| `PupilDiameterArcEvidence` | Current RAW edge/radius agreement and opposing-arc diagnostics | Still alongside RAW admission orchestration in the viewer; next observation-adapter extraction |
| `IrisRadiusReticle`, `PupilRadiusReticle`, `PupilSizeRuntimeStatus`, `SizeReticleMode` | Display products, status and operator guides | Remain with presentation; recentered/rendered supports never train the solver |

### Motion, eye state and targets

| Types | Coordinate/time contract | Decision |
| --- | --- | --- |
| `RawModelFrame` | ROI identity, source sequence/ns, sensor origin, native RAW dimensions, focus and geometry hints | Record moved to ROI evidence; TCP codec and validation remain in protocol module with unchanged bytes |
| `SimilarityMotion`, `MotionLayerStatus`, `NativeGlobalSimilarityEvidence` | Sensor-pixel motion, per-layer diagnostics and admitted vs diagnostic whole-ROI transport | Moved to ROI evidence; RAW matcher retains extraction; `rotation` is the legacy linearized similarity coefficient |
| `GlobalSimilarityTimeline`, `TimedGlobalSimilarity` | Bounded contiguous source-exposure motion composition, one instance per ROI/session | Moved from viewer to ROI evidence |
| `KinematicDerivatives`, `CoupledMotionStatus` | Source-clock v/a/j in px/s, radians/s and higher powers; pupil relative to tissue | Scene-owned kinematics, **not left/right-eye coupling** |
| `RotationCenterStatus` | Sensor-space algebraic fixed-point estimate and heuristic uncertainty | Not by itself an anatomical assertion |
| `ProjectedGlobePoseStatus`, `GlobeMotionRegime` | Latent effective pivot, translation nuisance, pose corroboration and short weak-frame transport | Scene-owned kinematics; heuristic confidence is not a posterior probability |
| `RingPoseObservation`, `ProjectedVisionPose`, `RotationCenterHistory` | Legacy image-space pose history, still using host Instant | Moved to scene; not silently upgraded to source-clock or metric state |
| `RelativeGazeVector` | Unit camera-facing direction, independent of crop/display | Shared scene-owned contract for existing contact/laser/target math |
| `SurfaceGazeSample`, `SurfaceGazeTracker`, `ContactSignHypothesis`, `GazeKinematicFrame`, `GazeSignCorrection` | Sensor-space surface, source timestamps when available, sign lineage and bounded antipodal motion history | Moved to scene; retain unresolved state and prohibit independent x/y sign inventions |
| `RotationRenderGeometry`, `ProvisionalSurfacePose`, `CameraFacingConvexContact` | Image-scale sphere/pivot geometry and validated convex visible surface | Scene geometry, not UI drawing; invalid output still follows the fatal invariant policy |
| `LockedRayOrigin`, `PresentationPivotContact`, `PresentationLaserLease`, `VirtualContactPose` | Viewer-authorized or held geometry with epoch/source binding | Deliberately remain with presentation/authority; never feed held data back as fresh evidence |
| `PupilCenterPrediction`, `PupilCenterStateTracker`, `PupilCenterTransportSource`, `PupilCenterTrackDiagnostics` | Native-ROI predictions, persistent absolute-sensor/canonical-limbus state, motion regime and provenance | `eye_scene_model::pupil_center`; host-Instant pursuit/relocation policy preserved; holds do not refresh observation age |
| `PupilCenterMotionEvidence` | Borrowed general motion, general/pupil layer support and coupled kinematics for one aligned ROI/frame | New narrow scene input; RAW-motion backend owns the zero-copy adapter, not the scene tracker |
| `DrivingAffinePose`, `DrivingPoseRefinement`, `DrivingPriorGuidedPose` | Driving-specific pose/proposal state | Not equivalent to a general anatomical or gaze solution |
| `VirtualDisplayPlane`, `GazeAffine`, `DisplayPlaneScore`, `GazeAffineScore` | Eye-relative inches and normalized screen fractions; robust fit diagnostics | Moved to target solver with unchanged development preset and calibration thresholds |
| `CalibratedDisplay`, `CalibrationGazeAuthority`, `CalibrationTargetEstimate` | Calibration lifetime/admission and authority binding | Keep with viewer state machine; pure fit math delegated to target solver |
| `CalibrationView`, `CalibrationFit` in `checkerboard_calibration` | Calibration-specific image/object correspondences and fitted camera parameters | Separate camera-calibration backend; not automatically the eye-scene pose |

### New, deliberately inactive contracts

`RoiId`, `SourceClock`, and `ExposureKey` distinguish ROI identity, clock
domain/session and source exposure. Equal local sequence numbers do not prove
cross-eye synchronization. The new metadata adapter requires the caller to
supply a clock/session; it does not invent one from a packet timestamp.

`BoundaryArcObservation`, `ConicObservation` and `RoiConicEvidence` preserve
native coordinates, boundary kind, raw-vs-fitted evidence and correlated support.
`JointConicRequest`, `SupportedConicSolution`, `BinocularRequest`,
`BinocularFactors`, `JointGazeRequest` and `JointGazeSolution` are scaffolding,
not working posterior models. Unknown settling/IPD/covariance remains unknown.
Accommodation is not directly observable from these outlines; any future
settling-time proxy must be labeled as such.

## Scale admission and recovery

The live Native/SAM radius adapter now carries `ExposureKey` and detector
lineage into the scene-owned scale tracker. Re-rendering a SAM answer neither
refreshes anatomical age nor counts as another observation. Automatic recovery
requires three complete, strong, mutually consistent conics spanning at least
200 ms of source time within a bounded three-second window. It can re-establish
an outdated scale only inside the existing hard support and only on a subsequent
frame. Tight adjacent-publication limits, explicit operator bounds, and frozen
support remain enforced. Unconfirmed fine-odometry transport has a 15% net budget
and a two-second anatomical-age limit; fresh coarse pose transport remains
independent. This is bounded reacquisition, not a probabilistic joint solver.

## Next incremental ports

1. Extract remaining Driving-specific pose/proposal history and RAW diameter
   admission adapters; shared radius/center trackers are now scene-owned.
   Preserve their admission/provenance contracts rather than merging similarly
   named pixel radii, physical ratios or mode-specific authority.
2. Adapt both SAM and native RAW arcs into the shared evidence packet. Retain
   alternative inner/outer edges and occlusion explanations without counting
   their shared pixels as independent observations.
3. Validate the implemented bounded joint hypotheses across the complete
   available stereo corpus; improve boundary selection and calibrated
   uncertainty. The effective pivot/camera pose should receive independent
   soft residual feedback, not become a fixed or self-fulfilling prior.
4. Add per-ROI settling/vergence coordination and target observability tests
   covering one missing eye, one defocused eye and incompatible second-eye
   evidence. Unknown depth must not become an invented 3D target.

Keep numerical helpers distinct where their contracts differ: the legacy
upper median returns zero on an empty slice, whereas the conic median filters
nonfinite values, averages the two middle entries and returns NaN when empty.
Combining these merely because both were named “median” would change fits.
