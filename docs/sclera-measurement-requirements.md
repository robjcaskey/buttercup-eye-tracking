# Measurement needed for sclera-based iris-sign validation

Prepared 2026-09-18 UTC. This is a host-side recording specification, not a
request to change camera firmware or bypass camera cooperation.

## Why the existing recording is insufficient

The retained 202 SAM and 193 Obelisk ambiguous rows come from 225 unique RAW
exposures in `sam31-mouse-3d-1789300371-414277114.tar`. The native
[measurement audit](../outputs/roi-sclera-measurement-audit-v1/audit.json)
decoded all 21,132 OIM1 metadata records and the 2,290-row frame index.
Its lightbox records describe only `SOLID WHITE`: 6,682 metadata occurrences
disabled and 66 enabled, including a repeated snapshot. These counts are not
independent exposures. Optical-clock metadata is null. Camera intrinsics,
camera-to-scene transform and sensor-to-host clock model are null throughout;
the monitor pose is explicitly estimated or preset, with no uncertainty bound.
EDID provides screen size, not its pose or emitted radiance.

Natural-light fits have not recovered a sufficiently precise globe center.
The native lid detectors and provisional visual estimates do not provide
reviewed anatomy or hidden-center truth. More confident predictions on these
same inputs would not establish sign accuracy. No additional model should be
trained using the current assistant estimates as labels.

## Sufficient measurement recipe to test next

Keep one fixation target visible while recording a reference illumination
condition and at least four individually identifiable, spatially separated
neutral-light conditions. Small screen patches at separated positions are an
option if their geometry and radiometry are calibrated. Record several cycles
and reserve complete cycles/light conditions for evaluation. Start with static
geometry, then explicitly add head/camera motion and lighting changes.

For the diffuse, unshadowed approximation, ambient-subtracted measurements obey
`I = rho L N`. Three independent rows of the **known effective lighting matrix**
are sufficient to recover the normal up to positive albedo; extra conditions
provide a consistency check. Record and evaluate singular values/conditioning,
not just the number of patterns. Changing colors or brightness with one fixed
spatial lighting distribution does not automatically supply independent
directions. Three directions are a sufficient recipe for normal recovery, not
a claim that every two-candidate sign test mathematically requires three.

A nearby display is an extended emitter. Use its calibrated emitting regions
in the forward model, or bound the error of any point/distant-light
approximation. Do not infer a separate unconstrained lighting geometry from
each candidate iris interpretation and call it independent evidence.

Retain Rob's two-second target acclimation and allowance for at least 40%
off-target time. A commanded target is not proof of fixation. Without optical
timing, allow the previously specified extra 100 ms display jitter and record
the remaining uncertainty; never relabel host submit time as exposure time.
Use stable interior portions of sufficiently long illumination plateaus and
reject intervals whose timing/motion uncertainty crosses a transition.

## Required archive evidence

Use the existing RAW bundle and bounded OIM1 metadata transport. Preserve
absence explicitly rather than filling unknown quantities with model output.

- Native packed RAW, crop origin, stride, CFA phase, exact source timestamp,
  camera attachment/session, source epoch and sequence for each eye. Preserve
  before/target/after exposures and actual global snapshots when available.
- The successfully submitted illumination condition, stable pattern/epoch ID,
  literal generating recipe and parameters, emitting pixel regions and linear
  channel intensities. Include the target, thumbnails, text and other screen
  content that changes emitted light. Retain dropped-event intervals and
  original snapshot timestamps.
- Sensor exposure interval or bounded timing model, rolling-shutter behavior
  if applicable, integration time, analog/digital gain, black level and clipping
  limit when the external camera service exposes them. Otherwise mark them
  unavailable and test whether the measurement remains usable. Record changes;
  a presumed fixed exposure must not silently replace missing telemetry.
- Camera intrinsics/distortion and camera-to-display/light geometry with
  coordinate conventions, units, calibration provenance and uncertainty.
  Track geometry changes or invalidate the calibration when the camera moves.
- The display transfer/radiance calibration, patch directions/strengths or
  calibrated emitter model, and reference illumination. Include uncertainty
  from ambient drift, motion between conditions, shadow and specular exclusion.
- Independently reviewed sclera, limbus and upper/lower lid portions, including
  uncertain/occluded areas. Use the canonical labeler specified in AGENTS.md,
  native RAW triplets and predictions hidden until `SAVE + DONE`. Keep labels
  in the capture's `annotator/labels`; never convert predictions or assistant
  estimates into human labels.

## Independent reference and acceptance

First validate the geometry on a physical eye-like target with independently
known orientation/center, or another measured geometric reference. A synthetic
sphere is useful arithmetic validation but does not establish real imaging
accuracy. A real human-eye validation then needs a reference independent of
the tested conic branch or shading fit—for example calibrated observations
of the same eye from another viewpoint with stated reconstruction error.
The left-eye and right-eye ROIs from one camera are not those two viewpoints.

If fixation targets supply weak supervision, record camera/display geometry,
eye-origin uncertainty, distraction handling and the optical-to-visual-axis
offset assumption. Do not count target-direction agreement as independently
measured 3D optical-axis sign accuracy.

For each frame, retain both circle explanations and compare the independently
estimated globe center with their permitted center families. The sufficient
condition is `center_measurement_error + model_departure < family_gap / 2`,
while still being compatible with at least one family. Camera/conic error and
anatomical deviations consume the allowance. Our current broad-radius cohort
has minimum half-gaps of 33.80 SAM and 35.17 Obelisk native pixels **before** those
additional allowances; these are requirements, not measured estimator errors.

Evaluate full held-out sessions/lighting cycles, abstentions, wrong accepted
signs, localization error and coverage. Apply the area gate before sign work;
report SN-FEIDA only when independent scale support exists. Do not turn rejected
frames or held predictions into fresh observations. Report the historical
202/193 cohort separately: a new recording cannot retroactively supply its
missing physical truth.

Before any custom-model training or reuse of derived training material, follow
`bootstrapability.md` and the native bootstrap DAG preflight. The prepared
twelve-frame lid review can improve anatomical evaluation, but lids alone do
not certify globe-center position or physical sign. No new capture has been
started and no camera/control connection is needed to review this document.
