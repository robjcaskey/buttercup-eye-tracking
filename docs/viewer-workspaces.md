# Main-screen workspaces

The main screen answers three different questions without mixing their controls:

1. **ROI**: what is happening in this preview? Both previews inherit global
   presentation defaults unless a pixel-view or overlay override is explicitly
   set. Selecting another preview does not change camera or gaze settings.
2. **Linked Views**: how do the observations compare? Compare, timing, and contact
   views keep separate source times and explicitly report missing regions. Two
   visible eyes alone do not prove an admitted joint stereo solution.
3. **Overview**: what does the camera see, and where are the regions? Sensor overview
   and prompted object inspection belong here, not in an iris overlay cycle.

Tab changes workspace. F changes the view within that workspace. V changes image
appearance where applicable. Shift+Tab switches between editing global preview
defaults and overriding the selected preview. Backspace clears the selected
preview's overrides so it inherits again. Object search is an explicit start/stop action,
not a side effect of browsing. The selected eye's presentation is separate from
the physical lens/exposure controls, which necessarily affect the whole camera.

The shell always has scope navigation, a view/title toolbar, a bounded image
workspace, and an accessible inspector. The inspector separates selection/view,
shared analysis, whole-camera controls, and diagnostics. Narrow windows use a
bottom inspector rather than drawing text over imagery or silently dropping it.
Comma (`,`) cycles inspector tabs without needing a function key; tabs can also
be clicked. Long diagnostic text wraps and scrolls inside its panel. Images preserve aspect
ratio; absent or disabled data gets an explicit placeholder, never a frozen
unlabelled tile. Global images remain labelled snapshots, not live sensor video.

Validation required before calling the redesign complete: all scopes rendered
and inspected at desktop and compact sizes; independent ROI settings and linked
views tested; navigation does not start capture; explicit object search still
works; prompt editing/cancel and camera controls remain usable; source-time and
missing-data labels are honest; calibration, lightbox and J cursor still work;
build and source-tree audit pass. Geometry algorithms are not changed by this
presentation rework.

## Verification (September 6)

- `viewer_ui` tests render all six views at 320x240, 640x480, 1200x850,
  800x1200, and 904x2048; test panel bounds/non-overlap, independent ROI view
  state, missing/disabled data, object crops, word wrapping, and editor hotkeys.
  The desktop, compact, and actual tall-window images were visually inspected.
  Portrait ROI/context proportions and card-local scale clipping were corrected
  from those inspections. Fixtures are synthetic presentation data, not anatomy
  accuracy measurements (`outputs/ui-layout-review`).
- The live control-interface run selected left/right ROIs independently, changed
  only the left overlay, visited linked/timing/global/object views, and verified
  that autofocus reference and acquisition state were unchanged by navigation.
- A real `hat` object prompt compiled through the native encoder. Explicit search
  captured a fresh global image and returned no candidate in an empty-room scene;
  explicit stop resumed SAM fine-eye processing. Iris prompt text and generation
  remained unchanged throughout (`outputs/ui-live-acceptance.log`). This verifies
  the no-match/retry path, not generic object-detection accuracy. Positive mask
  cropping is covered by the synthetic renderer test.
- Focused overlay, SAM presentation, calibration-thumbnail, completed monitor
  wireframe, cursor, keyboard-map, and scoped control/prompt-isolation tests pass.
  Logs: `outputs/viewer-ui-tests.log`, `outputs/viewer-scoped-control-tests.log`,
  `outputs/viewer-ui-regressions.log`, and
  `outputs/viewer-ui-geometry-regressions.log`.
- The SAM-enabled live build runs on the camera stream. Source-tree audit and
  diff whitespace checks pass. Existing compiler warnings remain; this is not
  a claim that the full repository/corpus test suite is green.

Calibration remains a separate distraction-free modal screen. Its geometric
solver and acceptance gates are unchanged. Main-screen lightbox/cursor
compositing and source-aligned eye-overlay helpers are reused. The new view
layer does not claim a stereo solve, fresh observations from held data, or an
improvement to SN-FEIDA/localization accuracy.

## Runtime limbus refinement

**Shift+F** or **LIMBUS ON/OFF** toggles experimental shared limbus refinement.
Plain **F** continues to select a view. This changes the shared SAM/Butter Obelisk
geometry stage before either monocular or stereo solving; it is independent of
preview selection. Default remains off. `LIMBUS ON|OFF|STATUS` is the equivalent
control-socket interface. The startup environment/CLI setting supplies the initial
selection; toggling requires no restart and does not promote the model as a default.

Each submitted source snapshots the mode (both eyes of a paired request use one
snapshot). Changes invalidate pending geometry and gaze history, including a rapid
off/on cycle between RAW callbacks. The existing source RAW, pupil consistency and
bounded-correction gates still apply. Missing/rejected refinement retains baseline
geometry with the existing immutable attempt diagnostics; it is not a successful
correction. The model remains CPU-only and experimental, without a cold-bootstrap
proof. Rendering a comparison cannot enable refinement.

## Stereo solver inspection (September 12)

For **SAM3.1** and **Butter Obelisk**, use **Tab** to reach Linked Views and **F** to select **Stereo solver** (the second
linked view). The MODEL inspector also has **Open stereo solver**. The control
socket accepts `VIEW STEREO`; `VIEW STATUS` includes `stereo_solver` diagnostics.
Changing detector to a non-mask method returns to the compatible contact view.

**Conics / Masks / RAW** select presentation layers from the existing shared
segmentation outputs. The images use each proposal's original native RAW,
dimensions, and sensor origin, even if the current ROI has moved or resized.
Green points are retained, pink points excluded, cyan is the source conic, and
dashed white is the joint reconstruction when its exact sources match. RAW
remains available when no limbus fit exists. **V** retains the normal image
appearance/inheritance controls.

Both stereo panels show **IPD EST** in millimeters when both eyes contribute to a
current source-matched publication; otherwise the value is `--`. This is the 3D
separation of the solved iris centers, a proxy for IPD under uncertain metric scale
priors, not an independent measurement of pupil-center spacing. The control report
includes `ipd_estimate_mm` and its basis.

The stereo VIEW inspector also reports each eye's sampled model mass within 15 degrees
of the chosen ray, with approximate numerical error in percentage points
(two standard errors). This describes sampling precision under the current
contours and priors, not measured gaze accuracy or a guarantee that all modes
were found. It belongs to the same fresh publication as the displayed spread;
stale or mismatched source frames cannot retain it. This diagnostic does not
change gaze admission.

**Shift+3** or **STEREO ON/OFF** in the toolbar toggles the solver independently
of **G** (detector) and **F** (view/subview). Both eyes can run monocular analysis:
**3** still controls second-ROI processing. Enabling stereo enables both ROIs;
disabling stereo leaves both enabled. Disabling the second ROI also disables
stereo. `STEREO ON|OFF|STATUS` provides the same control through the socket.
Switching solvers invalidates stale gaze output and calibration source eligibility.
Browsing a diagnostic view does not select a solver or change the detector.

A compact **Stereo scene** appears beside normal ROI/linked views while stereo
is enabled. It shows the source-matched candidate eye centers, fixation rays,
short surface normals and camera in an oblique metric projection. Unresolved
solutions remain diagnostics, not accepted gaze. Click the panel to open the
full stereo inspection view. It is absent from the global sensor view, object
search, and fullscreen calibration. F refinement comparisons may display a
candidate rim, but their stereo contact uses the shared solved surface.

The backend now computes a conditional gaze
distribution and uses its angular support for direction admission. The source
conics and joint MAP geometry retain their established fit. Training and model
preprocessing are separate from this mode.

The display distinguishes a two-eye contribution, a one-eye contribution, a
provisional result waiting for its paired partner, missing/disabled sources,
search pauses, and held/expired results. A joint
publication must match both displayed proposal source keys, clocks, crop
geometry, and solver generation; two visible eyes or equal RAW timestamps alone
do not establish a stereo solve. Source lag is measured on the sensor clock;
source receipt age uses the existing exact-source host clock. Unknown receipt
age or age over the existing 900 ms SAM allowance hides the current solve while
leaving its RAW images inspectable. This is a presentation liveness bound, not
an exposure-latency measurement or a new solver gate.

The relative-support bar divides the existing **used arc weights** between the
eyes. It is not a probability. Used/rejected correlation-group counts and
weighted RMS residuals retain rejected-fit diagnostics; point count is never
treated as independent observation count. Displayed sigma is the solver's
engineering boundary allowance. Target coordinates, competing-hypothesis cost
gap, direction-sign resolution, and search-bound status come from the existing solve.
The viewer now also displays **model 90% gaze radius**, per-direction model
mass, effective sample count and unresolved integration states. Local angular
sigma describes a single direction basin; the sampled radius includes explored
competing directions and nuisance eye geometry. These are conditional model
estimates, not calibrated gaze accuracy. The selected joint gaze is preserved;
a 90% model radius above 15 degrees, or unresolved sampling, withholds direction
authority. Existing source, freshness and paired-calibration gates still apply.
See the probabilistic integration section
in [joint-conic-solver.md](joint-conic-solver.md) for assumptions and replay results.

The 23 viewer UI tests cover SAM/Student parity, missing sources, source expiry,
conditional uncertainty and compact/tall layouts. The ten live-adapter tests
pass; the former cursor-routing fixture now explicitly tests its independent
sign and exact-source gates instead of assuming its synthetic solve acquired a
sign. The native-corpus calibration diagnostic is opt-in and was explicitly run on
the three recent caches and their original target windows. The SAM-enabled
viewer builds. No new human calibration, inference or training run was
performed by the stereo work; the concurrent student work is separate. Initial rendering
artifacts are under `outputs/stereo-ui-review`; current backend/build evidence is
under `outputs/probabilistic-stereo-20260912`.

## Global gaze settings versus preview settings

There is one global gaze configuration. **G** selects its detector and **Y**
selects its compatible rough-center source; analysis bounds and other solver
settings are global too. Focus follows eyes, absolute desktop mouse output,
the J cursor, calibration and accuracy measurement use this configuration.
They do not choose a detector from the preview being inspected. Camera focus
reference is also separate from preview selection; **F2** explicitly changes
the reference used by autofocus and gaze/calibration.

Cursor projection and the main/thumbnail laser axes use the same signed,
source-clock-bound gaze vector that calibration consumes, including Native,
Driving and Clusters. The contact mesh can retain its geometric surface normal;
that normal is not a fallback cursor direction when signed gaze is unavailable.

**F** and **V** are presentation controls, not alternate gaze providers. A clean
image, a mask, an ellipse, a contact or a diagnostic display can all inspect the
same analysis. In particular, the experimental SAM tweaked-contact visualization
is still preview-only; selecting it does not silently recalibrate or alter
desktop gaze. The Overview tab means sensor context, not global settings.

| Control | Scope |
| --- | --- |
| G / Y / solver bounds | Global analysis, regardless of preview edit scope |
| Shift+Tab | Edit global preview defaults or selected-preview overrides |
| F in ROI workspace | Overlay at the chosen edit scope |
| V | Pixel appearance at the chosen edit scope |
| Backspace | Remove selected-preview overrides; inherit defaults again |
| 1 / 2 | Select left/right preview without changing analysis or autofocus |

The default edit scope is global preview defaults. Overrides are per setting:
overriding F does not freeze inherited V, or vice versa. Changing defaults leaves
explicit overrides alone. Linked F changes the linked layout, and Overview F
changes the overview page; neither creates a new eye-analysis method. Prompt
editing and calibration retain their modal keyboard handling.

Every published eye frame carries its analysis-settings revision. Before gaze
is consumed, a common policy check compares the selected detector, settings,
prompt and enabled-eye state. During a switch, old frames may remain available
for inspection, but cannot drive outputs or calibration. This is a source
validity gate, not a new anatomical confidence threshold or a reset of sign
history for presentation changes. Completed calibration's existing behavior
across legitimate sign-epoch changes is preserved.

### Verification (September 12)

The SAM-enabled live binary builds, and the no-default-features portable build
checks successfully. The targeted Rust suite passes 91 tests; two real
Sway/uinput integration tests remain explicitly ignored. The shortcut helper's
10 Python tests pass. Source-tree and whitespace audits pass; existing compiler
warnings remain.

Coverage includes per-setting preview inheritance, Student single/linked F
cycles, default mask/outline/ellipse/pupil/contact layer isolation, source-crop
movement and resizing, stale global policy rejection, canceled in-flight focus
decisions, shared calibration/cursor/laser directions and completed-calibration
continuity. Rendered desktop, compact and tall layouts and isolated Student
layers were inspected. A clockless-contact legacy laser fixture was updated to
supply an explicit published gaze source, with a negative assertion that the
clockless contact cannot drive output.

Artifacts: `outputs/gaze-preview-scope.KnG3CF/verified-tests.log` and adjacent
build logs/PPM renders. Renderer evidence is synthetic, not a new corpus accuracy
or SN-FEIDA result. Model weights, inference, fitting and anatomical confidence
thresholds were not changed. Live user calibration and desktop focus accuracy
have not been remeasured by this change.

## Desktop gaze mouse (uinput)

Desktop movement is explicitly opt-in and starts **OFF on every viewer launch**.
`scripts/toggle-mouse-output.py` controls it through `MOUSE OUTPUT
ON|OFF|TOGGLE|STATUS` on the viewer's existing Unix socket. The installed Sway
shortcuts are **Super+Shift+M** and **Mod3+Shift+M**, including resize mode.
Plain **M** still opens calibration; **J** only controls visible gaze overlays.
The mouse uses the same current contact and monitor/affine mapping as the normal
viewer, with absolute coordinates and no cursor easing, clicks or keyboard events.
It continues processing while the viewer is unfocused or on another workspace.

The helper maps only `0:0:Buttercup_Gaze_Pointer` to the single active Sway output
before enabling. With multiple monitors, specify `--output NAME` (or
`BUTTERCUP_MOUSE_OUTPUT`); it refuses an ambiguous selection. It never disables or
remaps a physical mouse. The kernel endpoint is opened as the viewer's user:
missing write permission to `/dev/uinput` raises a visible warning and leaves
output OFF, without changing permissions or invoking sudo. Device-write failures
also disable output and warn. Status is in the VIEW inspector and control API.

Movement pauses during calibration/accuracy screens, prompt editing and object
search, or without current signed contact evidence. Only new source observations
produce motion: old/repeated results are not replayed to fight the physical mouse.
A 500 ms host-arrival-age bound rejects stale sources using the exact solve's RAW
clock, not a newer transported ROI's timestamp. This is a liveness heuristic,
not a claim of measured exposure latency. Device coordinates clamp at the screen
edge; recorded predictions and the existing geometric mapping stay unclamped.
OFF destroys the virtual device and invalidates in-flight projection work.

## Focus follows eyes (no pointer movement)

**Super+Shift+F** or **Mod3+Shift+F** toggles a separate, default-OFF window-focus
mode. The same shortcuts stop it, including in resize mode. The helper is
`scripts/toggle-mouse-output.py --focus`; its control API is
`GAZE FOCUS ON|OFF|TOGGLE|STATUS`. No `/dev/uinput` access is required.
Enabling focus mode turns pointer output OFF; enabling pointer output turns
focus mode OFF. Turning either mode OFF does not enable the other.

Look inside an already visible window for at least 350 ms and three distinct
signed gaze observations spanning at least 350 ms on the source clock too.
Focus is window-level keyboard focus, not a click on
a text field or button. Repeated/held, stale, off-monitor, pre-enable or
unresolved-sign results cannot advance the dwell. New gaze bases, moved window
rectangles, workspace changes, manual focus changes and gaps reset it. A 12
logical-pixel inset reduces switching at borders. Hidden tabs/workspaces are
excluded; floating windows occlude tiling, and ambiguous floating overlaps
abstain. Nothing opens, moves a window, switches a workspace or clicks.

The same `desktop_gaze` sampler feeds both modes, independently of viewer focus
or frame callbacks. Geometry and calibration mapping are unchanged. Sway layout
queries and focus commands run in a bounded-IPC worker, outside SAM and drawing.
Calibration/accuracy screens, prompt editing and object search pause both modes;
focus mode also pauses in non-default Sway binding modes. Connection failures
disable it with a warning. The VIEW inspector reports mode/dwell status.

The Sway configuration uses `mouse_warping none`, and focus commands enforce it,
so Sway cannot move the pointer as a side effect of a keyboard-focus change.
This also leaves the pointer stationary during ordinary keyboard navigation;
OFF does not restore automatic warping. Physical mouse motion remains normal.
Targets are limited to the selected output's *currently focused workspace*;
Sway rechecks that criterion at command execution to avoid switching back to a
workspace the user has just left. With multiple outputs, set
`BUTTERCUP_FOCUS_OUTPUT` when launching the viewer; ambiguous selection refuses
to enable. No compositor patch or desktop restart is needed.

For Sway installations, add the two `--focus` bindings beside the mouse bindings
in both the default and resize modes, and keep `mouse_warping none` in the config.
Bindings do not auto-enable either mode at login or viewer startup.

## Monitor defaults and deferred saving

The physical monitor pose (eye-relative center in inches, right/down axes,
width and height) loads from `outputs/settings/monitor-location.json`. The
initial saved pose was explicitly selected from accepted calibration session
`sam31-mouse-3d-1788765280-345175455`: center
`[-4.242625177491027, -16.718478032593996, 22.343548660299646]` inches,
center distance approximately 28.23 inches, EDID dimensions 590 x 333 mm.
The same pose is the built-in fallback if the file is absent. Rotation is saved
as orthonormal axes, preserving pitch/yaw/roll without Euler-angle ambiguity.
The decorative wireframe viewing orbit is not the monitor's physical rotation.

An accepted M calibration saves its cursor mapping to
`outputs/settings/gaze-calibration.json` and saves the offered monitor pose.
The mapping reloads at startup; pressing M replaces it after a successful new
calibration. Its input coordinate space is persisted with the coefficients.
Monitor pose controls remain separately available through **SAVE MONITOR
LOCATION** in the VIEW inspector or `MONITOR SAVE` on the existing control
socket. `MONITOR STATUS` reports saved and session poses, the path, and whether
an offered candidate is unsaved.

The separate monitor-save operation validates the physical pose and atomically
replaces the monitor settings file; it does not replace the eye's saved affine.
A loaded or newly accepted eye calibration uses its affine and physical plane;
otherwise the cursor intersects the session or saved physical pose. The status labels distinguish these cases. These are
model estimates, not independently measured anatomical/display geometry.

## Twenty-target accuracy check

Press **backslash (`\`)** or click **ACCURACY CHECK - 20 TARGETS** in VIEW.
Return from M first. The full-screen check presents 20 shuffled grid targets
over the central 75% of width / 70% of height, two seconds per target (~40 s).
The targets differ from the nine calibration points. A white dot and spinning
indicator appear on black; the predicted cursor is hidden to avoid encouraging
the user to chase it. J need not be on. Backslash or Escape returns to the
viewer; camera/model adjustment hotkeys are suppressed during the check.
B/N lighting and S/H RAW-recording controls remain available.

The test freezes the current cursor mapping and monitor pose. It does not fit,
recalibrate, or save monitor defaults. Each point gets a 650 ms settling period;
then a RAW source-clock boundary excludes delayed answers from the preceding
target. Only new, finite, sign-resolved, source-aligned gaze observations count.
Held/repeated samples, predictions ahead of the current RAW source, and stale
answers cannot inflate the sample count. Off-screen predictions are scored
without clamping, and no “best stable cluster” is selected to improve a score.

The result screen reports mean target error, median/P95 sample error in
presentation-buffer pixels, percentage of screen diagonal, targets with data,
and fresh sample count. JSON under `outputs/gaze-accuracy/` additionally records
every target and accepted sample, signed bias, RMS jitter, frozen mapping,
missing-observation reasons, and the final completion/cancellation reason.
Missing targets produce no error estimate, **not zero error**, and coverage is
always reported alongside accuracy. Target-mean error weights measured targets
equally; sample percentiles are sample-weighted. Report errors remain explicit.

A source-clock restart, gaze-basis change, display resize after settling,
focus loss or presentation interruption aborts the run rather than combining
incompatible measurements. Partial reports are retained. The test assumes the
user looks at each dot; fixation is not independently verified. Pixel units
refer to the current presentation buffer, not calibrated physical pixels,
millimeters or angular degrees. It creates no RAW recording and does not
alter an existing recording. Accuracy on Rob's eyes remains to be measured by
running the check; synthetic zero-error/known-offset tests are software tests,
not an empirical gaze-accuracy result.


## Calibration lighting frame

### Calibration target preview

During each stationary **M** target, the eye thumbnail is fully visible through
250 ms, fades to zero at 500 ms, and stays hidden. One framebuffer pixel at the
crosshair center cycles red, green, blue, white and black, 200 ms per color;
the surrounding white crosshair stays fixed. Its rendered color and phase are
recorded with the target appearance. Calibration results exclude
this preview: sampling opens only after a successfully submitted hidden buffer,
then requires the exact observation's original RAW host arrival to be strictly
later than that submission, as well as the existing 500 ms source-clock settle.
A delayed redraw therefore extends exclusion; delayed inference cannot turn a
preview receipt into a fresh sample. Each target, sign relock or source restart
requires a new hidden submission. Duplicate sources still get only one vote.
Native RAW recording retains the preview. Scene metadata records opacity and
the gate; session metadata records the rule and each target's first hidden
submission elapsed time. The first hidden presentation's scene describes the
still-closed pre-submit gate; its successful submit bounds establish the opening.
These host arrival/submission times do not measure sensor exposure or display
scan-out: transport and display latency remain unknown, so they cannot certify
physical exposure entirely after the last visible thumbnail.

### Recorded relative-motion stimulus

**Shift+\\** starts a recording-only target session, separate from **M** mouse
calibration and the unshifted **\\** accuracy check. It does not fit, clear or
replace the monitor mapping and does not wait for a successful gaze solve.
It refuses to interrupt another RAW recording or calibration.

The `micro-motion-five-locations-v2` recipe repeats the same micro-motion routine
at five positions: center, upper-left, upper-right, lower-right and lower-left
within the light-frame interior. Each position begins with two seconds to settle,
then uses 200 ms target steps and returns, with horizontal/vertical amplitudes of
1–5, 0.5 and 0.25 physical framebuffer pixels plus zero-motion controls.
Total duration is 106 seconds (21.2 seconds per position). A faint tail shows the
preceding three commands at the current position and clears on relocation.
The recording identifies each position and its local offsets independently.
Fractional positions use achromatic area-sampled antialiasing; this
is not RGB panel-subpixel addressing or proof of subpixel eye-tracking accuracy.

**B**, **N**, **[**, **]** retain the same light-frame toggle, pattern/Auto Cycle
and width controls. The reusable timed-stimulus session separates a versioned
target recipe from the common recording, lighting, gaze logging and presentation
shell; new recipes should use that shell rather than implement their own lights.
Choose the frame width before starting; changing it during the sequence aborts
the run so the measurement origin cannot move mid-step. Toggling frame lighting
or changing its color pattern preserves the target locations.
Escape stops an active session; once stopped, Escape returns to the viewer.
Focus loss, resize, a long presentation gap, RAW failure or the recording time
limit aborts the sequence. The target clock starts only after the writer confirms
recording. Completion means the sequence and archive finished, not a fit passed.

Recordings use `outputs/calibration-corpus/stimulus-micro-motion-*.tar` and a
`.session.json` sidecar declaring `purpose: recorded-stimulus`,
`presentation_type: micro-motion`, recipe/version, planned offsets/durations,
initial/final lights and outcome. The sidecar is created while arming and finalized
after the writer finishes. Mouse-calibration sidecars explicitly identify their
different purpose. Actual presented targets and tails (distinct roles), fractional
coordinates, lightbox state/phase, predicted gaze and host submission bounds are
in the existing `metadata.oim1` stream beside native ROIs and thumbnails.


Commanded display displacement **is not ground truth eye displacement**. Pursuit
delay, saccades, fixation drift, camera exposure, display refresh and unknown
scan-out timing must be modeled before using these recordings for training.
The saved successful presentations, not every planned step, define what was
actually submitted. The whole session and both eyes stay together in dataset
partitions; no subpixel accuracy claim follows from fractional rendering.

**B** toggles the existing lightbox; **N** advances its pattern once per keypress.
Startup is OFF with **SOLID WHITE** selected. Cycling while OFF only arms the
selection. From white, the first N selects **CLOCK / 5 HZ** and the second
selects **AUTO CYCLE / 4S**. Auto Cycle visits all thirteen lighting patterns,
four seconds each, and repeats; it excludes the separate optical-clock mode.
Pressing N again leaves auto cycling for the individual manual selections.
The manual cycle returns to white after warm/cool/red/green/blue solids,
rotating color, white pulse, color pulse, checker pulse, checker strobe,
checker horizontal color sweep and checker diagonal color sweep. The current
selection appears in the viewer inspector and calibration/accuracy footer.
**[ / ]** retain their frame-width controls in the viewer and M calibration.
The initial width is 16% of the shorter window dimension; each bracket step
changes it by four percentage points, from 4% through 44%. In the ordinary
viewer, enable the frame first so brackets control lighting thickness.

The colored checks have complementary hues and approximately ten tiles across
the shorter display dimension. Checker pulse uses a 3.2-second brightness cycle;
checker strobe uses a 1 Hz bright/dim cycle (50% duty, 30% minimum brightness).
Checker colors rotate over 12 seconds; horizontal and diagonal modes add a
spatial hue sweep. Pattern selection or B toggle restarts phase; redraws use
elapsed host time rather than frame counts. Auto Cycle includes the 1 Hz
checker strobe; neither auto cycling nor strobing is the startup mode.
B stops the lighting immediately.

The same frame surrounds M calibration and the twenty-target accuracy test.
B/N remain available during accuracy testing; its frame is capped at 8% of the
shorter dimension so the existing targets remain inside. Target coordinates,
progress indicators, calibration schedules and source clocks are unchanged.
The keyboard guide shows the lighting and RAW recording shortcuts during the test.
These are display lighting perturbations for recordings, not camera exposure
controls or evidence that a model has become more robust.

The existing recording scene's calibration metadata includes `lightbox` with
recipe `lightbox-v2`, enabled state, selected `pattern`, `effective_pattern`,
effective width and `phase_elapsed_ns` at rendering. Auto cycling additionally
records `auto_cycle_index`, `auto_cycle_step_ns` and the concrete pattern's
`effective_phase_elapsed_ns`, so its local animation phase can be reconstructed.
Older `lightbox-v1` recordings retain their manual-pattern interpretation.
Phase starts at the last toggle/pattern cycle;
this is host-render timing, not measured exposure or scan-out. It permits
sorting recorded presentations by lighting condition and reconstructing the
pattern phase alongside the existing presentation timestamps. No separate
recording subsystem or training material is created.

## Windowed J reticle coordinates

The J reticle uses the calibrated monitor prediction, converted into the client
window's desktop rectangle. Resizing or tiling the window does not rescale the
monitor into it. Predictions outside the client remain available to recording
and desktop gaze consumers, but are not drawn at a false clamped window edge.

On Sway, a bounded asynchronous read-only IPC query supplies the logical output
and client rectangles, including borders and fractional output scaling. Rendering
and camera ingest do not wait for IPC. Resized, missing or stale geometry hides
the reticle until the next matching snapshot; fullscreen mapping stays direct.
Native desktop window positions are used where the windowing backend supplies
them. Unsupported windowed compositors no longer assume a fullscreen origin.
Recording metadata distinguishes monitor predictions from window-local drawing
and includes the transform. Existing monitor calibration/solver math is unchanged.

## Automatic gaze calibration save/reload

An accepted M calibration automatically saves its monitor plane and six affine
coefficients to `outputs/settings/gaze-calibration.json`, atomically replacing the
previous accepted result. Startup reloads it if present and selects its reference
eye (enabling that ROI). No mandatory validation targets or repeat calibration
are required. Run M again when the mapping is inaccurate or the camera has moved.
A cancelled, refused, or failed M attempt preserves the previous saved mapping.
The system mouse cursor is hidden in calibration and accuracy screens and returns
when the normal viewer resumes.

Restored mappings retain their training metadata but do not depend on old
process-local authority or prompt-generation numbers. Existing same-eye/detector
routing and current-source/resolved-gaze requirements still apply. Reloading the
mapping does not turn held or unresolved geometry into fresh tracking evidence.
The independent monitor-location save remains available for the physical pose.

## Camera mounting assumption trial

**F8** cycles **Flexible → Below eyes → Above eyes → Flexible** without key-repeat
cycling. The Selection panel also has a button, and the inspector shows the
current assumption. Each successful change is atomically saved to
`outputs/settings/camera-mount.json` and restored at startup. A missing setting
starts Flexible. `CAMERA MOUNT FLEXIBLE|BELOW|ABOVE|NEXT|STATUS` exposes the same
setting through the viewer control socket.

This is an upright, screen-facing shortcut, not a measured camera pose. Below
favours a screen above the camera optical axis (sensor-up normal); Above favours
sensor-down. Physical camera height alone does not imply this gaze direction:
roll, pitch, looking elsewhere, or a different screen arrangement can make the
assumption wrong. Flexible retains the existing evidence-based sign selection.

Monocular geometry selects a whole antipodal normal after four fresh source
observations; duplicate/held frames cannot vote. Absolute vertical normal
component at or below 0.05 remains unresolved under this assumption. Selection
is labelled `mounting-assumption`, including during calibration acquisition;
it is conditional permission to use the chosen sign, not independent pupil or
motion evidence. Switching modes clears sign history and invalidates current
output settings. All normal gaze consumers use the shared result.

Stereo considers the existing bounded set of optimized hypotheses and selects
a compatible candidate for both contributing eyes. It never flips a fitted
normal after solving and never upgrades posterior confidence. No compatible
candidate produces an unavailable result; stricter stereo dropouts are possible.
Recordings include the active mounting assumption. Limbus fitting and area
estimation do not change merely because a monocular sign is selected.

M calibration uses a nine-point grid spanning 10%–90% of the visible screen in
both dimensions, including the center. The 10% edge margin keeps the reticle
visible; target rendering, recorded coordinates and fit coverage checks use the
same layout. Existing saved calibrations still reload; run M to fit the wider grid.

The wider M grid holds each target for at least 2.5 seconds, retaining the latest
12 fresh samples and the existing stability/fit acceptance checks. Routine
automatic exposure adjustments and focus/exposure status polling run through a
single bounded camera-maintenance worker. Slow control responses cannot block
RAW ingest; no stale adjustment backlog is queued. All camera sockets still use
the mandatory cooperation boundary, and the 350 ms RAW limit remains in force.

The F cycle includes **STEREO SEGMENTS / FITTING SUPPORT** for SAM3.1 and Butter Obelisk, in both preview and linked views. The stereo page also offers a SEGMENTS layer. It draws smooth sections of the selected solver conic on its exact inference-source RAW pixels, with separate dots at the unchanged bounded residual samples. Crosses mark rejected samples; rejected evidence is never drawn onto the fitted conic. Alternating colors and numbered callouts distinguish segments. The curves are model output, not smoothed or substituted observations; the UI does not refit anything or change gaze authority. Callout percentages and the eye total divide accepted evidence weights by the sum across both eyes. They are contour-support shares, not causal gaze influence or calibrated probabilities. Viewing this layer does not enable stereo automatically; Shift+3 remains the independent solver toggle.

Recorded `joint_conics.arcs` now retain `points_roi_px`, `arc_index`, `mask_level`, and a full source key. Points include the selected mask-level displacement and remain available for rejected groups. The existing source-projected-conic records provide ROI origins and dimensions. Older archives cannot recover exact segment pins from their full fitted ellipses alone.

Live segment callouts reserve their position, number and color by boundary kind and recorded evidence-group ID within each detector/source-clock context. These are diagnostic group IDs, not claims of tracked anatomical identity. Percentages refresh at most four times per second from advancing sources. A missing segment retains only its label for up to 350 ms with `H`; afterward its reserved slot shows `--`. Old segments and leader lines are never held. Callout storage is bounded to 20 slots per eye and resets on detector, prompt, dimensions or source-clock changes.


## Wide-field M calibration mapping

M still collects nine source-bound fixation clusters over 80% of the visible
screen. It first fits the metric display plane. A supported historical affine
on projected unit-direction XY keeps its existing behavior. When that linear
map cannot explain the wider, oblique field, calibration projects the ray onto
the accepted display plane and fits a small affine correction in screen UV.
Both forms retain the existing center/three-corner coverage, seven-target
minimum, residual, affine-conditioning and shared-support/correction limits.
The plane and its correction use the same data; their agreement is a consistency
check, not independent validation or a calibrated confidence probability.

`GazeAffineInput::target` is shared by completed calibration, the viewer cursor,
accuracy checks and desktop mouse/focus output. Recording metadata includes the
affine input space. New saved mappings use `buttercup-gaze-calibration-v2` and
require `gaze_affine.input`; v1 files retain projected-direction coordinates.
Missing or unknown input spaces in v2 are rejected instead of reinterpreting
coefficients. A ray without a forward monitor intersection remains unavailable.

The September 13 diagnostic uses six completed sessions from the current live
run, including one accepted session, three 2D-affine failures and two display
pose failures. Runtime reports and source-matched RAW reviews are under
`outputs/calibration-affine-20260913`. Centroid fit residuals are not held-out
accuracy; leave-one-target-out reports keep both failed folds and outliers.
