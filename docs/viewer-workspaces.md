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

An accepted recalibration overrides this pose **for the current session**.
Leaving M retains it, including when a changed gaze binding later invalidates
the affine cursor map. It does not overwrite the saved startup default.
After evaluating the result in the normal viewer, optionally click
**SAVE MONITOR LOCATION** in the VIEW inspector or call `MONITOR SAVE` on the
existing control socket. `MONITOR STATUS` reports saved and session poses,
the path, and whether the candidate is unsaved. There is no mandatory save
prompt, immediate confirmation, or automatic promotion of new calibrations.

Saving validates the physical pose and atomically replaces the settings file.
It never persists an eye's affine, provider generation, sign epoch, or a claim
that gaze is calibrated in a new session. A valid session calibration still
uses its affine and physical plane; otherwise the cursor intersects the session
or saved physical pose. The status labels distinguish these cases. These are
model estimates, not independently measured anatomical/display geometry.

## Twenty-target accuracy check

Press **backslash (`\`)** or click **ACCURACY CHECK - 20 TARGETS** in VIEW.
Return from M first. The full-screen check presents 20 shuffled grid targets
over the central 75% of width / 70% of height, two seconds per target (~40 s).
The targets differ from the nine calibration points. A white dot and spinning
indicator appear on black; the predicted cursor is hidden to avoid encouraging
the user to chase it. J need not be on. Backslash or Escape returns to the
viewer; all camera/model hotkeys are suppressed during the check.

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
