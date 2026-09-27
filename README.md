# Buttercup

Buttercup is an experimental Rust viewer and analysis workspace for lossless
RAW eye-camera data.

![Buttercup viewer with sensor overview and eye ROIs](docs/viewer-overview.png)

It combines a coarse sensor view with native-resolution eye regions and makes
its motion, segmentation, and projected-geometry state visible. The approach
could be useful for low-latency gaze input, calibration, and related
camera-space interaction work, but it is still a research prototype.

The [geometry architecture and type inventory](docs/geometry-architecture.md)
records the extracted module boundaries, coordinate/uncertainty contracts,
and which joint conic/binocular capabilities remain unimplemented.
The [iris-area validation guide](docs/flat-tire-area-and-motion.md) defines
scale-normalized frontal-equivalent iris disk area (SN-FEIDA), its uncertainty,
and the matched corpus checks used to evaluate geometry changes.
The [bootstrapability contract](bootstrapability.md) requires current-checkout
retraining from RAW, human labels and pinned SAM3 assets, with explicit model
ancestry and separate proof for new-user onboarding.

For reproducible random RAW samples and SAM3.1 prompt comparison contact sheets,
see [the contact-sheet workflow](docs/sam-roi-contact-sheets.md).
For pink waterfall lines bending into the pupil, see
[the pink waterfall workflow](docs/pink-waterfall-contact-sheets.md).

For the current upright rig, use the temporary **camera below eyes** assumption
(`F8`; saved between sessions). With stereo enabled, `M` now starts at the top
plus to establish direction, then collects display calibration; routine tracking
uses conditional continuation and allows off-screen gaze. `Shift+M` repeats just
orientation when a saved display mapping is available. After a camera move, use
`M` to update that mapping. Live startup warns when below-eyes is disabled.
Offline historical controls retain explicitly recorded legacy priors.
See [the mounting policy and its limits](docs/camera-mount-assumption.md).
The global **Continuous sign** button (`F9`) enables the separate experimental
trajectory module for single-eye and joint paths; the setting is saved and restored. Its camera-invariant scores
retain ambiguous alternatives; they do not yet authorize calibration or make
the saved screen mapping follow a moving camera.
For source-checked calibration reconstruction using the recorded stimulus,
see [the calibration playback guide](docs/calibration-playback.md).

The viewer also owns a small **wleyes** desktop cartoon on a transparent
background. Its pupils follow the shared current gaze and, when valid, calibrated
screen mapping. Without an accepted sign/matching calibration it displays
camera-relative motion labeled “APPROXIMATE”; an ambiguous branch can be wrong.
This display-only estimate never controls the mouse, focus or calibration.
Sleepy eyes and “WAITING FOR GAZE” mean no fresh gaze estimate. With Rob's Sway bindings,
**Super+Y** or **Super+Shift+Y** toggles it (also available with Mod3).
It starts off and opens floating in the upper right; right-click hides it.
It neither moves the pointer nor changes keyboard focus. There is no separate
wleyes executable: the existing shortcut client uses
`scripts/toggle-mouse-output.py --eyes`, and the viewer accepts
`WLEYES ON|OFF|TOGGLE|STATUS` on its existing control socket.

SAM3.1 runs inside the viewer with local LibTorch/model assets, without a SAM
server. Interactive startup registers optional CUDA dispatch hooks before CPU
Obelisk loads, so a later G switch to SAM works in the same process. CPU-only
offline/training paths retain their CPU initialization policy. Missing SAM
answers display the actual per-eye worker state with wrapped error text.

Buttercup expects a compatible external RAW camera service over TCP. Camera
firmware and device-side control live elsewhere.
For reboot, hotplug, missing-service and ownership failures, see the
[camera startup and recovery guide](docs/camera-startup.md). Connection logs
identify the failed stage and the external Podbay steps needed to restore it.

Coarse semantic reacquisition uses the MediaPipe Tasks C ABI directly from
Rust. It consumes a lossless 500x375 GRAY16 sensor overview, converts the
linear 10-bit samples in process, and reads the refined iris landmarks without
starting Python, loading libpython, or importing a Python package.

The native runtime and face-landmarker model are external runtime data and are
not stored in Git. By default Buttercup looks for them at:

```text
data/runtime/mediapipe/libmediapipe.so
data/models/mediapipe/face_landmarker.task
```

Set `BUTTERCUP_MEDIAPIPE_LIBRARY` and `BUTTERCUP_MEDIAPIPE_MODEL` to use other
locations.

```bash
cargo build --release
scripts/run-viewer.sh
```

SAM3.1 uses the native LibTorch/CUDA runtime under `data/runtime`. The launcher
starts in SAM31 when built with SAM support and the model, semantic prompts,
and tracker weights are present; otherwise startup defaults to Native.
`--segmentation native` (or `BUTTERCUP_SEGMENTATION_MODE=native`) overrides
that choice. `BUTTERCUP_ENABLE_SAM31=0` skips the optional SAM build;
`BUTTERCUP_ENABLE_SAM31=1` requires its runtime/assets instead of allowing
the launcher's automatic fallback. Explicit SAM requests still report errors
if loading or CUDA initialization fails.

`G` also offers the experimental `EYE-STUDENT` mode immediately after SAM.
It uses a compact CUDA-trained SAM mask student and the same RAW/conic/3D gaze
pipeline. Start it directly with `--segmentation eye-student`; weights and their
manifest live under `data/models/eye_student_v1.*`. It is faster but has lower
pupil coverage on the initial replay, so SAM remains the default. See
[training, measurements, and limitations](docs/eye-student.md).

The Student **Contact Geometry** preview shows the same source-matched candidate
as the stereo scene even when its direction is unconfirmed. The visible warning
does not authorize mouse/focus output; those still require accepted live gaze.

SAM's `F` cycle also includes experimental **Tweaked Contact Geometry** after
the original contact view (`6/8` for an ROI, `4/4` for linked ROIs). A separate
native-RAW local model makes bounded limbus corrections and distinguishes the
surface from deeper visible optical continuation. This is presentation-only,
not a change to gaze/calibration. See [the refiner and corpus results](docs/limbus-refiner.md).

The launcher
prefers `data/models/sam31_semantic_video_shared_features_u8.pt` when available,
and falls back to the older `sam31_semantic_video_features_u8.pt` export. The
shared export runs the image encoder once, then performs separate iris and
pupil text-conditioned decodes. Model weights remain external runtime data.
To generate that export from the supported detector archive:

```bash
cargo run --profile live --bin buttercup-sam31-video-graph -- \
  data/models/sam31_semantic_dynamic_u8.pt \
  data/models/sam31_semantic_video_shared_features_u8.pt --shared-features
```

When the SAM pupil-center mode is selected, the pupil prompt is followed by
flat-tire contour exclusion, limbus-relative shape/size checks, reflection-aware
RAW edge validation, and short temporal consistency checks. A failed pupil
prompt does not discard a valid iris or silently acquire a different RAW dark
component. `BUTTERCUP_SAM31_SEMANTIC_PUPIL=0` selects the older RAW-component
proposal path for comparison; `BUTTERCUP_SAM31_SHARED_FEATURE_PROMPT=0` disables
feature reuse. These switches do not bypass RAW publication checks.

Offline replay uses the same worker and original sensor timestamps, without
recorded prediction or annotation seeds. A stride simulates dropped frames:

```bash
scripts/run-viewer.sh --offline-sam-sequence-eval \
  outputs/replay.json outputs/EXTRACTED_CAPTURE subject-right 0 32 1
```

For a browser/video showcase using the live annotation renderer, set an output
directory that does not already exist:

```bash
BUTTERCUP_REPLAY_RENDER_DIR=outputs/showcase/native-frames \
  scripts/run-viewer.sh --offline-sam-sequence-eval \
  outputs/showcase/replay.json outputs/EXTRACTED_CAPTURE subject-right 0 120 1
```

Each replay case includes a `point_stream` with source-keyed retained/rejected
flat-tire points, dense ellipse points, and a fresh presentation contact pose.
Nanosecond point-stream timestamps are strings for lossless JavaScript use.
The optional renderer exports RAW color, blue-filter, RAW-luma, flat-tire, and
virtual-contact PPM streams using the same preview and annotation functions as
the live viewer. Missing fits remain visible as missing; stale contacts are
not exported as fresh. Rendering is supported only for native ROI replay.
All artifacts belong beneath the checked `outputs` link.

Replay includes the post-SAM pupil solver under an explicitly optimistic
settled-focus assumption. Candidate counts are not accuracy measurements;
pupil accuracy needs pupil-specific human reference.

For offline arc-combination experiments, export the ordered native-pixel SAM
outlines before the existing ellipse fitter can discard them:

```bash
scripts/run-viewer.sh --offline-sam-outline-export \
  outputs/iris-outlines.json outputs/EXTRACTED_CAPTURE subject-right
```

This uses the live single-frame preprocessing and outer-iris prompt but not
video-memory propagation. The report includes rejected candidates and the
existing stateless fit for comparison; it never reads recorded predictions or
human labels, and does not change the live fitter.

When Keyboard Peeper is running, the viewer optionally publishes its current
hotkeys over the versioned binary KPP/1 Unix-socket protocol. Mode-dependent
buttons are retained in the map with an enabled or disabled state, and updates
are atomic. The registration clears as soon as the viewer loses focus. No
peeper installation or library is required; set
`BUTTERCUP_KEYBOARD_PEEPER=0` to disable the silent background publisher.

`W` pauses/resumes automatic ROI and sensor-slice following, which starts on
at every launch. This is a host-only session control: the camera transaction
protocol stays active, and no environment/config option selects it. An already
queued move can finish after pausing. Global reacquisition remains separately
controlled by `R`.

Press `M` for eye-to-screen calibration. Its nine targets stay in the central
20% of the screen and advance after a minimum 1.5-second hold once six distinct,
stable gaze readings are available. A 500 ms settling interval excludes target
transitions; slow or unstable tracking can extend a step. The compact live-eye
thumbnail is one-third its former width and height, with a matching small
fixation spinner. Status text stays hidden for the first three seconds of each
point, and paired RAW recording remains automatic during calibration.

Press `Z` in the viewer to start the full-screen optical screen clock. It
shows a stationary fixation target over a time-coded chromatic field
and writes a presentation manifest under
`outputs/screen-reflection-calibration/`. Press `Z`, `Esc`, or `Q` in the
stimulus to return to the viewer.

While the clock is running, its upper-left readout reports optical recovery
as `WARMING`, `SEARCHING`, or `LOCKED`. The standalone **Z** launcher now uses
a checked temporal whole-field color sequence, including recovery of an unknown
reflection polarity. It collects approximately 13 seconds at 5 symbols/second
before decoding 63-symbol words. Ambiguous, missing and contradictory evidence
is rejected. The source is native RAW, independent of eye-tracking admission.
The monitor must be powered on; a window reported as visible on a powered-off
output cannot supply an optical clock. Rendering normally follows Wayland frame
callbacks, with a bounded watchdog that checks display availability.

The lag is host display submission to arrival of the camera packet carrying the
recovered code, **including an unknown position within the 200 ms held symbol**.
It is not precise display-to-exposure latency or a calibrated sensor/host clock
transform. The fixed-reflector test acquired live with 41 checked receipts;
native replay succeeds for both ROIs and rejects powered-off, reversed-time and
constant-signal controls. A subsequent live eye-ROI trial produced 35 checked
recoveries. Precise transition recovery and isolating corneal/pupil-only support
remain open; see [the clock investigation](docs/optical-clock-debugging.md).

For standalone optical debugging, the viewer binary also accepts
`--screen-clock-stimulus --temporal-code --fixed-target --code-hz 5 --amplitude 0.12
--duration-seconds 35`. Start a RAW recording in the camera viewer first.
The older spatial diagnostic remains available using
`--screen-clock-stimulus --large-cells --fixed-target --code-hz 5 --amplitude 0.12
--duration-seconds 30`. This keeps the same checked V3 code but draws one 8x4
tile, with four times the area per cell, instead of four 8x4 copies. Start a
RAW recording in the camera viewer first. The local snapshot and saved manifest
declare the layout; the native live/offline decoder uses that actual layout.
`CLOCK_RAW_DIAGNOSTIC` lines in the camera viewer's log expose witness support,
proposal score and checked-code results. This is a diagnostic recipe, not a
claim that optical lock has been validated on a person. At a slower code rate,
an arrival-minus-first-code-submit value includes the unknown position within
the held code interval; it is not an exact sensor exposure latency.

For a recoverable camera run, press `S` (or the legacy `H` alias) to begin the
lossless RAW recording, press `Z` for the stimulus, then stop the stimulus and
press `S` again after the viewer returns. Each bundle includes a
`predictions.jsonl` trace keyed by ROI id, sensor sequence, and sensor
timestamp; unclassified or unanalysed ROIs remain in the trace instead of
being discarded.

`S` bundles also include `metadata.oim1`: framed target/presentation events,
unclamped predicted 2D gaze, the actual drawn cursor, source clocks, and
change-only monitor/per-eye scene metadata. This uses the existing typed framing,
not a new JSONL stream or gzip. Unknown metric camera/eye poses remain null.
Native global and sensor-band thumbnails are retained without re-encoding
in `thumbnails.oic1`, indexed by `thumbnails.jsonl`. Host submission times are
kept separate from sensor exposure clocks. See [RAW recording evidence](docs/raw-recording-evidence.md)
for coordinates, snapshot/held-state semantics and the socket-stream format.

The matching lossless RAW decoder searches native packed-RAW recordings for
the repeated chromatic lattice without using desktop captures, resized
previews, or demosaiced pixels. `--host-phase-prior` lets packet time bound the
absolute counter family; optical evidence still selects the phase and solely
determines geometry, rate, fractional phase, and acceptance:

```bash
cargo run --release --bin buttercup-screen-reflection-raw-decode -- \
  --bundle outputs/raw-eye-hotkey/RECORDING.tar \
  --manifest outputs/screen-reflection-calibration/SESSION.jsonl \
  --whole-roi-clock \
  --host-phase-prior \
  --output outputs/screen-reflection-calibration/RECOVERED.jsonl
```
# Main-screen workspaces

Click the **Preview / Linked Views / Overview** tabs, or press **Tab**. **F** cycles
views within the current workspace; browsing never starts object acquisition.

The always-visible **G detector** bar names the global analysis mode, including
**EYE-STUDENT (3/6)**, and can be clicked to cycle it. The detector is shared by
all enabled ROIs; selecting an ROI changes its presentation, not its detector.
Each ROI card repeats the selected detector and identifies its displayed frame's
mode separately while a switch is pending. The **F view** heading below the bar
names only the visualization, not the analysis method. Focus follows eyes,
desktop pointer output, the J cursor, calibration and accuracy measurement all
use the global analysis settings. Old-method/settings frames cannot drive these
outputs while a change is pending.

**Shift+Tab** switches between editing **global preview defaults** and
**selected-preview overrides**. **F/V** edit the overlay/pixel appearance in
that scope. **Backspace** restores the selected preview's inherited defaults.
Overrides are per setting: a local F choice does not freeze inherited V.
G and solver settings stay global regardless of this presentation edit scope.

- **ROI:** one large region with global context underneath. **1** selects
  subject-left, **2** subject-right. Each inherits the global **V** pixel view
  and **F** overlay unless explicitly overridden. Selection does not change the
  physical autofocus or gaze reference.
- **Linked ROIs:** compare both retained ROI views, inspect separate source
  timings, or compare contacts. Missing/disabled ROIs are labelled explicitly;
  a paired layout alone is not proof of an admitted joint stereo solution.
  **V** still follows the chosen presentation edit scope.
- **Overview:** full sensor snapshot or prompted object search with an object crop.

The inspector has **View / Model / Cam / More** tabs (comma **,** cycles them).
View controls, shared analysis, physical whole-camera controls, and diagnostics
are separate. **PgUp/PgDn** or the wheel scrolls long panels. On smaller windows
the inspector moves below the images. **F2** explicitly sets the selected ROI
as the camera autofocus reference. Existing **J**, **M**, and lightbox controls
remain available; calibration still has its distraction-free screen.

For varied calibration lighting, **B** enables the frame, initially **SOLID
WHITE**, and each **N** press advances one pattern. Press N twice from white
for a repeating four-second auto cycle of colored solids, pulses, colored
checkerboards, and horizontal/diagonal color sweeps. Further N presses select
individual patterns; **[ / ]** adjust thickness. These controls also work during
accuracy checks, where the frame is capped at 8% to preserve the targets.
See [lighting controls](docs/viewer-workspaces.md) for pattern timing and recording metadata.

The shared light frame has an optical **CLOCK / 5 HZ** mode: **B** enables the
border and **N** cycles `SOLID WHITE → CLOCK → AUTO CYCLE → …`.
It works in normal viewing, **M** calibration, accuracy checks, and the
**Shift+\\** small-pixel-movement recording. **[ / ]** adjust border width.
Clock stays selected continuously (it is excluded from the four-second auto
rotation); allow about 13 seconds for its checked temporal word. The indicator
shows warming/searching/checked, not a guessed exposure latency. Existing RAW
recordings include the emitted symbols, actual host submission bounds, and
latest checked recovery in `metadata.oim1`, alongside targets and gaze. This
does not launch a separate window. See [clock validation and limitations](docs/optical-clock-debugging.md#shared-presentation-border-clock).

The **J** cursor and post-calibration cursor use absolute placement: each new
gaze target is displayed immediately, without cursor or gaze-direction easing.
Temporal sign validation and scale/geometry admission remain intact; SAM inference
latency still applies, and unsmoothed gaze can show more measurement jitter.
The SAM tweaked-contact F view remains an experimental preview, not an implicit
switch of the global gaze geometry. See [settings and output scope](docs/viewer-workspaces.md#global-gaze-settings-versus-preview-settings).

The local control socket supports `VIEW STATUS`, `VIEW ROI|LINKED|GLOBAL`,
`VIEW LEFT|RIGHT`, and `VIEW NEXT`. `VIEW PROMPT text` applies a prompt to the
current ROI/global-object context; `VIEW SEARCH` explicitly starts/stops object
search (start is only valid in its Global view). `VIEW STATUS` reports the current scope,
view, selected ROI, autofocus reference, both prompts, and object-search state.

Read-only presence/gaze cooperation must preserve the experimental camera/ROI
workflow. See [the cooperation boundary](docs/presence-cooperation.md) for source
freshness, unknown-versus-absent semantics, and the pending peer integration.

# Optional second-eye analysis

Press **3** to toggle subject-left (second ROI) analysis. It starts off; subject-right
remains enabled. Both enabled eyes may submit SAM requests with separate source
histories. Each eye has its own lazily loaded SAM worker and private CUDA stream;
the eyes process concurrently, but each eye's video memory advances in source order.
There is no queued frame backlog. Prompt reloads bind both workers to the same
revision while preserving the generation of any in-flight result.
This is not yet a joint stereo solve: the joint-conic and binocular coordinator
interfaces remain unimplemented. Enabling the second ROI can increase inference
contention; physical sensor-band eviction still takes precedence.
The local control socket also supports `SECOND ROI ON|OFF|STATUS`.
See [concurrency and gaze latency](docs/sam-concurrency-and-gaze-latency.md)
for architecture, runtime requirements and the matched RAW timing/geometry check.

In SAM mode, **Global → F → Prompted Object Search** is a generic object
inspector: **Enter** edits its independent object prompt, including objects such as
hat, mouth or ear. It exclusively captures a fresh full-sensor presentation,
runs the prompt through the primary SAM worker, and shows the selected
mask plus a 10%-padded object crop. The global view is always visible; a miss
clears the old crop and retries after a two-second cooldown. Scores >=0.5 and
nontrivial mask support are heuristic acceptance gates, not calibrated confidence.
There are no iris shape/size, eye-side or darkness gates in this path.

The object crop is a **software crop of the global capture**, not a newly moved
high-resolution physical sensor ROI. Global inference currently resizes the
presentation to 384x256, so small objects may be missed. This explicit inspection
mode captures independently of the eye-specific R recovery switch; it pauses
fine-eye analysis and cannot supply eye presence, gaze or mouse calibration.
After applying a prompt, **Space** explicitly starts/stops search. Search remains
active when browsing another workspace; Space can stop it there too. Stopping
resumes eye revalidation. Object prompts no longer overwrite the iris prompt.
Normal eye recovery also publishes each global thumbnail before MediaPipe runs,
including detection failures.

In SAM mode, **F → CONIC SEGMENTS** shows the de-flat-tire points and colored
contiguous support arcs for each available ROI, including single-eye operation.
Green dots are retained support, magenta dots are rejected, and colored lines
identify runs which never cross rejected contour samples. These are current
outer-limbus fit supports, not independently solved pupil/inner-limbus arcs or
an implemented joint stereo solution. Each tile retains its own SAM source image
and sequence/lag annotation; two visible answers need not share an exposure.

After successful **M** calibration, the result screen shows the estimated monitor
wireframe, pitch/yaw/roll, physical gaze intersection and eye-to-center/hit distances.
Yellow physical gaze and the white affine cursor are distinct. Screen size comes
from the selected monitor's EDID when readable, otherwise the labeled nominal
27-inch fallback. Angles are eye/camera-relative, not gravity-referenced.
The reconstruction viewpoint slowly oscillates ±45° horizontally (24-second
cycle) and ±10° vertically (32-second cycle). This is presentation-only: the
monitor fit, gaze hit, distances and printed pose angles do not rotate with it.

Leaving an accepted calibration with **M** turns on a small yellow **gaze
ring** that follows the calibrated screen gaze. It passes clicks through, never
takes focus and never moves the pointer; `scripts/toggle-mouse-output.py --cursor`
(or `GAZE CURSOR ON|OFF|TOGGLE|STATUS` on the control socket) toggles it.
While a calibration target or the orientation plus is shown, the screen shows
only the stimulus; status text appears once the attempt is accepted or fails.

Camera controls are also available outside the viewer through its control
socket (`EXPOSURE STEP n|AUTO|MANUAL|STATUS`, `FOCUS SET n|AUTO`, each change under
a short lease). `scripts/camera-exposure.py up|down|toggle` and
`scripts/camera-focus.py in|out|auto` wrap them for desktop shortcuts; Rob's Sway
bindings are Super+] / [ exposure, Super+Shift+] / [ focus, Super+\\ autofocus and
Super+Shift+\\ auto-exposure toggle. Each change shows `scripts/eye-focus-osd.py`,
a centered, click-through overview styled like the attention-manager surface: the
changed value with a gauge, then camera (AF/AE, lens, exposure), eye-tracking
(model, stereo, mount, calibration, measured lens, eye) and output (gaze ring,
mouse) status. It starts on demand or from the Sway session.

The stereo selection (Shift+3) is saved in `outputs/settings/stereo-solver.json`.
A measured lens pinhole in `outputs/settings/joint-camera-intrinsics.json`
(`[fx, fy, cx, cy]` in native sensor pixels, e.g. from the checkerboard collector)
replaces the 4000 px engineering default; `BUTTERCUP_JOINT_CAMERA_INTRINSICS`
still overrides it. The checkerboard square size is set with
`BUTTERCUP_CHECKERBOARD_SQUARE_MM` (measure the inner six squares and divide by 6).

A completed calibration is saved automatically to
`outputs/settings/gaze-calibration.json` and reloaded at startup. With stereo on,
routine tracking re-establishes eye direction from ordinary on-screen viewing
(the same screen half-space and coherent vote as the orientation plus), so a
reloaded calibration becomes usable without pressing Shift+M. A separate
loadable `*.gaze-calibration.json` is kept beside each calibration recording.
Tracker resets, source/sign epochs and application restarts do not discard the
screen mapping. Fresh signed gaze from the matching eye and detector resumes
the same fit; current prompt/source matching remains enforced by the shared gaze
pipeline. Missing or unsigned gaze pauses output without deleting calibration.
If the direction reference is lost, **Shift+M** reorients using the existing map;
**M** performs a replacement nine-point calibration. A real sign reversal can
make the cursor jump. Training/current epochs remain recorded for diagnosis;
unfinished calibration still restarts its sample collection on a sign change.
Near-frontal correspondence now distinguishes a continued crossing from a
turnaround using source-timed motion and bounded pivot support. See the
[meridian regression tests and native-video review](docs/meridian-sign-continuity.md)
for the measured results and remaining limits.

If **M** starts with an unresolved surface direction, follow the small moving
plus first. This recorded acquisition phase stays within the middle 20% of the
screen and waits for fresh, sustained sign evidence before the usual stationary
targets. It times out explicitly after 20 seconds; it does not guess a sign or
train calibration on the moving target. Already usable gaze skips that phase.
See [the acquisition investigation and live-retest notes](docs/sign-acquisition-trials.md#live-retest-follow-up).
