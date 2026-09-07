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

Buttercup expects a compatible external RAW camera service over TCP. Camera
firmware and device-side control live elsewhere.

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
shows a smoothly moving fixation target over a locally balanced chromatic
frame code and writes a presentation manifest under
`outputs/screen-reflection-calibration/`. Press `Z`, `Esc`, or `Q` in the
stimulus to return to the viewer.

While the clock is running, its upper-left readout reports optical recovery
as `WARMING`, `SEARCHING`, `CHECKED SINGLE FRAME`, or `LOCKED`. Every V3 symbol
is a session-keyed RM(1,4) `[16,5,8]` word carried by complementary cells and
four spatial copies. One frame can therefore correct as many as three wrong
logical symbols; an ambiguous frame is rejected instead of being published.
The first checked frame immediately shows its individual lag. Agreement over
time upgrades the status to `LOCKED` and adds the robust median. The interval
runs from the host's Wayland display commit to arrival of the camera packet
carrying that recovered code, so it includes display scan-out, exposure,
camera transport, and packet delivery rather than claiming to be sensor
exposure latency alone. Rendering is paced by Wayland frame callbacks rather
than a drifting userspace timer.

For a recoverable camera run, press `S` (or the legacy `H` alias) to begin the
lossless RAW recording, press `Z` for the stimulus, then stop the stimulus and
press `S` again after the viewer returns. Each bundle includes a
`predictions.jsonl` trace keyed by ROI id, sensor sequence, and sensor
timestamp; unclassified or unanalysed ROIs remain in the trace instead of
being discarded.

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

Click the **ROI / Linked ROIs / Global** tabs, or press **Tab**. **F** cycles
views within the current workspace; browsing never starts object acquisition.

- **ROI:** one large region with global context underneath. **1** selects
  subject-left, **2** subject-right. Each retains its own **V** pixel view and
  **F** overlay. Selection does not change the physical autofocus reference.
- **Linked ROIs:** compare both retained ROI views, inspect separate source
  timings, or compare contacts. Missing/disabled ROIs are labelled explicitly;
  paired presentation is not an implemented joint stereo solve. **V** here
  deliberately changes both ROI pixel views together.
- **Global:** full sensor snapshot or prompted object search with an object crop.

The inspector has **View / Model / Cam / More** tabs (comma **,** cycles them).
View controls, shared analysis, physical whole-camera controls, and diagnostics
are separate. **PgUp/PgDn** or the wheel scrolls long panels. On smaller windows
the inspector moves below the images. **F2** explicitly sets the selected ROI
as the camera autofocus reference. Existing **J**, **M**, and lightbox controls
remain available; calibration still has its distraction-free screen.

The **J** cursor and post-calibration cursor use absolute placement: each new
gaze target is displayed immediately, without cursor or gaze-direction easing.
Temporal sign validation and scale/geometry admission remain intact; SAM inference
latency still applies, and unsmoothed gaze can show more measurement jitter.

The local control socket supports `VIEW STATUS`, `VIEW ROI|LINKED|GLOBAL`,
`VIEW LEFT|RIGHT`, and `VIEW NEXT`. `VIEW PROMPT text` applies a prompt to the
current ROI/global-object context; `VIEW SEARCH` explicitly starts/stops object
search (start is only valid in its Global view). `VIEW STATUS` reports the current scope,
view, selected ROI, autofocus reference, both prompts, and object-search state.

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
