# Buttercup

Buttercup is an experimental Rust viewer and analysis workspace for lossless
RAW eye-camera data.

![Buttercup viewer with sensor overview and eye ROIs](docs/viewer-overview.png)

It combines a coarse sensor view with native-resolution eye regions and makes
its motion, segmentation, and projected-geometry state visible. The approach
could be useful for low-latency gaze input, calibration, and related
camera-space interaction work, but it is still a research prototype.

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

When Keyboard Peeper is running, the viewer optionally publishes its current
hotkeys over the versioned binary KPP/1 Unix-socket protocol. Mode-dependent
buttons are retained in the map with an enabled or disabled state, and updates
are atomic. The registration clears as soon as the viewer loses focus. No
peeper installation or library is required; set
`BUTTERCUP_KEYBOARD_PEEPER=0` to disable the silent background publisher.

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
