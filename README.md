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

For a recoverable camera run, press `H` to begin the lossless RAW recording,
press `Z` for the stimulus, then stop the stimulus and press `H` again after
the viewer returns.

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
