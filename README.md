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
