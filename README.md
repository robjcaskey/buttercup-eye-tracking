# Buttercup eye tracking

A host-side Rust eye tracker and Wayland viewer. The project consumes lossless
RAW eye frames from an already-running camera service over TCP. It does not
contain, install, start, or modify camera firmware, kernel modules, USB/UVC
drivers, sensor code, or a proprietary camera application.

## Build

```bash
cargo build --release
```

The default build includes checkerboard camera-intrinsics calibration. The
optional SAM 3.1 adapter is source-only and requires an externally supplied
model/runtime:

```bash
cargo build --release --features sam31
```

## Run

With a compatible camera service already available:

```bash
scripts/run-viewer.sh
```

Addresses and initial sensor coordinates are supplied through `BUTTERCUP_*`
environment variables documented in the launcher. No camera provider is
started by this repository.

Runtime output is stored through `data` and `outputs`, both of which resolve
to the separate `/mnt/bulk_data/buttercup-eye-tracking` data root. Run
`scripts/audit-tree.sh` to verify the source-tree boundary and file allowlist.
