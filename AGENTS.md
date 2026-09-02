# Repository boundary

This repository owns host-side eye analysis and presentation only.

- Treat the camera as an external, versioned TCP service.
- Never add camera firmware, sensor/kernel modules, USB/UVC control code,
  proprietary camera binaries, extracted root filesystems, or vendor SDKs.
- Never add captures, recordings, model weights, compiled objects, archives,
  copied dependency trees, or opaque binary blobs.
- Keep runtime data under `/mnt/bulk_data/buttercup-eye-tracking` through the
  checked top-level `data`/`outputs` links.
- Add source files deliberately and update `scripts/audit-tree.sh` whenever the
  allowlist changes.
- Do not make this repository depend on paths inside another source checkout.
- Canonical human labeler: always use `/home/rob/eye-training/paired-limbus-annotator/server.py`; do not substitute another annotation UI.
- Feed it native RAW10 before/target/after frames and keep recorded predictions hidden until `SAVE + DONE`.
- Save paired, triplet, and possibly-occluded evidence beneath the capture's `annotator/labels` directory.
