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
