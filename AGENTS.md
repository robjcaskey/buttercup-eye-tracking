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

# Geometry development validation

- Use **scale-normalized frontal-equivalent iris disk area (SN-FEIDA)** as a
  north-star diagnostic for outer-limbus consistency. Its definition and
  limitations live in `docs/flat-tire-area-and-motion.md`; it is not visible mask
  area, pupil aperture area, or a measured curved anatomical surface area.
- When developing or testing geometry theories, favor matched baseline/candidate
  corpus evaluation alongside synthetic/unit tests. Run independent evaluations
  in parallel when practical, then inspect and explain the results, including
  regressions; merely launching a replay is not validation.
- Pair area stability with human-label localization error, coverage/dropouts,
  source-time/motion alignment and independent scale support. Never normalize by
  the candidate's own radius, count held predictions as fresh observations, or
  reward a frozen/wrong ellipse merely because its area is constant.
- State the actual corpus subset, missing labels/scale/timing, uncertainty
  assumptions and remaining failures. Treat bounded support and heuristic
  fidelity as defeasible estimates, not calibrated probabilities.
