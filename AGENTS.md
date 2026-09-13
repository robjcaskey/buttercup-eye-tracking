# Repository boundary

## Mandatory camera cooperation

Rob requires assertion-style process-fatal enforcement of camera cooperation.
All capture and control connections must use `camera_cooperation::connect_timeout`
after `camera_cooperation::start`. Do not add a direct TCP fallback or catch a
missing ownership invariant as a recoverable camera error. Missing initialization,
changed endpoints/attachment, replaced retained lock, and invalid live ownership
abort the entire process, including other camera workers. An existing but stale
or unavailable UPC runtime refuses startup; it never authorizes standalone use.
The runtime watcher also aborts a standalone session if UPC appears later.
Keep the socket-boundary regression test and subprocess abort tests effective.
The authorized local coordination file is `/tmp/o.xt`, not `/tmp/o.txt`.

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

# Training bootstrapability

- Read and follow [bootstrapability.md](bootstrapability.md) before training,
  reusing model-derived training material, or promoting a custom model.
- Every custom model ancestor must be provably regenerable from native RAW,
  canonical human labels and explicitly pinned SAM3 assets using the current
  checkout. SAM3 is an allowed external bootstrap dependency; unknown custom
  checkpoints, pseudo-label caches and learned feature dependencies are not.
- Preserve the distinction between single-user results, cold bootstrap and
  new-user personalization. Do not claim cross-user readiness from Rob-only
  recordings or silently reuse his personal calibration for another user.
- Run the native bootstrap DAG preflight before training/reusing derived
  training material; a structural certificate alone is not a cold-run proof.
- Maintain training/preparation/reporting recipes in Rust. Shared offline
  training may use CUDA; user/scenario last-mile training and inference MUST
  be CPU-only, with isolated provenance and bounded resources.
- Front-load shared robustness. New users and abrupt lighting/screen-color
  changes must not depend on dynamic retraining to obtain useful tracking.

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
