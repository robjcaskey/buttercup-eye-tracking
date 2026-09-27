# Frame-pair diagnosis and CPU limbus training

`buttercup_report_limbus_refiner pairs` examines neighboring native exposures
from a source-matched sequence replay. It reopens the archive RAW, verifies
frame identities, checks the recorded target/clock evidence and measures image
motion independently of the fitted ellipse. The strict diagnostic limits are
consecutive exposures at most 150 ms apart, at least 10 matching RAW patches,
translation and residual at most one native pixel, one target discrepancy at
most 1 degree and the other at least 4 degrees across the declared clock/lag
range. Failed registration with a zero compatibility transport is unavailable
motion, never a measurement of stillness.

The output includes `pairs.csv`, `error-jump-candidates.csv` and
`all-neighbor-pairs.csv`. The latter two retain motion failures and larger
movements instead of concealing them or calling them stationary. Every row
includes source keys and RAW hashes. A commanded target under a fitted monitor
pose is a conditional reference, not an independent observation of fixation.
These files are diagnostic, with `training_eligible=false`.

```sh
cargo build --profile live --no-default-features --bin buttercup_report_limbus_refiner --bin buttercup_limbus_cpu

data/target/live/buttercup_report_limbus_refiner pairs outputs/NEW_PAIRS \
  REPLAY.json CAPTURE/metadata.oim1 CLOCK.json BLINK_REVIEW.json

data/target/live/buttercup_report_limbus_refiner calibration-pairs outputs/NEW_M_PAIRS \
  SESSION_1.session.json SESSION_2.session.json
```

The completed-calibration command requires an accepted, completed M session
with the historical projected-direction affine. It uses the calibrated eye's
exact analysis source, rejects held/unsigned geometry and other sign epochs,
and rechecks target windows, settling, successful hidden-thumbnail submission
and visible target identity. Its saved same-session mapping is an in-sample
diagnostic, not held-out accuracy. Host arrival and buffer submission do not
measure exposure/scanout latency. An unfamiliar mapping input space is refused.

## Cold CPU experiment

```sh
data/target/live/buttercup_limbus_cpu cold \
  CANONICAL_LABEL_INVENTORY.json outputs/NEW_COLD_RUN 40
```

An inventory can contain `labels: [PATH, ...]`, or be an older canonical limbus
patch dataset used **only as an inventory of human label paths**. Current Rust
preparation reopens every original annotation and RAW payload. Cached points,
fits, partitions, teacher material and calibration are not reused. Evaluation-
only datasets are refused.

The recipe trains the existing portable normal-offset network from seeded
initialization using canonical human inner/outer band vectors to orient native
RAW patches. A local circle carries that vector to the shared extractor; its
size and bearing inputs have permanently zero weights. It does not turn sparse
labels into a completed human ellipse. Missing band orientation is excluded;
missing landmark roles remain unsupervised. No SAM or custom-model ancestor is
required for this particular component experiment.

Plain training and fixed Quad-Bayer channel-gain augmentation use identical
initialization. Four conservative recording groups alternate between train,
validation and test. Only validation loss chooses fold checkpoints. Full
development exports use the upper median of validation-selected epochs;
the held-out errors do not choose that schedule. The input budget is 4,096
patches and the training budget is 120 seconds per model. Training and inference
use native Rust CPU operations without LibTorch or CUDA. Existing visibility,
spread, contrast, clipping, four-pixel correction and whole-shape limits still
apply in the shared refinement stage.

Preflight and final dependency graphs, source stamps, verified roots, feature
hash, immutable model hashes, fold metrics and timing are written alongside the
models. A structural certificate is not a quality certificate. The development
comparison requires at least 5% lower pooled gated rim-offset error, no group
regression above 0.25 px and at least 80% of control correction coverage. The
report never deploys a model. Follow `bootstrapability.md` before reusing a
checkpoint as a teacher or promoting it.

The initial experiment uses 16 reviewed RAW frames of Rob in four conservative
sensor-time groups. Only 87 observed bands supply orientation; some patches
fall outside RAW. Recorded clock identity and independent physical scale are
missing on these annotations. Normal-offset improvement cannot establish
signed gaze, full-contour localization, SN-FEIDA stability, or new-user
robustness. Those require matched full-worker replays, visual inspection and
additional independent evidence. In particular, a four-pixel patch correction
cannot recover an entire unsupported rim that displaced a fit by tens of pixels.
