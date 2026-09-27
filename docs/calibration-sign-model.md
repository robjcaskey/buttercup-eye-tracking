# Two-frame calibration sign experiment

`buttercup_calibration_sign` trains a small CPU model from native RAW and
recorded screen-target presentations. It reads every session inventory in the
specified calibration corpus. Each input is the last two consecutive fresh
frames of one eye, blurred and reduced to 32 by 24 grayscale pixels. No
segmentation, conic, saved gaze calibration, old sign decision or custom model
is a neural input. The original `cold` mode uses target supervision only;
`cold-branches` additionally generates conditional sign labels from fresh
classical RAW geometry, without reading any recorded prediction.

```sh
cargo build --profile live --no-default-features --bin buttercup_calibration_sign
data/target/live/buttercup_calibration_sign inventory outputs/calibration-corpus outputs/NEW_INVENTORY
data/target/live/buttercup_calibration_sign cold outputs/calibration-corpus outputs/NEW_COLD_RUN
data/target/live/buttercup_calibration_sign native-conics outputs/calibration-corpus outputs/NEW_RAW_PROBE
data/target/live/buttercup_calibration_sign branch-labels outputs/calibration-corpus outputs/NEW_LABEL_AUDIT
data/target/live/buttercup_calibration_sign cold-branches outputs/calibration-corpus outputs/NEW_BRANCH_RUN
data/target/live/buttercup_calibration_sign review outputs/NEW_COLD_RUN
data/target/live/buttercup_calibration_sign infer outputs/NEW_COLD_RUN/model.json PAIR.json outputs/NEW_PREDICTION.json
data/target/live/buttercup_calibration_sign movie outputs/calibration-corpus outputs/NEW_COLD_RUN outputs/NEW_MOVIE
```

Use the host resource coordinator around those commands. Outputs must live
under the checked runtime links. The cold command creates a fresh directory,
reopens and hashes native RAW, regenerates its inputs, runs the native bootstrap
DAG preflight, trains from seeded random initialization, evaluates held-out
recording days and writes a loadable JSON artifact with provenance. No installed
checkpoint or derived training cache is read. It refuses a changed source tree
at the final provenance check. CPU inference has no LibTorch/CUDA dependency.

Target admission uses successful presentation submissions, not the old
solver's accepted samples. Both exposures must occur after two seconds of
acclimation plus 100 ms allowance and after thumbnails have been hidden. A
100 ms guard precedes a target change. Pairing requires consecutive native
source identities at most 250 ms apart. Missing timing, missing archives,
unsupported images and conflicting duplicate labels are reported explicitly.
Failed calibration attempts remain eligible when their target records qualify.

The model learns nine intended screen locations, with horizontal and vertical
signed direction scores obtained by summing their columns and rows. A recorded
target is **weak supervision**, not proof of fixation on every image. Bounded
generalized cross entropy reduces the influence of disagreeing examples; it
does not identify distractions or turn uncertain labels into ground truth.
No timestamp, target index, session identity or elapsed time reaches the model.

All eyes, neighboring frames and sessions from one recording day stay in the
same test partition. Validation uses entire separate viewer sessions on other
days. Only validation selects checkpoints. A matched current-frame-only arm
checks whether the preceding image actually adds information. The final model
uses all eligible pairs and the median validation-selected epoch count.
Softmax scores are uncalibrated support. Results are Rob-only development
measurements; neither new-user transfer nor lighting robustness is established.
The corpus has informed method development, so this is development cross
validation with disjoint model fitting, not an untouched final benchmark.

`calibration_sign_model::Prediction::choose_projected_branches` can compare two
conic directions **after both have been projected into the screen using the
same independently valid camera/display mapping**. It abstains when that
mapping is unavailable, the candidates converge, or image support is weak.
Screen-target accuracy alone is not camera-relative conic-sign accuracy. This
experiment does not borrow a fitted legacy monitor pose to manufacture sign
labels, and its report leaves independently measured 3D sign accuracy null.

The optional binary head directly scores the lower/higher camera-normal-X
circle solutions. Its teacher fits RAW gradients with deterministic, bounded
RANSAC, using gradient tangents as well as positions. Upper-rim scoring is
censored to avoid treating an eyelid as an iris boundary; left, bottom and
right evidence is required. A subsequent RAW contrast check rejects unsupported
completions. These gates do not establish human-label localization accuracy.
Measured samples and fitted curves remain separately visible.

The teacher assumes nominal focal lengths of 4000 pixels, principal point
(4000,3000), stable approximately upright camera/display placement, and positive
screen-angle scales with roll sampled between -15 and +15 degrees. It searches
branch assignments across at least four supported target positions, including
both horizontal and vertical extremes, separately per recording/eye/stream.
All near-best maps must agree with a separation margin before an example gets
a branch label. This retrospective finite-grid teacher is neither a continuous
geometric certificate nor measured sign truth. Off-target fixation, wrong RAW
fits, missing intrinsic/extrinsic measurements and camera movement can invalidate
its labels. Missing labels never remove images from target supervision.

Training uses the same shared hidden layer, an additional class-balanced bounded
branch loss, and validation branch cross entropy when branch labels exist.
Folds with fewer than ten training labels in either branch omit the binary
head. Reports include label coverage, class balance, held-out teacher agreement,
current-only and training-majority controls. Inference requires neither target
coordinates nor a monitor fit. It abstains without supported fresh geometry,
when camera-X branch ordering converges, or below 80% uncalibrated model support.
That threshold is not an 80% correctness guarantee.

The recipe emits source-matched contact sheets, per-frame held-out predictions,
confusion matrices, coverage at a fixed support threshold, per-session results,
model hashes and reload parity. It changes no ellipse or measured boundary, so
SN-FEIDA and human-label localization are unchanged rather than improved by
this experiment. The model is experimental and is not automatically enabled
in the viewer.

The movie command replays every native exposure in each qualifying recording,
including settling periods and unscored frames. It loads the model that held
out the recording's entire day, verifies model and archive hashes, and checks
every scored prediction against the saved evaluation. It shows RAW, previous
and current blurred inputs, their change image, both eyes' screen predictions,
the single-frame baseline, target-score grids, recent traces and both physical
conic hypotheses when source-matched geometry is available. Target-only runs use
archived fits and segment samples as retrospective diagnostics only: they never
enter training, sample selection or neural inference. Missing fits remain missing.
Branch runs use freshly regenerated classical or SAM geometry and show the learned
choice, abstention, and conditional teacher agreement separately. Every saved
binary score and comparison count is checked against held-out evaluation.

The renderer uses system Cairo and FFmpeg, with portable CPU model inference.
A fifth argument limits rendering to a matching archive and its first twelve
source seconds for layout inspection. The full movie has per-recording
chapters and an event manifest containing exact RAW identities, predictions,
source times and encoded-frame intervals. Long source gaps receive explicit
slates; held images are not scored again. The cold-run source stamp must match
the current checkout, so regenerate the cold run after changing the recipe.

## Regenerable SAM teacher inspection

`sam-export CHECKPOINT NEW_OUT` verifies the original SAM3.1 multiplex checkpoint
SHA-256 and upstream revisions, invokes a pinned upstream PyTorch adapter, and
writes a standalone detector plus export/reload parity and bootstrap receipts.
The Rust recipe owns dependency pins, prompts, hashing and provenance. Its small
embedded adapter only calls the external SAM/PyTorch export API; RAW decoding,
selection, mask processing, geometry and reports remain native Rust.

The detector uses a BF16 image backbone with FP32 semantic encoding, iterative
box decoding and masks. Explicit normalized BF16 image inputs avoid amplified
rounding differences across the serialization boundary. Empty ROIAlign results
for fixed text-only prompts are folded exactly, so native inference needs no
TorchVision extension or external SAM service. Export admission keeps the
predeclared maximum absolute error limits of 0.01 for scores and 0.15 for mask
logits. Seeded-input parity is an arithmetic check, not a mask-quality test.

`sam-native CORPUS EXPORT NEW_OUT [ARCHIVE]` requires the `sam31` feature and
the existing LibTorch CUDA runtime. It hashes the export, checks the recipe,
uses the existing Quad-Bayer RGB adapter, and shows up to twelve matched RAW
exposures per recording. The temporary five-copy strip is solely a way to call
that existing single-image color conversion; duplicate copies are not temporal
observations. In that diagnostic, independent ROIs form a 3-by-4 inference mosaic. Its exact input,
unfiltered top mask and proposal scores are retained for inspection.

Mask admission requires score at least 0.15, at least 64 positive pixels at
the SAM mask resolution, and at least 80% of that instance's mask within one
tile. The selected mask uses the same contour fitter as the main SAM viewer;
native coordinates, conic construction and RAW left/bottom/right contrast
checks are shared with the classical diagnostic. Mask contour samples are
predictions, not human annotations. A score is not a calibrated probability.

On recording `1789432526`, the initial long iris-description prompt produced
zero admitted fits on twelve sampled exposures; its highest mask score was
about 0.026 despite an iris-shaped proposal. The short `iris` prompt produced
three fits passing the unchanged gates. RAW overlays were inspected, including
eye 1 sequence 956, where the classical fit follows an eyelid edge. The new
fits improve that visible defect but do not establish sign accuracy, broader
coverage, human-label localization or scale-normalized area consistency.

The successful short-prompt export receipt is under
`outputs/calibration-sign-sam-export-20260918-v25`; the matched RAW inspection is
under `outputs/calibration-sign-sam-raw-probe-20260918-v5`. Earlier failed export
checks remain failed; their tolerances were not widened.

`sam-native-single` keeps the same diagnostic selection and thresholds but
feeds one ROI to the detector at a time. On those same twelve exposures, all
twelve fits passed the RAW rim gates, compared with three using the mosaic.
The source-matched sheet is in
`outputs/calibration-sign-sam-single-probe-20260918`. This small within-recording
comparison motivated single-ROI teacher generation; it is not a corpus accuracy
or independent localization result.

`cold-sam-branches CORPUS CHECKPOINT NEW_OUT` adds the pinned SAM root and
export to the bootstrap DAG, checks the planned graph before teacher generation,
regenerates the export in the new run, and infers every unique RAW image
contributing an eligible pair. It uses the shared SAM contour fitter and native
RAW rim gates above. It then constructs the conditional labels and runs the
same held-out-day two-frame/current-only comparison. The geometry, mask hashes,
class coverage, RAW identities and export receipt accompany the model. Adjacent
RAW overlays with contour samples and both nominal normals are written during
preparation. Only the shared SAM teacher uses CUDA; custom model optimization
and inference remain native CPU operations.

`sam-branch-labels CORPUS CHECKPOINT NEW_OUT [ARCHIVE]` exercises the same cold
teacher and label preparation without training, optionally on a single recording.
It never accepts an existing teacher cache as a cold input. The label audit
remains a conditional feasibility check even when every structural check passes.

For a SAM run, movie geometry is available only on the freshly prepared RAW
exposures. All other exposures remain visible, with no conic or geometric sign
selection. There is no silent fallback to classical or archived geometry.
Geometry keys include RAW content and crop/decoding metadata. Portable `infer`
still computes branch scores using only the two images; for an actual geometric
branch choice, its optional last argument accepts a source-matched row from
`native-conics.jsonl`. Without it, SAM-trained inference explicitly abstains
from choosing a physical conic. This experiment does not enable a live viewer
mode automatically.

## Short movie with both projected candidates

`candidate-movie TRAIN_RUN EXISTING_MOVIE_DIR NEW_OUT` makes a roughly one-minute
presentation from an existing completed movie and its event manifest. It shows
the two RAW eyes and four cursors: cyan for eye 1, pink for eye 2, filled for
candidate A and outlined for B. Both original conic normals pass through the
same saved per-eye/source-epoch calibration map, chosen once by lowest saved
RMSE. The original temporally associated A/B order is retained. This is a
retrospective calibration projection, not two independent neural predictions
or an independent sign-accuracy evaluation. Missing maps or geometry are never
invented.

The main view pads the display by 20% of its width/height on each side. Predictions
within `[-0.2,1.2]` remain at their actual positions; farther predictions stop at
the padded boundary along their direction from the screen center. Their labels
give total shortest distance beyond the actual display in screen heights `H`,
including the margin: `hypot((u-clamp(u,0,1))*16/9, v-clamp(v,0,1))`.
Numerical coordinates and a fixed-scale overview retain the actual endpoints.
Screen heights avoid implying measured physical size or viewing distance. Both
screen diagrams preserve 16:9. Excerpts require continuously available admissible geometry for
both eyes, span up to three recordings and several targets, and play three
times slower. Selection uses geometry availability, duration and source order,
not gaze error or model scores. Hard cuts identify discontinuous excerpts.

The command verifies the historical training/movie identities, source video
hash and native-conic hash, and matches every displayed normal to its original
RAW/crop identity (allowing the recorded A/B swap). It hashes the event manifest
and records exact original and output frame intervals and projected coordinates.
It neither executes nor trains a custom model. Historical training provenance
and the current presentation-source stamp remain separate; producing this
presentation does not re-certify or promote the older model on a new checkout.
