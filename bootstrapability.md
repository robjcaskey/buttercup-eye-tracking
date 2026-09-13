# Bootstrapability contract

Buttercup must be rebuildable and retrainable from its native RAW corpus,
canonical human-labeled points, and explicitly pinned SAM3 bootstrap assets
using the **current checkout**. Existing custom weights and opaque intermediate
datasets must be disposable accelerators, never irreplaceable foundations.

This matters especially because the current training corpus contains one
person: Rob. Reproducing a successful model for Rob and quickly bootstrapping
for a new person are distinct requirements. Neither proves the other.

This is a normative development contract. `MUST` and `MUST NOT` govern future
training, model reuse and promotion. The native structural checker below is a
preflight, not a completed bootstrap executor or certification of existing
checkpoints.

## Permitted bootstrap roots

The dependency chain may terminate only in:

- Native RAW captures and genuine acquisition metadata: pixel format, stride,
  sensor/crop coordinates, exposure/source clocks, and available measured
  capture settings. A recorded prediction is not acquisition ground truth.
- Canonical, reviewed human point/band/visibility annotations tied to exact RAW
  identities, plus explicitly documented human measurements where available.
- **SAM3**, including explicitly pinned SAM3.1 assets used by this project.
  SAM3 is an intentionally allowed external learned dependency; reproducing
  SAM3's original foundation-model training is not required.
- Versioned source algorithms, declared non-learned priors, deterministic
  transformations, seeded random initialization, and pinned build/runtime
  dependencies. Installing CUDA or LibTorch is not an additional teacher.

The SAM3 exception MUST identify the exact weights, prompt/tokenizer assets,
tracker assets when used, upstream version, export/conversion recipe and
checksums. Record acquisition instructions and applicable access/license
requirements. An unversioned `latest` download or an old local exported graph
without a recoverable source is not a bootstrap recipe. Expected availability
does not remove the need to pin and verify dependencies.

Other learned models are not silently grandfathered in as teachers or feature
extractors. For example, recorded MediaPipe estimates MUST NOT become a hidden
required training root. Treat such hints as unavailable unless their dependency
is independently admissible under this contract. A live-only acquisition aid
is a separate dependency, not proof of training bootstrapability.

## No unprovable model ancestry

Before using a custom model as a teacher, initializer, feature extractor,
sample selector, confidence filter or pseudo-label generator, its full
dependency chain MUST have a successful bootstrap proof on the current source
version. A usable checkpoint file, a model manifest, a successful inference
replay, or a recipe that worked on a previous checkout is not enough.

This applies transitively to cached masks, embeddings, contours, fitted
ellipses, target coordinates, rankings and selected training examples. Saving
a model's output as a dataset does not erase its model dependency. Distillation,
self-training and warm starts are allowed only when every custom ancestor can
be regenerated from permitted roots, in dependency order, without a cycle.

Unknown provenance MUST stop downstream training and promotion. The remedy is
to regenerate the ancestor, replace it with a permitted bootstrap path, or
obtain an explicit change to this contract—not quietly reuse its predictions.
Existing unverified experiments may be retained and explicitly previewed for
comparison; that does not authorize training descendants or calling them
bootstrap-verified defaults. Do not delete existing evidence to satisfy a test.

## What counts as proof

A successful **cold bootstrap** MUST demonstrate the complete required path:

1. Start from a fresh runtime output directory with no existing custom model,
   optimizer state, teacher cache, feature cache or personal calibration made
   available as an input. Do not remove the user's deployed artifacts.
2. Verify allowed RAW, annotation and SAM3 inputs by content hash. A manifest
   must describe the actual subset and missing inputs; filenames alone are
   insufficient. Missing required roots fail explicitly.
3. Build and run the preparation, decoding, annotation interpretation, optional
   SAM3 export/pseudo-label generation, training, conversion and evaluation
   steps using code shipped in the current checkout. Essential glue cannot
   exist only in a developer's `/tmp`, an old output directory, shell history,
   another source checkout or an undocumented manual procedure.
4. Recreate custom ancestors before any descendant consumes them. Verify cache
   provenance when caches are used in ordinary development; the cold proof
   itself must regenerate them. No path may fall back to installed weights.
5. Produce a loadable inference artifact and run the agreed source-matched
   evaluation. Inspect the results, including regressions and abstentions;
   successful compilation, training completion or a constant ellipse is not
   successful bootstrap validation.

The proof receipt MUST include:

- Git revision and exact source-tree identity, including relevant uncommitted
  and untracked source changes; an unrecorded dirty tree is not reproducible.
- Hashed input manifests, subject/session/source-clock partitions, annotation
  schema, label counts, explicit excluded material and complete ancestor DAG.
- SAM3/runtime/toolchain versions, hardware, seeds, resolved hyperparameters,
  preprocessing, augmentation, feature/target definitions and checkpoint rule.
- Commands, logs, model/export hashes, missing-data behavior, numerical results,
  inspected failures, and the thresholds/tolerances declared before evaluation.
- Wall-clock and compute costs, including data preparation and SAM3 labeling,
  resource-coordination claims/conflicts and time to the first usable result.

GPU training need not be falsely advertised as bit-identical across machines.
Hashes identify the artifacts actually used; repeatability must meet declared
numerical and quality tolerances on stated hardware/runtime configurations.
No post-hoc widening of tolerances to turn a failed run into a passing claim.
Proof is tied to its exact evaluated version. After changing any preparation,
label, feature, training, model-loading or required evaluation path, rerun the
affected chain and its end-to-end checks before claiming current-checkout
bootstrapability. Historical receipts remain evidence about historical code.

## Sparse labels and honest targets

Human labels MUST retain their actual role, source and uncertainty. Use the
canonical labeler specified in `AGENTS.md`, native before/target/after frames,
and hide recorded predictions until `SAVE + DONE`. Do not reclassify assistant
guesses, SAM3 output, unreviewed points or fitted geometry as human labels.

Sparse limbus points do not label every pixel or establish a full hidden rim,
pupil aperture, signed 3D gaze or anatomical depth. Unknown, occluded and
out-of-crop regions MUST remain unknown/censored, not negative targets or
invented dense ground truth. Any rasterization or weak-target construction
must be reproducible and disclose its assumptions and supervision strength.
SAM3 pseudo-labels remain a separate, defeasible source even when they agree
with a geometric fit. Do not count correlated alternatives as new observations.

## New-user bootstrap and personalization

The supported workflow MUST distinguish:

- **Cold bootstrap:** allowed roots and fresh custom initialization; no existing
  personalized or generic Buttercup checkpoint is assumed.
- **New-user personalization:** optionally start from a proven bootstrappable
  shared checkpoint and collect a bounded amount of the new person's RAW and
  reviewed labels. That checkpoint is an optimization, not the only route.

A new person MUST NOT depend on Rob's learned eye shape, pupil size, IPD,
rotation-center estimate, focus/lighting setup, scale history or monitor/gaze
calibration. Separate shared weights from per-person state, expose which is
being used, and provide a fresh-user/reset path. Missing measurements remain
missing with stated uncertainty; they are not silently borrowed from Rob.

Each onboarding profile MUST declare its intended capture duration/frame
budget, required sparse label roles/counts, allowed SAM3 work, target hardware,
time/compute budget and measurable acceptance criteria before testing. Measure
preparation, annotation effort, cold training and optional personalization
separately. Report accuracy/coverage versus added labels and time to a usable
model; a quick fine-tune must not conceal hours of undocumented setup. Stop
with an actionable insufficient-data result when the budget is exhausted.
Numerical onboarding budgets are still to be established, not claimed here.

Use subject-disjoint evaluation for new-user claims and session/source-clock
separation within each person. Both eyes from the same exposure, neighboring
frames, duplicate RAW and augmented derivatives stay with their source group.
Keep final evaluation labels out of training, teacher/candidate selection and
hyperparameter decisions. A new user's adaptation set and evaluation set must
be distinct. Once multiple subjects exist, evaluate held-out users and report
per-user results rather than hiding failures in a pooled average.

Until that evidence exists, results MUST be described as **single-user
development results**. More recordings of Rob, or synthetic shadow/rotation
augmentation, do not establish cross-user generalization or fast onboarding.

## Front-loaded robustness and the last-mile carveout

The shared, bootstrappable foundation MUST do as much work as possible before
a person sits down. The intended first-use experience must not require that
person to train a model. Personal calibration and optional refinement must be
clearly distinguished from basic eye detection and useful tracking.

An optional **last-mile refinement** may adapt a verified shared foundation to
a particular person or bounded scenario using a small, declared local corpus,
or bounded near-real-time updates. It need not cold-train that foundation at
each update. This is a reuse/latency carveout, **not a provenance exception**:

- Bind it to the exact proven foundation, current recipe and local evidence.
  All custom ancestors must still have a current-checkout bootstrap path.
- Declare user/scenario identity, maximum samples, update duration, memory
  and per-frame inference budgets. Measure these budgets; do not invent a
  claim that an untested adaptation is fast enough.
- **User/scenario refinement training and inference MUST be CPU-only. They
  MUST NOT use CUDA**, including an automatic accelerator fallback. Keep
  these small native Rust models usable without a CUDA runtime. The separate
  shared foundation may use its normal accelerator; this does not authorize
  moving the local refinement's own computation onto it.
- Keep adaptation optional, resettable and isolated. Publish a new immutable
  version only after checks; failed or late updates retain the usable base.
  Do not block acquisition while retraining or treat held output as fresh.
- Do not feed a user's refinement, its selected examples, features or
  pseudo-labels back into a shared base or another user's/scenario's model.
  Independently reviewed human labels remain admissible evidence with their
  true collection/provenance history; adaptation output is not human truth.

**Immediate lighting robustness is a foundation requirement, not an adaptation
job.** A new lamp, shadow, or change in screen-emitted color/brightness must be
handled by ordinary inference on incoming frames, without waiting for dynamic
retraining. Front-load RAW photometric augmentation and suitable image/context
features, respecting clipping, sensor noise and actual visibility. Saturation,
occlusion or missing image information should cause an explicit bounded
uncertainty/abstention, not an invented perfect fit.

Evaluate abrupt illumination and screen-color transitions with adaptation
disabled as well as enabled, from fresh state and across ROI movement. Measure
latency, fresh coverage, human-label localization, and recovery behavior. A
model that needs to relearn each display color does not meet this requirement.
No existing single-user experiment is claimed to have proven this yet.

## Native Rust training boundary

Training recipes, augmentation, loss/optimization, preparation and required
reports MUST be maintained in Rust. CUDA awareness means Rust selects devices
and invokes the native CUDA-capable tensor backend; it does not require
rewriting LibTorch's kernels or claiming they are Rust kernels.

| Stage | Implementation/device policy |
| --- | --- |
| Shared Student and shared limbus patch-model training | Rust `tch`/LibTorch with CUDA; offline, front-loaded work |
| SAM3 bootstrap export and teacher generation | Native Rust SAM3 path; pinned external SAM3 assets allowed |
| RAW selection, label preparation, validation and reporting | Native Rust CPU tools; no Python, LibTorch or CUDA runtime required |
| Optional per-user/scenario last-mile training and inference | Native Rust CPU only; no CUDA dependency or execution |

The existing `sam31_student.rs` and `limbus_refiner_train.rs` optimizers were
already Rust/CUDA implementations. Their former Python preparation/reporting
tools have now been ported to Rust. The existing CUDA limbus trainer is a
**shared offline trainer**, not a compliant CPU-only personal adaptation
trainer. A new CPU adaptation trainer is not supplied by this migration.
The external canonical human labeler remains unchanged; ancillary historical
corpus-analysis scripts are not training optimizers and are not all ported.

Build the portable preparation/report/check tools independently of CUDA:

```sh
cargo build --profile live --no-default-features \
  --bin buttercup_prepare_eye_student --bin buttercup_report_eye_student \
  --bin buttercup_prepare_limbus_refiner --bin buttercup_report_limbus_refiner \
  --bin buttercup_bootstrap_check
```

See [Student reproduction](docs/eye-student.md) and
[limbus-refiner reproduction](docs/limbus-refiner.md) for native commands.
Use the host resource coordinator for builds, preparation, training and
evaluation. Training artifacts and manifests stay under `outputs`, not Git.

## Machine-checked dependency graph

Before training or reusing derived training material, run
`buttercup_bootstrap_check` on the declared dependency graph and retain its
output with the run receipt. It is a required development preflight, but is
**not yet automatically invoked by every trainer**. Successful graph validation
does not waive the cold-execution proof above.

```sh
data/target/live/buttercup_bootstrap_check --describe-source
data/target/live/buttercup_bootstrap_check outputs/RUN/graph.json
# The same source stamp is required even when invoked from a subdirectory:
data/target/live/buttercup_bootstrap_check outputs/RUN/graph.json --repo .
```

The manifest is one JSON document (not another frame stream), schema
`buttercup-bootstrap-graph-v1`, with these required top-level fields:

- `schema`, `source` (the exact `revision` and `tree_sha256` returned by
  `--describe-source`), `targets` (nonempty node IDs), and `nodes`.
- Each node has `id`, `kind`, `sha256` and `dependencies` (its direct input
  node IDs). IDs are unique and content is represented by a single canonical
  node; renaming the same digest cannot disguise an ancestor as a new root.
- Allowed root kinds are `raw`, `human_labels`, `measurements`, `sam3`,
  `source`. Roots cannot have dependencies. The `source` digest equals the
  checkout stamp. Bundle digests must identify a separately retained canonical
  inventory of the actual material and its individual content hashes.
- Derived kinds are `derived_data`, `features`, `selector`, `custom_model`,
  `last_mile_refinement`, `export`, `evaluation`. They require declared inputs;
  model nodes must trace to native RAW and current source.
- A not-yet-produced derived node uses `"planned": true, "sha256": null`.
  Root placeholders are forbidden. Any planned nodes mean this is a recipe
  preflight, **not an artifact-complete receipt**; dependent completed-looking
  nodes do not make an unbuilt ancestor proven.
- A `last_mile_refinement` additionally requires `refinement` with `scope`
  (`user` or `scenario`), `scope_key`, `base_model` (a direct shared-model or
  SAM3 dependency), `training_device: "cpu"`, `inference_device: "cpu"`, and
  positive `max_samples`/`max_update_ms`. Other kinds cannot carry this policy.

The checker rejects missing dependencies, duplicate IDs/content aliases/edges,
invalid fields and hashes, unknown root kinds, stale source stamps, cycles
even in disconnected components, local-to-shared leakage and cross-user or
cross-scenario refinement ancestry. It includes cache/feature/selector edges,
not only direct checkpoint-to-checkpoint edges. Legitimate versioned warm
starts and shared-ancestor diamonds remain allowed.

An iterative three-color depth-first search produces either a closed cycle
witness or an ancestor-first topological order. In the latter, every declared
edge satisfies `rank(input) < rank(consumer)`, providing a checkable acyclicity
certificate. Traversal avoids recursive stack overflow and bounds manifests
to 8 MiB, 10,000 nodes and 100,000 edges.

The source stamp hashes Git-known and non-ignored untracked files, dirty
content, deleted tracked paths, executable bits and symlink targets across the
whole checkout. It does not follow the runtime `data`/`outputs` links. Keep
graphs/receipts under ignored runtime output paths to avoid self-referential
source stamps. Do not edit the checkout during a stamped run.

The certificate explicitly has scope `declared_dependency_graph_only`.
**It cannot detect a lied-about root or undeclared file read, verify material
bytes, execute regeneration, enforce actual device/time budgets, or prove
accuracy.** Hash the material, audit actual inputs, enforce runtime budgets
and complete the cold run separately. A structural certificate is necessary,
not sufficient, bootstrap evidence.

## Quality and repository boundaries

Pair corpus checks with synthetic/unit tests. Report human-label localization,
coverage/dropouts, native source-time and ROI-reframe alignment, focus/lighting
subsets, latency and independent scale support. Use **SN-FEIDA** as defined in
[the area/motion contract](docs/flat-tire-area-and-motion.md), alongside those
measures—not as a replacement for them. Never normalize by the candidate's
own radius or count held predictions as fresh evidence. No calibrated gaze or
3D accuracy claim follows merely from mask agreement or a successful fit.

Source recipes and proof schemas belong in Git; captures, labels, weights,
generated datasets, logs and proof receipts stay under the checked `data` and
`outputs` runtime links. Keep the existing external-camera boundary and source
allowlist. The offline bootstrap must not require a connected camera or copied
firmware/vendor binaries, or depend on another source checkout.

## Current status and next obligations

Migration validation on September 12, 2026:

- 40 distinct native tests passed (48 test executions because the eight
  shared limbus preparation/report tests run in both binaries): 21 graph and
  source-stamp checks, five Student preparation, six Student report, and eight
  limbus label/clock/area checks. These run without CUDA or LibTorch.
- Python-reference versus Rust selection on the existing 730-frame,
  64-lineage Student development subset selected exactly the same native RAW
  hashes/order/discrete metadata and 574/84/72 train/validation/test partition.
  Floating-point scale metadata differed by at most `1.43e-14`.
- Preparation from the canonical inventory retained the same 16 reviewed
  frames, four conservative groups and all landmark/occlusion roles; native
  source receipts matched exactly. Label-coordinate parsing differed by at
  most `5.69e-14` px.
- All original fields of the 730-frame Student report (with 84-frame matched
  worker replays), four-fold 16-frame limbus report and 37-frame archived
  sequence report matched within `1e-12` absolute/relative tolerance. Largest
  observed scalar difference was `3.98e-13`. Sequence reporting retained ten
  supported area pairs and **zero ROI-reframe pairs**, rather than filling gaps.
- Runtime comparisons are under `outputs/bootstrap-rust-validation.waPTE4`;
  worker synthetic migration fixtures are under
  `outputs/student-prepare-parity-YfLKli` and
  `outputs/refiner-rust-parity-lWGjOQ`. The legacy Python implementations were
  used only as migration references, then replaced by native tools/tests;
  they remain in Git history.

These checks validate port behavior and input/metric invariants. They do not
retrain weights, establish new geometry accuracy, measure CUDA speedups or
prove robustness for new people/illumination. Historical SAM agreement remains
defeasible, not human ground truth; the old sparse sequence has missing scale,
timing and reframe coverage as described in the refiner report.

As of September 12, 2026, the Student and limbus-refiner experiments are useful
component evidence, **not a completed cold-bootstrap proof** under this
contract. SAM3-derived supervision is permitted; SAM3 ancestry alone is not a
violation. However, existing caches, historical training logs and warm-start
checkpoints do not by themselves prove regeneration on this checkout.

The native ports and graph checker have component/parity tests, not a complete
new cold training run. The next bootstrap work must supply a checked-in
end-to-end entry point and
proof receipt, rebuild all needed custom ancestors from the permitted roots,
and exercise missing-cache/unknown-ancestor rejection. New-user state isolation,
bounded onboarding budgets and multi-user evaluation remain to be demonstrated.
This contract adds development obligations; it does not pretend that those
runtime checks or the new-user dataset already exist.
