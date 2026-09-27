# Compact ROI ellipse histories and focus regions

`buttercup_roi_focus` is an offline experiment. It inventories recording bundles,
keeps one record per indexed ROI exposure, converts each available ellipse into
both mathematical circular-iris interpretations, learns competing 3D focus
volumes from the stream, then classifies each interpretation against those
volumes. It does not alter the live viewer or select a trusted gaze sign.

The reusable input type is `focus_region::TheoreticalEllipseExplanations`:
two `GazeRay` values with origins in iris-radius units, unit directions, and
their angular separation. Its constructor uses Buttercup's shared perspective
circle unprojection. Camera coordinates are sensor-right/down, +Z toward camera.
No intended calibration target, previous selected sign or screen mapping enters
the region estimator. An optical-axis ray is not calibrated visual gaze.

## Native commands

Use `agent-coord run` around material CPU/I/O work, as required by the host.

```sh
cargo build --profile live --no-default-features --bin buttercup_roi_focus
data/target/live/buttercup_roi_focus pack /tmp/roi-ellipses.bin \
  outputs/calibration-sign-sam-cold-20260918-v1/native-conics.jsonl \
  outputs data/recordings
data/target/live/buttercup_roi_focus replay /tmp/roi-ellipses.bin outputs/NEW_FOCUS_RUN
data/target/live/buttercup_roi_focus compare /tmp/roi-ellipses.bin outputs/NEW_COMPARISON.json
data/target/live/buttercup_roi_focus controls outputs/NEW_CONTROLS.json
data/target/live/buttercup_roi_focus movie /tmp/roi-ellipses.bin \
  outputs/NEW_FOCUS_RUN outputs/NEW_AMBIGUITY_MOVIE
data/target/live/buttercup_roi_focus continuity \
  outputs/NEW_FOCUS_RUN outputs/NEW_CONTINUITY_AUDIT
data/target/live/buttercup_roi_focus model-eval /tmp/roi-ellipses.bin \
  outputs/NEW_FOCUS_RUN outputs/calibration-sign-sam-cold-20260918-v1 \
  outputs/NEW_CONTINUITY_AUDIT outputs/NEW_AMBIGUITY_MOVIE outputs/NEW_MODEL_EVAL
data/target/live/buttercup_roi_focus fresh-focus \
  outputs/NEW_SEGMENTATION outputs/NEW_FOCUS_RUN outputs/NEW_FRESH_FOCUS
zstd -T4 -6 --keep /tmp/roi-ellipses.bin
```

`compact OLD NEW` upgrades the earlier 128-byte format and preserves the old
file. Current recordings and original RAW pixels are referenced, not copied
into the ellipse archive. The explicit `/tmp` output is for this temporary
experiment; generated reports remain under the checked bulk-data output link.

`movie BINARY REPLAY_DIR NEW_OUTPUT_DIR [RAW_RECOVERY_DIR]` renders every saved
`multiple` record, exact-timestamp companion eyes and one nearby exposure on
either side. It verifies the binary hash, unchanged candidate geometry and
classifications, source frame-index hashes, and exact coverage. The output has
recording/clock chapters and a per-exposure JSONL timeline with RAW SHA-256,
candidate indices, classifications and areas. The optional recovery directory
is used only when an archive lacks its RAW streams, and must have the identical
frame-index hash; filenames alone cannot authorize substituting a recording.

White curves are the saved fitted ellipses; cyan/pink arrows show both nominal
3-D normals, and green boxes show competing inferred focus volumes. Normal
continuity stabilizes presentation colors only; it does not choose a sign.
The monochrome RAW10 mosaic retains every source pixel without 4x4 averaging,
with a fixed tone range per eye/clock group. Source time is visible; playback
slows short intervals by 2x with 0.12–0.6-second presentation holds, shortening
and labeling longer gaps. Missing companion exposures remain absent. This is
historical diagnostic review, not new model inference or verified gaze truth.

`continuity REPLAY_DIR NEW_OUTPUT_DIR` counts ambiguous exposures whose two
camera-coordinate normals are unequally close to the preceding unique
interpretation. Compare the immediately previous same-eye source exposure and
the most recent uniquely compatible exposure, without crossing recordings or
clock epochs. Main thresholds are at most 5 degrees to one candidate and at
least 20 degrees to the other. Report 3/10-degree sensitivity and maximum ages
of 0.25/0.5/1/2 seconds. Missing/rejected intervening observations, provider
changes and unnormalized area changes remain explicit. This direct comparison
does not propagate a selected branch through a chain of ambiguous frames.

The earlier uniquely compatible branch is not known-correct gaze. The audit
also reports how far that earlier frame's rejected alternative missed the
inferred focus volume. A miss just above the 2-degree cutoff is a marginal
exclusion, even if the two normals are far apart. RAW examples can expose this
threshold crossing without any convincing new sign evidence.

`model-eval BINARY REPLAY_DIR MODEL_RUN CONTINUITY_DIR MOVIE_DIR NEW_OUT`
evaluates the frozen current-only and two-frame ROI sign networks on every
`multiple` record. It verifies model hashes, reconstructs original fitting and
validation partitions, and excludes their RAW content and viewer sessions.
Recording days from the original corpus use their saved held-out-day model;
previously unseen days use the unweighted mean of all three frozen folds, with
individual scores and disagreement retained. This is historical experimental
comparison, with a native bootstrap provenance preflight; it does not retrain,
promote a model or establish a current-checkout cold-bootstrap proof.

The two-frame arm requires the immediately preceding same-eye native exposure
to pass the original source-clock, consecutive-sequence and 250 ms gates. The
current-only arm receives the current image in both input slots, matching its
training. Shared preprocessing and inference remain unchanged. Exact original
held-out frame predictions must reproduce both target and branch scores within
1e-6. Source-index and RAW hashes tie every result to the ambiguity movie.

The original image-support threshold of 0.8 and camera-X branch-separation gate
of 0.08 remain in force. Mapping scores onto archived virtual-contact fits is
explicitly diagnostic; same-provider SAM selections are reported separately.
The input ellipses, disk areas and both possible normals remain unchanged.
Missing sign labels stay missing: model agreement and agreement with the prior
geometric continuity heuristic are not physical sign accuracy. The output
includes per-record predictions, per-provider/day/source totals and selected
neighboring RAW review sheets. Their large images use display-only 4x4 CFA
means; the smaller images show the exact normalized 32x24 neural inputs.

`fresh-focus FRESH_SEGMENTATION_DIR ORIGINAL_FOCUS_DIR NEW_OUT` now requires
consistent frontal-equivalent area **before** calculating circle interpretations,
fitting focus regions or counting ambiguity. Missing/rejected fits, area outliers
and frames without sufficient area evidence are excluded from those stages.
Only retained inputs are used for subsequent analysis. The original focus
directory supplies cohort identity metadata; its inferred regions and ray
classifications are not reused.

The shared `area_consistency` stage uses each fresh fit's same-provider/source/
epoch/eye neighbors within one source second on either side. Exclude the tested
frame from its reference. Require at least six usable neighbors, two on each
side and 100 ms of span; trim 20% at each tail before taking their mean. Accept
only an area ratio within `[2/3,1.5]`. Admission is independent of gaze candidates,
regions, old ambiguity labels and sign predictions. The area formula requires
no tilt-sign choice. This preserves the previous main area rule while moving
it ahead of inference; insufficient evidence is now a hard exclusion.

Build fresh focus regions separately for each provider using only retained
exact-timestamp eye pairs and the shared estimator's unchanged support rules.
Only retained frames enter `classifications.jsonl`; empty or unsupported regions
remain unresolved. `eligibility.jsonl` and `excluded-inputs.jsonl` are separate
audit trails. `retained-inputs.jsonl` is the filtered dataset for follow-on
experiments, with original RAW references, fitted contours and admission evidence.
`sam-regions.json`, `obelisk-regions.json` and `intersections.jsonl` record which
retained exposures contributed. Runtime invariants reject duplicate exposures
and ensure every estimator/scorer input passed area admission.

The prior frozen-region/post-hoc-area report remains historical evidence under
`outputs/roi-fresh-focus-v1`; the current ordering is recorded under
`outputs/roi-area-first-focus-v1`. This is a retrospective analysis of the
original ambiguous subset, not a whole-corpus ambiguity rate or measured sign
accuracy. Prefix-region results still use symmetric, retrospective area
admission and must not be presented as a causal online validation.

Only exposures actually rerun through each model enter its local area reference.
This subset omits surrounding non-ambiguous frames; archived contact areas are
not mixed into fresh model references. The measure is unnormalized `pi*a*a`,
not pupil aperture area or independently scale-normalized SN-FEIDA. Head motion
can change it; a sustained wrong ellipse can pass a relative stability check.
Area rejection removes an exposure and does not distinguish its two signs.

## Input and missing-data rules

Search uncompressed tar bundles and extracted directories with `frames.jsonl`
under the supplied roots. Training/replay indexes without an identified ROI
are listed as excluded, with counts and source paths. Original native indexes
are not restricted to successful calibration intervals or good fits.

Use an exact clock/sequence/crop match to the historical SAM conic export when
present. Otherwise retain a source-bound archived semantic ellipse or explicitly
marked weak-perspective virtual-contact reconstruction. The latter is a
counterfactual shape from old model output, not a fresh segmentation or a
verified ancestor for training. Retain missing fits, rejected evidence and
failed circle unprojections as distinct states. Missing exports do not prove
detection failed. Match asynchronous contact output to its original exposure,
not to the later frame on which it happened to be displayed.

Metadata-identical copies are aliases only when their encoded per-frame
ellipse/clock/crop histories also match. This does not certify RAW equality.
Do not pool those aliases as independent observations. Contact sheets read
original RAW and check its SHA-256 against the original SAM-conic source.

## Binary version 2

Everything is little-endian. The header is 32 bytes: magic `BCROI002` (8 bytes),
version `2` (u32), record size `160` (u32), JSON dictionary length (u64), and
record count (u64). The dictionary follows, then exactly that many records.
The dictionary contains source paths/prefixes, stream names, metadata hashes,
source-clock epochs, aliases, exclusions, scale references and assumptions.
The `.bin.zst` file is an ordinary Zstandard-compressed copy of the entire file.

| Byte offset | Type | Meaning |
| --- | --- | --- |
| 0, 4 | u32 each | Source dictionary ID, clock epoch ID |
| 8 | u16 | Eye/ROI ID |
| 10, 11 | u8 each | Ellipse provider, frame flags |
| 12, 20, 28 | u64 each | Sequence, source nanoseconds, host-arrival nanoseconds |
| 36, 40, 44, 48, 52 | u32 each | Sensor x/y, width/height, RAW stride |
| 56 | u64 | Offset within original RAW stream |
| 64, 68 | u32 each | RAW byte length, source-local stream ID |
| 72 | five f64 | Ellipse sensor cx/cy, major/minor semi-axis pixels, angle radians |
| 112, 116 | f32, u32 | Heuristic evidence grade, original frame-index row |
| 120 | f64 | Projected complete ellipse disk area, `πab`, pixels² |
| 128 | f64 | Frontal-equivalent disk area, `πa²`, pixels² |
| 136, 144 | f64 each | Independent linear scale `s`, SN-FEIDA `πa²/s²` |
| 152, 156 | u32 each | Scale-reference ID, area validity flags |

Providers: 0 missing, 1 historical pinned SAM conic, 2 archived semantic
ellipse, 3 archived virtual-contact reconstruction. Frame bits: 1 ellipse
present, 2 saved SAM admissibility gate passed, 4 recorded clock epoch present.
The evidence grade is not a probability. Area bits: 1 unnormalized areas
available, 2 independent normalization available. Missing areas/scale use NaN;
missing reference ID is `u32::MAX`. JSON exports represent these as null.

“Post-affine area” here means **frontal-equivalent outer-limbus disk area** under
the weak-perspective circular-disk assumption. Its warp determinant is `a/b`,
so `πab * a/b = πa²`. Both sign interpretations share it. It is not visible mask
area, pupil aperture area, annulus area or curved anatomical surface area.
The current export has no verified independent scale, so all SN-FEIDA fields
remain missing. Never use the candidate's own radius as normalization. See
[the area contract](flat-tire-area-and-motion.md).

## Streaming estimate and second sweep

For each recording/source epoch, use only fresh, exact-timestamp pairs from
eyes 1 and 2. Evaluate all four ray combinations. Retain forward closest
approaches with at least 3° crossing, distances 2–250 iris radii, and gap within
the declared 2° angular allowance (0.75-radius floor). These are hypotheses.

Cluster midpoints online with a 6-radius association distance. Each exposure
pair updates a cluster at most once; alternatives split its weight. Weight
repeated samples within a 200 ms bin less, keep at most 64 clusters and 512
points per cluster, and report discarded births when capacity is reached.
Sort possibilities geometrically so branch numbering does not change results.
Retain clusters supported by at least eight pairs, three time bins, 0.4 seconds,
and 35% of the strongest qualifying cluster's weight. Their 5–95% coordinate
extents plus 0.5 radius give conservative axis-aligned focus volumes. This is
an engineering heuristic, not a calibrated probability or identified monitor.

The second sweep computes the exact shortest distance from each forward ray to
each retained box. `inside` means intersection; `nearby` means angular miss at
most 2°; otherwise `outside`. Save miss distance in iris radii, angle in degrees,
and forward distance. “In frame” in this experiment refers to inferred focus
volumes, not the sensor's ROI crop. A frame then has zero, one or multiple
compatible interpretations. Missing ellipses, unusable evidence, invalid
unprojections and unresolved regions stay separate from zero interpretations.
The existing ellipse is never modified to obtain stable area or a desired ray.

`region-evolution.jsonl` records intermediate estimates. Full second-sweep
classifications are retrospective and self-influenced. A prefix-70% estimator
also scores the remaining source frames without fitting to them. This measures
future compatibility, not accuracy. A late new fixation may be outside earlier
volumes even when its ellipse and gaze are correct.

## Validation and limits

The binary round trip is byte-exact. `compare` verifies every stored area,
compares ±10% nominal focal length and SAM-only fitting on the same admissible
SAM source exposures, and requires alternative-order invariance. `controls`
projects known 3D circles, unprojects them, and exercises forward-ray distance
and known planar fixation histories. Incorrect interpretations can form coherent
competing focus regions even with exact conics; the synthetic result explicitly
counts those rather than calling coherence sign ground truth.

The nominal camera has focal `(4000,4000)` and principal point `(4000,3000)`.
Equal iris radius across eyes, unmeasured distortion/refraction, visual-axis
offset and camera motion can all bias inferred focus. No physical centimeters,
monitor boundaries, human label localization or true sign accuracy are measured.
These historical diagnostics do not train descendants or promote a custom model.
No claim about improved SN-FEIDA follows from selecting a branch: both branches
share the input ellipse, and independent scale is absent.
