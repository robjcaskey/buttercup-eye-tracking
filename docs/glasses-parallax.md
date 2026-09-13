# RAW glasses evidence discovery: bounded trial

This is an offline, non-learned **candidate discovery experiment**, not a
validated glasses detector. It can find temporally persistent patch copies and
test whether their residual motions differ repeatably. Every real-corpus result
currently remains `unknown`. A pair alone never establishes glasses; absent
pairs never produce glasses-off. Nothing is installed in the viewer or trained.

`src/glasses_parallax.rs` implements the numerical path;
`src/bin/buttercup_glasses_parallax.rs` supplies native RAW preparation,
source receipts, bounded selection, reporting and review views. The numerical
detector reads `frames.jsonl` acquisition fields and native RAW ranges. Targeted
context review additionally reads native `thumbnails.jsonl`/`thumbnails.oic1`,
excluding recording-start snapshots. No predictions, review guesses, installed models, learned masks, SAM
features, eye ellipses or personal calibration are inputs.

## Geometry, CFA and source time

Decode the camera's `RAW10_LE40_1X1` packing, then average complete physical
4×4 Quad-Bayer RG/GB cells, aligned to absolute sensor coordinates. The two
off-diagonal 2×2 green blocks provide a green-channel repeat check. This is a
correlated color check, not independent evidence of a reflection. Reduction
removes the regular Quad-Bayer microtexture; it does not prove remaining image
structure is non-anatomical. Tests cover all 16 crop phases with colored flat
fields. All displayed coordinates map back to native sensor cell centers:
`aligned_origin + 4 * analysis_coordinate + 1.5`.

Each eye is independent. Source-session/stream changes, nonmonotonic source
time and gaps exceeding 250 ms split sequences. For each selected 17-frame
window, use the common absolute-sensor ROI intersection, aligned to complete
Quad-Bayer cells. Margins outside this intersection are unobserved. Refuse an
intersection smaller than 64 native pixels in either dimension. Changes of ROI
generation are retained in every source receipt; they do not authorize treating
different local pixel coordinates as the same sensor point. Velocities use
the actual timestamp differences in seconds. Missing source identity metadata
in older captures remains missing, not inferred from host time.

## Candidate and control definitions

The historical cleanroom tools were inspected read-only. Their essential
ideas—RAW reduction, self-copy candidates, temporal persistence, motion controls
and anonymous layers—are native here. This **is not numerical parity with their
SIFT/CLAHE, pyramidal LK, clustering or cross-eye transform algorithm**. The
new fixed-scale patch candidate generator has different recall and failure
modes. No execution/import from that checkout is required.

Version-one numerical settings are deliberately explicit:

- Maximum 96 spatially suppressed minimum-eigenvalue corners per initial image.
- 9×9 cell normalized patches: a **36×36 native-pixel footprint**. Endpoint
  patches cannot overlap. Separation is 9–60 analysis cells (36–240 native px).
- Initial normalized correlation ≥0.90; this must persist in ≥75% of source
  frames. Green correlation ≥0.85 must independently meet that temporal fraction.
- Integer patch tracking searches ±5 cells (±20 native px) per exposure;
  normalized correlation ≥0.88 and forward/backward error ≤1 cell (4 native px).
  A lost track is excluded, never held and counted as fresh support.
- Robust shared image-plane similarity removes translation, rotation and
  isotropic scale. Four trimmed fitting rounds retain the best 75% of tracks;
  at least 12 tracks must lie within one cell in every exposure. Its texture
  support remains anonymous; it is not an established eye/material layer.
- Both copy residual RMS values and centered pair-differential RMS must exceed
  0.75 cell (3 native px), with support in both chronological halves.
- Fit a scalar residual-velocity gain on the first half, evaluate untouched
  second-half velocities against the zero-velocity baseline. Require skill
  ≥0.5 and ≥0.15 above the best cyclic endpoint-time null. These seven nulls in
  a 17-frame window are limited diagnostics, not probabilities or a full
  temporal-block significance test. A scalar gain misses direction-changing
  inter-layer transforms.

A surviving result is only `anonymous_differential_candidate`. Strong glasses
evidence additionally needs independently supported eye/material provenance,
two reflective copies rather than anatomy/same-rectangle edges, and adequate
camera/head-motion controls. This implementation does **not** establish those
conditions, so `strong_glasses_interval` is explicitly false. No front/back
surface assignments or physically meaningful depths are invented.

## Reproduction and review

Use the host resource coordinator, CPU only, two build jobs and low priority:

```sh
agent-coord request --owner glasses-parallax --resource 'cpu=*;memory-bandwidth=*' \
  --mode shared --priority 2 --ttl 7200 --note 'bounded RAW-only trial; nice 15'
nice -n 15 cargo test --profile live --no-default-features \
  --bin buttercup_glasses_parallax -j 2 -- --nocapture
nice -n 15 cargo build --profile live --no-default-features \
  --bin buttercup_glasses_parallax -j 2
nice -n 15 data/target/live/buttercup_glasses_parallax outputs/NEW-TRIAL \
  outputs/calibration-corpus/sam31-mouse-3d-1789236126-183797734.tar \
  outputs/calibration-corpus/sam31-mouse-3d-1789236132-000803142.tar \
  outputs/calibration-corpus/sam31-mouse-3d-1789236215-642693000.tar \
  outputs/calibration-corpus/sam31-mouse-3d-1789236243-817662332.tar \
  outputs/calibration-corpus/sam31-mouse-3d-1789236275-715858045.tar \
  outputs/calibration-corpus/sam31-mouse-3d-1789236306-104694526.tar
```

The output directory must be new and resolve beneath the checked bulk runtime
root. The bounded driver selects at most three windows per eye: beginning,
middle and end of the longest eligible acquisition epoch, then another epoch
only if fewer than three unique starts exist. This is not a full-corpus mode.
Each source has stream/range, source timestamp/key, original acquisition crop
and SHA-256 of the actual RAW bytes. The frame index is also hashed.

Each window has `window.json`, `contact.ppm`, optionally `contact.png` (standard
ffmpeg encoding), and `motion.html`. The standalone HTML shows actual ordered
source frames with a pause/scrub control and exact source receipt. It does not
animate hypothetical surface depths. Contacts display five actual exposures;
both views overlay the first eight candidate pairs, yellow A/magenta B. Every
candidate, including those not drawn, is in JSON. Contrast stretching is only
for display and can accentuate illumination changes.

Nine native tests cover packing, all Quad-Bayer phases, missing/gapped times,
duplicate anatomy, no motion, common translation, same-layer copy motion,
common rotation/scale, time-shuffled tracks and genuine differential motion.
The three-layer image test synthesizes native LE40 and exercises the actual
decode/corner/track/pair path: zero accepted differential candidates in the
three negative variants and four in the differential variant. Synthetic
positives still do not receive a glasses label.

## September 12 small-trial results

The final reproducibility rerun is
`outputs/glasses-parallax-small-v3/report.json`: the same six recent captures
plus the one older capture below, 42 windows / 714 frame usages / 70 initial
pairs / 25 persistent instances / **zero differential candidates** in 3.66 s.
It includes compiled driver/algorithm, Cargo.lock and executable SHA-256.
The v2 paths below preserve the first independently inspectable trial; v3
produces the same per-capture candidate counts, with final source receipts and
explicit malformed-metadata handling. Its window-directory names are identical.

`outputs/glasses-parallax-small-v1/report.json` records the original fixed-ROI
coverage failure: only two eligible windows in the six recent recordings.
It is retained as evidence; it is superseded for ROI handling by
`outputs/glasses-parallax-small-v2/report.json`.

The six-recording v2 trial used 36 windows, 612 eye-source frame usages and
598 unique eye-source identities out of 2,184 available (27.4%). Short overlap
between windows in the 44-frame capture accounts for duplicate usages. There
were 54 ROI-origin transitions inside these windows. All source ranges decoded;
9/36 windows had fewer than 12 surviving tracks. No calibrated uncertainty,
human reflection localization score or cross-user accuracy is available.

| Recording timestamp | Initial self-pairs | Persistent pairs | Temporal/null-qualified pairs |
| --- | ---: | ---: | ---: |
| 1789236126 | 10 | 4 | 0 |
| 1789236132 | 6 | 1 | 0 |
| 1789236215 | 19 | 7 | 0 |
| 1789236243 | 19 | 5 | 0 |
| 1789236275 | 7 | 0 | 0 |
| 1789236306 | 2 | 1 | 0 |

These are 63 tracked candidate-pair instances and 18 temporally persistent
instances, **not 18 distinct reflections or independent observations**.
The elapsed CPU trial was 3.17 seconds on this host, excluding build and human
inspection. The earlier assistant review suggests the first four recordings
are probably off and the last two probably on. These guesses are not detector
inputs, human truth or a train/test split; all six are one sitting of Rob.

Inspect `1789236215…/subject-right-epoch-2-start-0`: persistent pairs mark
repeated eye/eyelid structures and edges, not established duplicated glare.
Inspect `1789236306…/subject-left-epoch-0-start-0`: the pair remains correlated
in all frames, but held-out velocity skill is −0.725 with null margin −0.634.
Its 4.27-native-pixel differential RMS alone would be misleading; the temporal
gate rejects it. Inspect `1789236275…/subject-right-epoch-0-start-94`: only two
complete tracks survive as appearance/illumination changes. This is missing
evidence, not a glasses-off inference.

At that small-trial checkpoint, no full sweep was cleared and no training
started. After independently inspecting off6215 and on6306 contacts, the root
reviewer authorized **corpus-wide candidate discovery only**, to look for better
controlled-motion evidence in older captures. That authorization does not
validate a glasses detector, promote any labels, or authorize training.
The unresolved issue remains real reflection recall and independent material/eye
provenance. An all-unknown sweep would not complete the glasses classifier.

The separate oldest compatible calibration archive,
`sam31-mouse-3d-1788520697-558721839.tar`, was selected by filename acquisition
time, without an eyewear/model selector. Its bounded report is
`outputs/glasses-parallax-older-v2/report.json`: 6 windows, 102 of 483 eye-source
frames (21.1%), seven initial/persistent pairs, zero differential candidates;
five windows have fewer than 12 complete tracks. The absent modern source-key
metadata remains null in receipts. This provides an older session failure check,
not an independent human-labeled evaluation or a new-user result.

## Corpus-wide bounded discovery

The maintained Rust inventory/batch implementation is
`src/glasses_parallax_inventory.rs`. Invoke the same binary with:

```sh
nice -n 15 data/target/live/buttercup_glasses_parallax outputs/NEW-CORPUS-RUN \
  --corpus outputs data/recordings data/labeled-corpus
```

It recursively discovers `.tar` bundles, `frames.jsonl` bundles and `.raw10`
files under those roots, canonicalizes symlinks and directories, and excludes
its current output. It prefers original calibration tar archives, then other
archives, then directory copies. It reads acquisition metadata and native RAW
only. All compatible indexed RAW ranges are content-hashed for inventory even
though only a bounded subset is decoded and evaluated.

An exact source-crop identity uses sensor timestamp, sequence, eye, native crop,
shape, stride and format; offsets and optional viewer metadata do not create a
new identity. The same identity is collapsed only after confirming matching RAW
SHA-256. Conflicting bytes are reported and not admitted as another observation.
Metadata identity collisions and missing metadata are not silently repaired.
`unique-sources.jsonl` retains every accepted identity/hash/first container;
`inventory.json` reports canonical locations, duplicates, missing/unsupported
records and unindexed RAW files. An isolated RAW file lacks a temporal index and
remains parallax-ineligible, even if its bytes match an indexed source.

After deduplication, the same acquisition-only bound of at most three windows
per eye per container applies. Long recordings are **not exhaustively sampled**;
the report gives unique sampled source-crops versus the inventory denominator.
No recorded model geometry, training ranking or assistant review selects the
windows. A duplicate extraction contributes no additional observations.
Missing streams, short sequences, source gaps and insufficient ROI overlap are
explicit failures/abstentions. This is single-user discovery, not a trained model
or cross-session accuracy measurement.

The run streams `inventory-progress.jsonl` and `evaluation-progress.jsonl`.
`report.json` includes source/executable/inventory hashes and aggregate coverage.
`candidate-ranking.json` puts anonymous gate survivors first, then ranks by
held-out residual-velocity skill, including rejected candidates. This ranking
is a debugging aid, not a confidence probability or a glasses classification.

### Completed discovery run

`outputs/glasses-parallax-corpus-v1/report.json` completed in 153.48 seconds,
including the inventory hash pass. Its numerical algorithm hash matches the
small-v3 trial exactly; no thresholds were loosened after seeing this corpus.
Ten native tests passed (the nine numerical tests plus source-identity
deduplication). Source, executable, inventory and unique-source manifest hashes
are in the final report.

| Inventory/coverage measure | Result |
| --- | ---: |
| Canonical archive/index locations | 364 |
| Containers with unique compatible source crops | 274 |
| Duplicate-only containers | 80 |
| No-compatible-source / malformed-index containers | 8 / 2 |
| Indexed records | 679,355 |
| Unique compatible source crops | 409,138 |
| Byte-verified duplicate records | 35,120 |
| Unsupported or missing-stream records | 235,097 |
| Conflicting identities with differing bytes | 0 |
| RAW bytes hashed (includes duplicates) | 59,107,375,440 |
| Unindexed RAW files (parallax-ineligible) | 107 |
| Containers with analyzed windows | 219 |
| Analyzed windows / frame usages | 1,240 / 21,080 |
| Unique analyzed source crops | 19,694 (4.81%) |
| Windows with fewer than 12 complete tracks | 563 (45.4%) |
| ROI-origin transitions within analyzed windows | 498 |
| Initial pair instances / persistent instances | 2,078 / 1,025 |
| Motion-only gate passes | 34 |
| Full anonymous differential candidates | **0** |
| Positive glasses labels | **0** |

The 55 selected containers without an analyzed window remain short or otherwise
ineligible, not negatives. Seventeen eye entries have no unique source frames.
Three prospective windows fail the minimum common-ROI overlap. Of the 107
isolated files, 89 match indexed frame content; none contributes temporal
support. Missing-stream records account for most exclusions: 224,074 references
to absent archived streams and 1,293 absent directory streams. Recovered RAW
directories still contribute available originals. Two very large diagnostic
indexes exceed the bounded 64 MiB index reader and are explicitly excluded.
The inventory supports `.tar`, `frames.jsonl` and `.raw10`; unrelated `.raw`
unit/parity fixtures are not treated as measured acquisition roots.

The result is **failure to find qualifying evidence with this matcher and this
sampling policy**, not evidence that all recordings were glasses-off. The
4.81% temporal sampling fraction, 36-native-pixel patch footprint, complete-track
requirement and minimum common-motion support can miss useful reflections.

Reviewed examples in the final corpus output:

- `sam31-mouse-3d-1788787050-568039862-77001c70/subject-left-epoch-0-start-86`:
  pair 0 has 100% patch/color persistence, held-out skill 0.951 and null margin
  0.920, but only 9 common-fit inliers at the worst frame (12 required), despite
  43 complete tracks. The contact places the endpoints on lower eye/lid
  structures; independent reflective provenance is absent. Its numerical motion
  support is not enough to identify glasses.
- `both-eyes-1788654713-795382588-9595443f/subject-right-epoch-0-start-104`:
  very high scalar motion agreement is accompanied by inadequate common-motion
  support and too little motion in one endpoint. Contacts show the points along
  lower eye structures. A near-perfect scalar fit is not independent evidence.
- `sam31-mouse-3d-1788857699-852916521-10f606a5/subject-right-epoch-0-start-38`:
  pair 1 has 100% copy persistence and skill 0.990, yet only 0.125 native-pixel
  differential RMS. Common motion, including repeated lid/scleral structure,
  explains the apparent match; it fails differential support.
- The recent probably-off `1789236243…/subject-right-epoch-0-start-204`
  includes a motion-only pass with skill 0.981, but copy persistence is only
  23.5%. This reinforces that motion and copy-identity tests must both hold.

Two reporting limitations are explicit for v1. The generic rejection string
`differential_or_temporal_null_gate_failed` also covers a common-similarity
inlier failure; consult `common_similarity_by_frame` in the linked window
rather than assuming the motion-only boolean is false. Also, a zero held-out
velocity denominator can yield displayed skill 1 with null margin 0; it never
passes the motion gate but can sort high in the debugging list. Such degenerate
scores are not interpreted as supporting evidence. These limitations do not
change the zero accepted-candidate count or authorize threshold relaxation.

## Known glasses-on 6306 targeted investigation

After the user redirected work to the recent known-on recording, broad-corpus
work stopped. `--dense BUNDLE.tar` evaluates 17-frame windows every eight frames
within that named recording; it does not change numerical gates. The final
targeted diagnostic is `outputs/glasses-parallax-6306-dense-v3/report.json`,
covering all 398 eye-source frames in 48 windows. Initial proposal and failed
tracking-prefix diagnostics distinguish proposal failure from persistence loss.

Read `outputs/glasses-parallax-6306-dense-v3/targeted-review.md` for the visual
evidence and exact source identities. A clear central reflection is tracked for
17 right-eye frames (sequences 1223–1239), changing (−4,+4) native pixels relative
to lid texture. The first and alternative iris-reference attempts fail after
three and ten frames, respectively; both are retained. The central reflection
has no automatic matching-copy proposal in that good-support window. Its surface
could be corneal or spectacle; a second reflective layer is not established.
This is measurable single-reflection motion, **not a validated glasses-specific
parallax detection** and not evidence that the whole recording lacks motion.

`--tracks WINDOW.json ASSISTANT_SEEDS.json` provides maintained native diagnostic
tracking of explicitly assistant-selected native-coordinate seeds. It verifies
each RAW hash against the original window and outputs fresh/censored paths,
common-motion residuals and pairwise relative motion. Seeds are not human labels
or physical surface assignments. The optional `native-raw-source-index.jsonl`
also supports the existing native RAW preview tool without any learned overlay.
