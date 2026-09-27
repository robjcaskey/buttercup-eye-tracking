# Pink waterfall eye contact sheets

Ask an agent: **“Follow docs/pink-waterfall-contact-sheets.md to make the pink
waterfall overlay for these RAW image filenames: [list].”**

This is the pink-line illustration whose strokes cross the iris and turn sharply
into the pupil like a waterfall over a well's lip. It uses the built-in image
generator to edit a RAW-derived contact sheet. SAM region masks, colored mask
grids, smooth funnels and spiral overlays are different experiments. The pink
strokes are artistic depth cues, not measured anatomy or a sign solve.

## Inputs and selection

### Twelve-frame diagnostic repeats

The preparer also accepts twelve metadata-backed frames of one common native
ROI size (up to 1920×1080), arranged without gutters in a 4×3 grid. Dimensions
come from the source records and are checked for every tile; they are not
resized to 384×256. This mode supports matched debugging of an existing sheet.
The September 21 pupil-fitter repeat used twelve 420×280 frames, giving a
1680×840 input. The generator returned 1774×887 and altered pupil shading.
The native compositor therefore transferred only extracted pink strokes onto
the original previews; it verified zero changed pixels outside that mask.
Artifacts, exact prompt and source identities are under
`outputs/pupil-fit-debug-20260921/pink-input/` and `pink-composite/`.

### Verified native-size generation: 16 images per sheet

Use **16 images in a 4×4 grid without gutters**, exactly **1536×1024**.
Each eye occupies its original **384×256** rectangle. The first built-in
generation trial returned exactly 1536×1024, with all sixteen tiles in the
original order and visually aligned tile boundaries. This is the recommended
generation layout; repeat the size/layout check for every result.

Pass a 16-frame selection and corresponding filename list to the same command
below. It automatically selects this layout. For 96 exposures, use six ordered
16-image batches. Do not silently process all 96 in one generation: that
experiment returned 1245×1263, and the 48-image experiment returned 1759×894
instead of the requested 3120×1584.

Use the emitted prompt and additionally specify: no gutters, borders or margins;
fixed x boundaries 0,384,768,1152,1536 and y boundaries 0,256,512,768,1024.
The successful trial used only the unannotated input sheet, without an extra
style-reference image. Artifacts and exact request are saved under
`outputs/pink-waterfall-16-native-v1/`.

Native-size output does **not** establish pixel preservation. The trial still
changed dark pupil shading and drew unsupported strokes on an obscured pupil.
Check those separately from dimensions; keep the untouched input for comparison.

The taller 24-image trial used four columns and six rows without gutters,
requesting 1536×1536. It returned 1254×1254: roughly 313.5×209 pixels
per eye instead of 384×256. The 4×6 layout remained visually intact, but
native dimensions failed. Keep 16 as the verified native-size batch.

### Larger native input sheets (generation resized these)

The preparer also supports **96 different RAW exposures in one input sheet**.
Prepare an ordered selection manifest with the same `frames` records as the
reference selection and a matching newline-delimited RAW filename list (stream
filenames may repeat when offsets select different exposures). Use:

```bash
cargo build --profile live --no-default-features --bin buttercup-pink-waterfall-sheet
data/target/live/buttercup-pink-waterfall-sheet SELECTION.json RAW_FILENAMES.txt outputs/NEW_RUN
```

Run from the repository root, with shared resource coordination as required by
AGENTS.md. The output directory must not already exist. This native preparer
currently accepts 16, 24, 48 or 96 metadata-backed **384×256** RAW ROIs and rejects
other dimensions rather than resizing them. It validates the source records,
saves native-size decoded previews and their RAW hashes, and places them in
an **8-column × 12-row, 3120×3168-pixel** sheet. Each cell is 390×264 with a
384×256 eye at offset (3,4); gutters account for the remaining pixels.
There is no tile resizing, cropping, or stretching. Native dimensions describe
the display previews; color reconstruction still follows the decoder below.

A 48-frame manifest uses the same command, producing an **8×6, 3120×1584**
sheet and a matching 48-image prompt. Use a frozen subset of the original
selection for matched comparisons, preserving its order and recording the
subset policy. No new random draw is necessary to compare batch sizes.

Use the emitted `pink-drop-prompt.txt` with the built-in image generator on
`contact-sheet-original.png`, preserving all 96 eyes and this exact layout.
One generation call processes the whole sheet. Do not silently split into
smaller pages or substitute a lower-resolution result. Check output dimensions
and individual tile geometry: the generator may rescale or rearrange a sheet
despite the prompt. If so, report that native-size output failed; enlarging a
smaller generated image does not restore original detail. Local preparation
guarantees native tile dimensions; the hosted edit does not guarantee them.

The historical 12-image examples below remain style references. For 96-image
runs use the generated 96-image prompt, not their 4×3 layout instruction.

Use the user's supplied native RAW filenames, preserving their order. Resolve
dimensions, stride, pixel format, Bayer phase/sensor origin and exposure identity
from capture metadata. A packed recording stream filename alone is not one
image: also retain the recording/archive, stream entry, sequence and byte
offset/length. Ask for unresolved identity rather than guessing. Do not replace
the supplied images with convenient previews or random frames.

If neither filenames nor an existing comparison selection were provided, offer
to choose random native RAW eye ROIs from the recorded corpus. Do not silently
sample. An explicit random-sample request already authorizes selection.

For an accepted random sample, enumerate metadata-backed native RAW10 eye ROI
exposures beneath `outputs/cross-video-corpus-v3/extracted`. Validate stream
existence, dimensions, stride and byte-range bounds before drawing. Deduplicate
exposures by recording/stream/offset/length; choose without replacement with a
recorded seed, before viewing the pictures. Save the population size and sampling
policy. If using an area-filtered or distinct-recording subset instead, state
that policy explicitly; it is a different population. Do not filter by whether
the eventual illustration looks convincing.

The reference draw contains 12 exposures from a population of 1,657 with seed
`7217496618581029915`. Its exact selection is recorded in
`outputs/fresh-eye-lasers-20260918/selection.json`. Replay that manifest, not just
the seed: the population or ordering may have changed.

For new files, use native Rust RAW decoding and contact-sheet assembly. Save
individual decoded previews and their processing settings, RAW hashes, ordered
source metadata, and tile coordinates in a new output directory. Use the actual
source aspect ratio; do not stretch eyes. The reference layout has four columns
and three rows with narrow dark gutters and no labels. Do not introduce matched
blur copies unless requested. The SAM `roi-contact-sheet` command is not the
preparer for this workflow. Use the 96-image native preparer above for new full
batches; filenames must be accompanied by their exact frame metadata.

The blind test's saved native assembler is
`outputs/pink-waterfall-blind-20260918-v1/preparer/`; its processing receipt is
`preparation-provenance.json` in the parent directory. It uses
`src/bin/buttercup_raw10_preview.rs` for Quad-Bayer decoding and PNG output,
color mode, contrast 1.0, per-channel 2–99.5 percentile display range and gamma
2.2. These are display settings, not untouched sensor values. The capture
manifests omit an explicit CFA phase: this decoder uses the repository's
IMX582 Quad-Bayer convention and absolute sensor origins. Record that assumption
rather than claiming the phase was independently verified from metadata.
The saved assembler is a bounded example tied to that selection/output path;
inspect and adapt it before using another list, and do not overwrite a prior run.

## Exact reference and repeatable edit

These files are the authoritative reference, under the checked runtime outputs
link (keep generated images out of the source tree):

- `outputs/fresh-eye-lasers-20260918/contact-sheet-original.png`: untouched
  1560×792 input sheet.
- `outputs/fresh-eye-lasers-20260918/contact-sheet-pink-drop.png`: accepted
  1760×894 pink waterfall style reference.
- `outputs/fresh-eye-lasers-20260918/selection.json`: ordered RAW identities.
- `outputs/fresh-eye-lasers-20260918/pink-drop-prompt.txt`: original edit prompt.

To repeat the reference experiment, use the original input sheet above. To
process new images, substitute the newly prepared unannotated sheet. Never use
the already annotated sheet as the sole edit target, which could accumulate
strokes or preserve old eye positions. Inspect the input before editing.

Use the imagegen skill and built-in image-generation tool, with the original
sheet supplied through `referenced_image_paths`. The accepted output may also
be supplied as a second, explicitly identified style reference; tell the tool
to edit the first image only and retain its eyes and layout. Do not reuse pupil
coordinates from another sheet. Use this prompt (adjust layout/count/aspect only
when the actual input differs):

> Edit this original 4-column 3-row contact sheet of 12 eye photographs. Add fine
> pink NONSPIRAL meridian-like lines running from each outer iris edge across the
> iris, then making an abrupt almost 90 DEGREE TURN DOWN INTO DEPTH at the
> existing pupil rim, like lines passing over the sharp lip of a deep cylindrical
> well / black-hole event horizon. The iris is the broad nearly flat annular
> ledge; the dark pupil is the steep shaft. Each pink line has TWO clear parts:
> a gently curved radial approach over the iris surface, then a very tight
> rounded elbow at the pupil boundary followed by a short near-axial segment
> descending steeply into the hole and fading. Emphasize the sharp
> near-right-angle change in 3D surface direction. NOT a smooth shallow funnel.
> NOT a hurricane, spiral or pinwheel. No azimuthal rotation. Use foreshortening
> appropriate to each actual eye pose; 90 degrees means into the depth of the
> pupil, not uniformly downward on the image. About 10 fine pink strokes per
> iris. Keep aperture centered on original dark pupil and its observed elliptical
> boundary, not on white glints or below pupil. Preserve original photo, iris
> texture, pupil size, reflections, grain, exposure, blur and exact 4x3 layout.
> No added dark shading, black fill or altered eye anatomy: only pink strokes.
> No dots, labels, arrows, horizontal rings, text or glow. Lines confined to
> visible iris and pupil, occluded by eyelids. Artistic depth overlay. Output
> entire contact sheet in original approximately 1.97:1 landscape aspect ratio.

Save the exact prompt, input paths/hashes, selection manifest, generator/tool
identity, output dimensions and generated result in a fresh directory beneath
`outputs`. Preserve the reference files. This makes the process repeatable;
generative edits are not guaranteed to reproduce identical pixels. The hosted
image-generation step is separate from local native RAW preparation.

## Review and show the result

### Native validator and illustrated center extraction

To keep the original photograph untouched and transfer **only the pink strokes**,
use the compositor before validation:

```bash
data/target/live/buttercup-waterfall-validate --composite \
  outputs/pink-waterfall-16-native-v1/contact-sheet-original.png \
  outputs/pink-waterfall-16-native-v1/contact-sheet-pink-drop.png \
  outputs/pink-waterfall-16-native-v1/preparation-provenance.json \
  outputs/NEW_COMPOSITE
```

This maps generated dimensions back to the original, then fits bounded per-tile
translation (up to 9 native pixels) and isotropic scale (up to 5%) using
nonpink surrounding texture. Alternating spatial samples are held out of the
fit. Poor correlations, worsening held-out alignment, boundary optima and
excessive masks are rejected; those tiles stay unchanged. It assumes the same
tile ordering and proportional grid, not arbitrarily rearranged generated tiles.
Only the extracted strokes move. No generated pupil shading, reflections or
photographic texture are copied into the final image.

Outputs: `raw-with-pink-strokes.png`, binary `stroke-mask.png`, soft
`stroke-alpha.png`, `composite-report.json`, an aligned generated diagnostic
(not the final composite), and `blink-comparison.html`. Pink excess relative to
the aligned original defines an approximate alpha; tiny color flecks and broad
solid components are removed. Natural pink glare can still contaminate the mask,
and weak stroke portions can disappear. This is not the generator's true alpha.
The native RAW recording is never modified; the background is its original
decoded RGB preview. An invariant checks exact equality outside the mask.

Run the validator on the composite with `stroke-mask.png` as the final argument
to verify that invariant independently. This mask describes the compositor's
edits; passing does not validate its anatomy or authorize changes made by the
original generator. The 16-image run at
`outputs/pink-waterfall-16-composited-v2/` preserves every outside-mask pixel.
All alignments were subpixel translations except one additional 0.5% scale
change. A known 2% scale / (3,-2) pixel control was recovered exactly on the
search grid. These are bounded checks, not a calibrated registration-error
claim. Stroke convergence still failed; do not convert preservation success
into a claim of a valid gaze center.

Build `buttercup-waterfall-validate` with `cargo build --profile live
--no-default-features --bin buttercup-waterfall-validate`. Run from the repository
root (under shared resource coordination when needed):

```bash
data/target/live/buttercup-waterfall-validate \
  outputs/pink-waterfall-16-native-v1/contact-sheet-original.png \
  outputs/pink-waterfall-16-native-v1/contact-sheet-pink-drop.png \
  outputs/pink-waterfall-16-native-v1/preparation-provenance.json \
  outputs/NEW_VALIDATION
```

An optional fifth positional argument is an authorized mask PNG of identical
dimensions: red channel above 127 permits edits at that pixel. Prefer a mask
declared independently of the returned image. Without it, the program infers
new pink strokes by red/blue excess over green relative to the original, plus
a one-pixel antialias halo. That inferred mask is a heuristic, not independent
proof: naturally pink changes may also be excluded. A tile with 25% or more
pixels excluded is rejected. Nothing establishes preservation inside a mask.

The validator uses FFmpeg only to decode PNG into RGB24 and Rust for comparison,
stroke tracing and fitting. It verifies the original PNG hash against the
preparation receipt. Dimensions must match; it never registers, resizes or
color-normalizes a generated image to conceal changes. Every RGB byte outside
the mask must match exactly. It also records absolute error magnitudes to
distinguish tiny changes from larger ones. New output directories must be under
the checked `outputs` link.

The initial center estimator thins the pink strokes, follows their inward
endpoints and robustly intersects local tangent rays. It requires six rays,
adequate directional conditioning, median perpendicular residual at most 3 px,
leave-one-ray-out movement at most 1 px, and at least 80% forward intersections.
These are conservative engineering gates, not calibrated probabilities.
Curved or nearly parallel waterfall shafts need not have one 2D intersection;
this initial estimator can abstain and is not a reconstruction of a 3D well.

Outputs are `report.json`, `centers.csv`, `excluded-mask.png` and
`outside-mask-changes.png` (red means a changed pixel outside the mask).
Dimension mismatch produces a rejection report without centers. Exit status
0 means all tiles passed preservation and yielded an admitted illustrated
center; 2 means rejected evidence; 1 means an input/tool error.

Accepted `pixel_xy_roi` is a single integer `[x,y]`, with zero-based pixel-center
coordinates. Sheet coordinates add the original tile offset; sensor coordinates
add the recorded sensor origin. Rejected centers are null. Floating-point fits
remain explicitly unvalidated diagnostics: decimal output is not evidence of
subpixel accuracy. No physical gaze origin is claimed. Establish accuracy using
independent source-grounded labels before admitting subpixel results.

Initial actual-image check: the 16-eye native-size generated sheet changed
1,474,280 of 1,474,834 pixels outside the inferred mask, with mean absolute
channel error 9.27/255. All 16 tiles failed preservation and convergence gates.
See `outputs/pink-waterfall-validation-16-v2/report.json`. Native size and
recognizable layout did not establish an unchanged photograph. The 24-eye
resized result was rejected before extraction. Analytic radial strokes on one
unchanged real RAW preview passed; a one-pixel unauthorized edit was rejected,
and an unchanged image without strokes yielded no invented center. These
controls are under `outputs/waterfall-validator-controls-v1/`; they validate
program behavior, not anatomical accuracy or a successful real-image solve.

Compare the generated sheet to its untouched input and the accepted pink-drop
reference. Check every tile: same eye and pupil, sharp elbows at the actual
pupil rim, no pinwheel rotation, no invented black fill, and strokes hidden by
eyelids. Keep failures visible and describe them; do not present the strokes as
SAM measurements. Even when instructed to preserve photographs, imagegen may
alter their pixels. Do not claim an exact RAW-preserving composite.

Check individual tile rectangles as well as overall aspect ratio. In the first
blind test, the tool preserved the overall 4×3 sheet but widened the last column,
deepened dark pupil areas, and drew strokes over a partially hidden pupil.
That test repeated the aesthetic but failed strict photographic preservation.
Retain and report those failures. On obscured pupils, an attractive well shape
does not establish where the actual pupil rim is.

Display the generated image inline and link the output. For a requested desktop
review, use the installed `queued-open` attention workflow and inspect the
outcome; queued does not mean opened. A blind workflow test must follow this
guide and produce pink waterfall lines. A successful SAM mask grid is not a
successful test of this process.
