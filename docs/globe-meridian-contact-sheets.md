# Single globe-meridian illustration variant

## Three-phase landmark variant

For explicit endpoint control, use two image-generation calls followed by native
dot extraction and curve fitting. The saved matched experiment is
`outputs/canthus-three-phase-16-v1/`. Both exact generation prompts are saved
there as `phase-1-prompt.txt` and `phase-2-prompt.txt`.

1. On the original sheet ask for one pink dot on a visible extreme canthus per
   eye (leftmost if both are visible; none if neither is). No curves or other
   marks. A canthus is the actual upper/lower lid junction, not a crop edge.
2. Extract those dots and redraw only them on the unchanged original. Feed
   **that restored RAW preview sheet**, not the generated photographs, to the
   second image-generation call. Preserve the first dot, add another pink dot
   only if the opposite corner is visible, and add one blue predicted apex for
   every eye even when no corner is visible. Ask for dots only.
3. Extract phase-two dots, retain phase-one positions rather than letting the
   generator move them, and fit the curve in Rust. The final HTML exposes each
   phase separately and alternates the fitted curve pink/black every second.

Build `buttercup-waterfall-validate` as below. Commands for the saved images:

```bash
data/target/live/buttercup-waterfall-validate --landmarks \
  outputs/globe-meridian-16-native-v1/contact-sheet-original.png \
  outputs/canthus-three-phase-16-v1/phase-1-generated.png \
  outputs/globe-meridian-16-native-v1/preparation-provenance.json \
  outputs/NEW_PHASE1

# Generate phase 2 using NEW_PHASE1/raw-with-landmarks.png first, then:
data/target/live/buttercup-waterfall-validate --landmarks \
  outputs/globe-meridian-16-native-v1/contact-sheet-original.png \
  outputs/canthus-three-phase-16-v1/phase-2-generated.png \
  outputs/globe-meridian-16-native-v1/preparation-provenance.json \
  outputs/NEW_PHASE2 outputs/NEW_PHASE1/landmarks.json
```

The saved second generation specifically used
`outputs/canthus-three-phase-16-v1/phase-1-v2/raw-with-landmarks.png`.
Use that exact prior receipt when replaying it; a different first-stage image
requires a new second-stage generation. Output includes `landmarks.json`,
`raw-with-landmarks.png`, `curve-pink.png`, `curve-black.png`, both edit masks,
and `three-phase.html`. Coordinates are in the original ROI; fractional dot
centroids do not establish anatomical subpixel precision.

Registration reuses the bounded surrounding-texture alignment. Dot extraction
uses new saturated pink/blue components with bounded size and roundness.
The first generated dots were larger than requested, so extraction permits up
to 500 colored pixels and a 22-pixel bounding dimension; accepted positions are
redrawn with radius 3. Multiple ambiguous candidates are not silently selected.

With corners P0/P2 and predicted apex A, the quadratic Bezier control point is
`C = 2*A - (P0+P2)/2`. Thus B(0)=P0, B(1)=P2 and B(0.5)=A exactly; A is the
maximum signed departure from the endpoint chord, not necessarily the highest
image-y point. This is an illustrative interpolation rule, not a recovered 3D
globe. Reject short chords, extreme or out-of-span apexes. One or no marked
corners cannot uniquely determine this curve: retain the blue apex and show
`insufficient_landmarks`, without inventing a second endpoint.

The initial run produced 16 first-corner markers, 16 apexes, and 14 curves;
tiles 12 and 14 lacked a second extracted corner. Several apex predictions
still hug the upper eyelid, and some corner dots may be crop-edge proxies.
The two-stage prompting makes those failures inspectable; it does not prove
the model located true canthi. Verify anatomical placement separately. Native
rendering verifies that pixels outside the dot/curve masks are unchanged.

## Single-prompt variant

This variant draws one smooth pink arc across the front of each eyeball,
roughly left-to-right between the extreme inner and outer eye corners (the
medial and lateral canthi, where the upper and lower eyelid margins meet).
Missing endpoints are
illustratively inferred. They are not measured anatomy or training labels.
The original waterfall recipe remains separate.

Follow the input identity and RAW filename selection policy in
[the waterfall guide](pink-waterfall-contact-sheets.md). Use 16 exposures in a
4×4 sheet, 1536×1024 without gutters, for the verified native-size layout.

```bash
cargo build --profile live --no-default-features \
  --bin buttercup-globe-meridian-sheet --bin buttercup-waterfall-validate
data/target/live/buttercup-globe-meridian-sheet \
  SELECTION.json RAW_FILENAMES.txt outputs/NEW_MERIDIAN_INPUT
```

Use shared resource coordination for builds/preparation. The wrapper reuses the
same RAW decoder, source validation and assembly code; it emits
`meridian-prompt.txt` instead of the waterfall prompt. Use built-in imagegen to
edit the newly prepared `contact-sheet-original.png`, after inspecting it.
The hosted edit is separate from the native preparation command. Save the exact
request and result in that run directory. Check dimensions and tile order.

Canonical prompt (layout/count/dimensions substituted by the preparer):

> Edit this original 4-column 3-row contact sheet of 12 eye photographs. Add
> exactly ONE fine pink curved meridian line per eye. The line travels roughly
> left-to-right from the extreme inner eye corner, over the convex anterior
> surface of the eyeball, to the extreme outer eye corner, like
> a single arc drawn on a globe. Use a smooth, pose-dependent curve that describes
> the globe's rounded front, not a straight chord and not a traced eyelid edge.
> FIRST locate both canthi: the pointed junctions where the upper and lower
> eyelid margins meet at opposite ends of the eye opening. The very first and
> last pixels of the arc must touch those actual junctions when visible.
> Do not stop short, extend onto skin, terminate at the iris boundary or use
> a crop edge instead of a visible eye corner. Endpoint anchoring takes priority
> over decorative curvature. Do not force the arc through the pupil or glints.
> Do not repeat one template arch across the tiles.
> Between the corners, keep the curve inside the open eye, roughly midway
> between the upper and lower lid margins, with visible eye tissue on both sides
> of the curve. Do not let the apex hug the upper lash line.
> Continue smoothly over the iris and pupil without dipping into the pupil.
> If a lateral endpoint or edge is absent or hidden, extrapolate the same smooth
> arc toward where that feature would be. Such missing parts are an illustrative
> continuation only; do not invent or repaint underlying anatomy. Keep the
> line within its own photograph. One continuous thin pink arc only: no radial
> spokes, well, waterfall, funnels, multiple meridians, rings, dots, arrows,
> text, glow or shaded surfaces. Preserve every source eye, pupil, glint,
> eyelid, blur, exposure and texture. Add only the pink line. Preserve exact 4x3 layout
> with no added gutters, borders, cropping or shifts. Output the entire sheet in
> approximately 1.97:1 landscape aspect ratio.

The first trial often started or ended away from the eye corners. The explicit
canthus-anchored prompt variation is saved as
`outputs/globe-meridian-16-native-v1/canthus-anchored-prompt-v2.txt`.
Review endpoint placement separately from smoothness and pixel preservation:
neither of the latter validates anatomical anchoring.
The v2 trial drifted toward the upper eyelid. The subsequent
`canthus-anchored-prompt-v3.txt` explicitly requested an interior centerline.
Its composite is `outputs/globe-meridian-16-canthus-composited-v3/`.
It restored a shallower interior arc, but several endpoints still meet the ROI
boundary and precise canthus anchoring remains unverified. Preserve this
limitation; do not describe the prompt variation as an anatomically validated fix.

After generation, reuse the mask-extraction/alignment pipeline:

```bash
data/target/live/buttercup-waterfall-validate --composite \
  outputs/NEW_MERIDIAN_INPUT/contact-sheet-original.png \
  outputs/NEW_MERIDIAN_INPUT/contact-sheet-meridian-generated.png \
  outputs/NEW_MERIDIAN_INPUT/preparation-provenance.json \
  outputs/NEW_MERIDIAN_COMPOSITE
```

It produces `raw-with-pink-strokes.png`, `stroke-alpha.png`, `stroke-mask.png`,
alignment receipts and `blink-comparison.html`, alternating pink/black every
second on the unchanged original background. Generic compositor filenames are
intentional: the same machinery handles both illustration types. Inspect for
fragmented extraction, incorrect curvature and unsupported endpoint placement.
The color-derived mask may lose weak parts of the line. Record these limitations.

Use the validator with the extracted mask to verify zero changes outside it.
Its multi-ray center estimator does not apply to a single meridian: one arc
does not establish a unique gaze origin. Preservation can pass while center
extraction correctly abstains. Do not report the latter as a failed composite.
