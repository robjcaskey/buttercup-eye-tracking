# Pupil aperture center and sight-line review

Press **F** to select **PINK WELL + SIGHT LINE** with SAM3.1 or Buttercup
Obelisk. It is preview view 9/11 for SAM and 12/15 for Obelisk, immediately
after tweaked contact geometry. Linked Views has the same entry. **V** still
changes the source-pixel rendering. Global/selected-preview overrides work as
for the other F views.

`BUTTERCUP_VOID_SIGHTLINE=1 scripts/run-viewer.sh --segmentation sam31` selects
this view at startup. The legacy `BUTTERCUP_NESTED_PUPIL_COMPARE=1` launch flag
also selects it; neither flag overrides later F navigation or non-mask methods.
Normal camera cooperation remains mandatory. This does not change sign
selection, calibration or mouse authority.

Twelve thin pink meridians follow great-circle arcs across one illustrative
spherical surface, then make a sharp 90-degree inward turn at the pupil rim.
The surface and drop have constant opacity: there is no depth fade or rounded
lip transition. The inward direction is the local sphere normal at each rim
point, so it is perpendicular to the approaching surface tangent. Depth is an
illustrative 1.8 pupil radii. The front rim occludes the near wall, and the
shaft is clipped to the photographed aperture. Antialiased coverage is
composited once, preserving photo pixels outside the strokes.

The flat circle pose still supplies the pupil rim and orientation. A second
rendering step places a sphere through that rim. Its initial radius is 1.5
times the largest outer-boundary radius in the flat pupil plane; this is an
illustration choice, not measured eye curvature. Outer camera rays intersect
the sphere so both ends stay registered to the original image boundaries.
If needed, the common radius increases by 1.25, with at most eight attempts,
to obtain front-surface intersections without moving the fitted endpoints.
Each arc lies in a fixed plane through the common sphere center, with no
spiral. Failure to obtain those intersections omits the well. Offline records
include the displayed meridian count and illustrative radius in `well_surface`.

The view draws one coherent well using the pink normal, labelled `WELL:
ILLUSTRATIVE / UNSIGNED`. The drops point into the illustrative sphere; the
sight line remains the outward normal of the fitted circle. Both thin dashed axes and small projected
3D circle-center dots remain visible. Pink is the more upward normal in shared
camera coordinates; if their Y components differ by at most 1e-6, the more
leftward normal is pink. Cyan is the other exact pose. This is a display
convention, not binocular correspondence or evidence that pink is correct.
The illustration does not choose or change the gaze sign. Overlaying two
opposing shafts in one pupil made the earlier display's depth cue incoherent.

Both fits are needed for the well; missing pupils still permit the existing
iris-only axis fallback. Stroke pixels are clipped by the selected detector
mask when available and shaft strokes by the fitted aperture. Cached review
records without masks cannot establish eyelid occlusion. Fits may still be
wrong, and an attractive well is not independent localization evidence.

The preview decodes the exact RAW allocation retained with the geometry proposal,
not the newest camera frame. Its source sequence remains visible. Pink and cyan
show both circle-pose hypotheses: projected circle centers and dashed outward
normals of length two circle radii. If the pupil fit is missing, the outer iris
fit supplies these hypotheses, labeled `IRIS AXIS ?` and
`PUPIL OPENING: NO ACCEPTED FIT`. It never invents a pupil from the iris. Live
outer proposals may also have failed downstream admission; these are review
fits, not approved gaze measurements. The offline corpus comparison restricts
outer fits to previously admitted source-matched candidates.
The dots can overlap. Gray outlines are fitted conics, not independently
measured curves.
The well view uses the shared camera-direction colour convention above. The
older axes-only/nested offline diagnostics retain their local A/B solver order.
Neither is a persistent tracked branch identity; near a horizontal tie colours
can change as the directional convention switches between the two poses.

These are circle centers in the aperture plane, not retina centers or globe
rotation centers. The physical scale/depth is undetermined; coordinates use a
unit circle radius. Projection uses the existing configured pinhole intrinsics,
which default to an uncalibrated engineering approximation. Refraction and the
optical-to-visual-axis offset are absent. Both hypotheses fit the same ellipse;
neither drawing nor reprojection agreement supplies independent sign evidence.
A pupil is not logically required for sign selection: an independently justified
head-relative gaze range could exclude one outer-iris pose. Image bounds cannot
do this because both poses project to the same ellipse. This diagnostic has no
measured head-relative range and applies no additional range-based sign filter.

Replay existing recorded measurements without loading or training a model:

```sh
cargo build --profile live --no-default-features --bin buttercup-eye-viewer
data/target/live/buttercup-eye-viewer --offline-void-sightline \
  outputs/calibration-sam-comparison-20260916/sam-worker.jsonl \
  outputs/NEW_VOID_REVIEW
```

The output directory must be new. `report.json` retains every input record,
missing/rejected cases and both projections; `review.ppm` shows twelve evenly
spaced exposures, including failures. Set `BUTTERCUP_VOID_PINK_WELL=1` to draw
the new F-view well; without it the previous axes-only review remains available.
The renderer does not reuse these predictions as training material. The frozen
measured pupil and outer-limbus
fits are unchanged, so this presentation-only comparison does not estimate new
SN-FEIDA, human localization error or physical sign accuracy.

September 21 check: 346 of 446 Rob-only recorded SAM/RAW pupil cases produced
two projections. Ninety-eight had no pupil fit and two could not be projected.
Maximum conic roundtrip error was 1.63e-9 pixels; median projected-center versus
ellipse-centroid offset was 0.137 pixels, maximum 0.696 pixels. These are model
consistency checks, not image localization precision. Tests cover known 3D
circle recovery, both conic branches, ROI origin invariance and invalid input.
The initial live connection attempt timed out; no live sign improvement has
been demonstrated. Artifacts: `outputs/void-sightline-20260921/`.

## F-view verification (September 22)

The SAM-enabled viewer builds; 33 focused projection, source-registration,
view-cycle and workspace checks pass, as does the source-tree audit. Tests
check the perpendicular inward drop, lack of azimuthal twist, mask clipping,
missing fits, both detector routes, and unchanged gaze/clock state after ROI
movement. A broader workspace test run also exposes the separate minimum-window
stereo-toggle assertion in
`stereo_ui_layout_layers_and_pending_states_render_without_changing_analysis`;
this is not a claim that every viewer test passes.

Matched axes-only/well reviews retain identical source and geometry records on
all 446 Rob-only cached SAM frames: 346 pupil-axis, 26 iris-only and 74 unavailable
displays. Twelve evenly spaced source RAW overlays were inspected. A second
review covers the 208 native RAW frames from the September 21 live clip, with
66 pupil-axis, 102 iris-only and 40 unavailable displays. Its inspected sheet
keeps neighboring paired exposures, missing fits and the known wrong pupil
components near reflections at sequences 576/626. Those localization failures
remain visible in the new view. No new fit, sign, SN-FEIDA, or human-localization
accuracy claim follows from this display change; Obelisk routing is covered by
source-preserving renderer tests, not a new model/corpus inference run.

Artifacts: `outputs/pink-well-view-20260922/`, including matched full reports,
`final-well/review.png`, `live-clip/review.png`, and build/test/audit logs. No
camera session was running during this change; the rebuilt view is ready for
the next viewer launch. Saved calibration files are unchanged.

The subsequent live review identified a sunburst appearance: straight bright
spokes dominated the short, immediately fading shaft. The first rounded-lip revision
is under `outputs/pink-well-rework-20260922/`. Its 446 source
and geometry records match the previous renderer exactly; the twelve matching
RAW tiles were visually compared. The extended geometric checks exercise lip
registration, radial-plane confinement, smooth entry into the perpendicular
shaft, fading and mask occlusion; source-reframing checks cover both detectors.
This corrects presentation, not the underlying pupil-fit failures.

The paired-eye colour report was also reproduced: all 158 drawable paired
exposures in this 446-frame cache had opposite local up/down order between
eyes. At sequence 1734, solver slot zero has camera-normal Y +0.906 for the
right eye and -0.339 for the left; slot one reverses those directions. Sorting
only for display gives both eyes the same colour meaning. A regression test
uses those exact paired ellipses and equivalent angle representations. The
original two physical poses, source records and all fit/admission decisions
remain identical; offline reports export `display_color_candidate_indices`
separately so the relabelling is auditable. The paired RAW review includes
missing fits and neighboring exposures. This fixes the systematic colour
reversal in this subset without claiming sign accuracy or identical eye normals.

The spherical-surface revision replaces that rounded lip with great-circle
arcs and the sharp, non-fading drop described above. Eleven focused projection
and source-preservation tests pass, including common-sphere membership,
registration at both boundaries, the right-angle turn and constant opacity.
The source-tree audit passes. Matched reviews retain every original fit, pose,
source timestamp and display-colour assignment across 446 cached Rob SAM frames
and every original fit/pose/source across the additional 208-frame live clip.
All 331 and 64 respective frames with both required fits produce twelve arcs;
the 115 and 144 missing-pair cases remain missing. Paired and neighboring RAW
tiles were inspected. The known wrong pupil fits near reflections at sequences
576/626 remain wrong, and mask-free cache reviews do not validate eyelid
occlusion. This adds no human localization or independent scale/sign evidence;
SN-FEIDA inputs and gaze/calibration authority are unchanged. Artifacts:
`outputs/pink-well-spherical-20260922/`, including source-matched reports,
paired/neighboring contact sheets, a before/after loop and comparison checks.

## Joint outer-iris / inner-pupil comparison

On the offline command, `BUTTERCUP_NESTED_PUPIL_COMPARE=1` enables a
source-matched preview with
two predicted pupil outlines, center dots, dashed axes and pixel RMS errors.
Pink A and cyan B now refer to the **outer iris's** two pose hypotheses.
Gray is the observed fitted pupil and outer outline. The colored pupil curves
are constrained predictions; they must not be mistaken for measured boundaries.
This diagnostic has no gaze, calibration or mouse authority.

For each fixed outer-circle pose the code unprojects the same observed pupil
conic into a candidate plane, fits its circle, and refines radius and in-plane
offset against image-space residuals with an equal bounded budget. It does not
move the outer fit to make a candidate win. Sampling the fitted pupil at 48
points is numerical quadrature, not 48 independent observations.

The comparison reports all twelve combinations of maximum decentration
0/0.05/0.10/0.20 outer radii and inward depth 0/0.10/0.20 outer radii. Radius is
bounded to 0.08–0.80 outer radii and fits remain contained in the modeled disk.
Nominal is depth zero, decentration 0.10. These are declared engineering probes,
not measured anatomy. A conditional preference requires a winning RMS <=2 px
and margin >=1 px, with the same winner for every nonzero-decentration variant.
Exact concentricity is reported as a brittle baseline, not enforced on real eyes.
Corneal refraction, lens distortion and individual pupil decentration remain
unmodeled. No confidence probabilities are claimed.

September 21 results on 446 source-matched Rob-only exposures from the saved
SAM replay in the command above:

| Input / focal pixels | Compared pairs | Conditional preferences |
| --- | ---: | ---: |
| Cached pupil, 4000 | 331 | 0 |
| Cached pupil, 2000 | 39 | 0 |
| Cached pupil, 8000 | 332 | 0 |
| Fresh RAW free-shape pupil, 4000 | 18 | 0 |

At nominal intrinsics, 114 rows lack an admitted cached pair and one has invalid
geometry. Five pairs clear the nominal error/margin thresholds, but none survives
the offset/depth sensitivity check. Median best nominal RMS is 9.73 px. The
cached comparison includes 135 of 332 available pairs whose pupil angle and axis
ratio exactly match the outer ellipse. The worker has a RAW-checked temporal
recovery that imposes this shape, so agreement is not independent shape evidence.

`BUTTERCUP_VOID_REFIT_PUPIL=1` on the offline command instead uses fresh CPU RAW
component fitting with search-domain censoring, retained-arc constraints and
independent pupil shape. It never falls back to the cached pupil. Only 18 frames
return a pupil fit; 428 lack a new admitted pair. Median best nominal RMS is
11.58 px on those 18. This poor coverage and model mismatch do not establish that
nested geometry is useless, but do not justify switching the live gaze sign.
Optional `BUTTERCUP_VOID_REVIEW_INDICES` is a JSON list of up to twelve source
row indices for inspecting successes alongside neighboring exposures.

Artifacts are under `outputs/nested-pupil-20260921/`: full per-source/per-variant
reports, RAW sheets and build/test logs. Outer geometry, existing gaze authority
and therefore baseline outer-area/SN-FEIDA inputs remain unchanged. There are no
independent sign labels or human pupil localization labels in this experiment.
Synthetic controls verify both true tilt signs under exact perspective,
near-frontal/weak-perspective abstention and bounded decentration fitting.

## Pupil rejection debugging

The 18-fit result above was an acceptance-policy result, not a count of visibly
identifiable pupils. A matched September 21 ablation kept the same 446 sources,
358 admitted outer fits, RAW threshold/components, search-domain censoring,
independent pupil shape and downstream RAW/geometry gates. Only the contour
policy changed:

| `BUTTERCUP_VOID_PUPIL_POLICY` | Tangent constraints | Final support rule | Accepted pupils | Original 18 lost |
| --- | --- | --- | ---: | ---: |
| `strict` (default) | yes | at least 48 retained samples | 18 | 0 |
| `count-only` | no | at least 48 retained samples | 137 | 3 |
| `tangent-position` | yes | positional coverage/conditioning | 157 | 0 |
| `position` | no | positional coverage/conditioning | 137 | 4 |

Use these with `BUTTERCUP_VOID_REFIT_PUPIL=1`. All remain offline diagnostics;
no live acceptance or sign policy was promoted. `BUTTERCUP_VOID_PUPIL_FITS_ONLY=1`
draws the actual 2D fitted pupil in pink, fitted outer limbus in cyan, retained
contour samples in green and excluded samples in orange. This separates pupil
localization from the previous predicted 3D boundaries. Reports include
provisional ellipses, retained counts, exact contour rejection stages and the
downstream geometry gates, including when no pupil is admitted.

Concrete source-matched examples from the original twelve-tile review:

- Row 58 / sequence 1743: 39 retained samples failed the 48-sample floor;
  positional support admits the fit without dropping tangent constraints.
- Row 24 / sequence 1726: constrained contour fitting failed; the `position`
  variant locates the pupil with 45 retained samples.
- Rows 264 and 308 / sequences 1846 and 1868: free-shape contour fits pass
  positional support, but normalized center offsets 0.375 and 0.372 exceed
  the unprompted acquisition limit 0.35. They pass the broader geometry gate.
- Row 286 / sequence 1857: the unconstrained fit extrapolates to a displaced
  ellipse; positional support correctly refuses that candidate.
- Row 430 / sequence 1929: no admitted outer fit, so this outer-dependent
  pupil diagnostic never attempts an inner fit despite a visible dark pupil.

Inspecting twelve evenly spaced **newly accepted** `tangent-position` results
found visible wrong-component fits at rows 165 and 403 (sequences 1796 and 1915).
The centrality restriction can reject the actual pupil while accepting a
central iris fragment. Thus increased acceptance is not measured accuracy and
does not justify promoting the variant. Other fitted boundaries still need
localization review, and there are no canonical human pupil labels in this run.
Outer fits, independent scale inputs and SN-FEIDA inputs remain frozen; this
experiment does not establish gaze-sign accuracy or cross-user robustness.

The earlier pink-waterfall prompt was also repeated on the same twelve RAW
exposures. Pink-only compositing preserves the original photographs outside
the extracted stroke mask. The generator broadly locates the dark apertures,
but strokes extend beyond some iris boundaries; they are a visual aid, not
anatomical measurements or training labels.

Artifacts: `outputs/pupil-fit-debug-20260921/`, including per-policy full
reports, `recovered/review.png`, `position/review.png`, and
`pink-composite/raw-with-pink-strokes.png`. The relevant pupil unit tests and
source-tree audit pass. A broader pupil-filtered library test run encounters an
unavailable legacy RAW fixture at `../outputs/paired-eye-reverse-stereo-20260809T113752Z/`;
this is separate from the matched 446-frame replay.

## Live clip and iris-only display correction

The September 21 live session recorded 104 paired exposures (208 native
420x280 RAW10 frames) in `outputs/live-pupil-check-20260921/live-eyes-10s.tar`.
Fresh production SAM sequence replays from empty memory used the recorded
source times and ROI origins, without seeding from recorded predictions.
Completion-paced replay is not the live asynchronous/drop-frame schedule and
does not measure live latency. GPU work shared resources with the live viewer.

| Source eye | Exposures | RAW-admitted outer fits | Pupil fits | Fresh semantic pupils | Prior recoveries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Right | 104 | 68 | 39 | 17 | 22 |
| Left | 104 | 99 | 27 | 9 | 18 |

The 66 pupil fits are acceptance results, not correct-localization counts; 40
are prior-guided recoveries, not independent shape observations. Source-matched
overlays at sequences 576 and 626 show small/off-center fits near the monitor
reflection. The left outer fit at 626 is also visibly wrong. At right sequence
536, semantic queries 49 and 28 have plausible ellipse geometry but only
0.340/0.210 positive RAW edge fractions and 3/1 strong sectors, below the
0.42/four-sector requirement. Higher-ranked queries include broad iris masks
and narrow fragments. The mask, contour, geometry, center, history and RAW
gates are exported separately by `BUTTERCUP_SAM31_PUPIL_CANDIDATE_AUDIT=1`.
Thus missing pupils cannot be fixed responsibly by lowering one count threshold.

The matched fresh RAW component experiment accepts zero pupils under the
strict independent-shape/arc policy and 17 with positional support replacing
the 48-sample floor. Inspection includes a false fragment below/right of the
reflection at left sequence 626. This policy remains offline. Repeating the
earlier 446-source positional-support run gives exactly the same per-source
fits and RAW diagnostics as `final-tangent-position`: 157 fitted pupils, of
which 145 produce two valid circle projections. Fit counts and projection
availability must be kept separate.

The shipped correction changes only the diagnostic display. It can show the
outer iris's two unsigned poses when the pupil fit is missing, and names the
missing feature as the pupil opening. On the frozen live-clip replay it adds
102 iris-only displays to the 66 pupil displays, leaving 40 without a drawable
pose. On the earlier 446-source cache it adds 26 iris-only displays to 346 pupil
displays. No ellipse, acceptance gate, pupil prior, outer-area/SN-FEIDA input,
calibration or gaze authority changes. The 64 admitted live-clip nested pairs
remain ambiguous. There are no canonical pupil/sign labels, independently
measured head-relative gaze limits or calibrated camera intrinsics in this
review; these results do not establish sign accuracy or cross-user readiness.

Artifacts live under `outputs/live-pupil-debug-20260921/`: native capture
members, full per-eye baseline reports and candidate logs, `live-worker.jsonl`
with exact source references, `live-fits/review.png`, `live-axes/review.png`,
the RAW policy comparisons, and build/test logs. Geometry projection tests and
the source-tree audit pass. The first SAM-enabled test invocation lacked the
LibTorch shared-library path; rerunning with the installed runtime passes.
