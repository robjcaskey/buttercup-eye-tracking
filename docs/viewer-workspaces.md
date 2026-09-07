# Main-screen workspaces

The main screen answers three different questions without mixing their controls:

1. **ROI**: what is happening in this region? Each eye remembers its pixel view
   and overlay independently; selecting another region must not erase them.
2. **Linked ROIs**: how do the observations compare? Compare, timing, and contact
   views keep separate source times and explicitly report missing regions. Two
   visible eyes do not imply that the unimplemented stereo solver ran.
3. **Global**: what does the camera see, and where are the regions? Sensor overview
   and prompted object inspection belong here, not in an iris overlay cycle.

Tab changes workspace. F changes the view within that workspace. V changes image
appearance where applicable. Object search is an explicit start/stop action,
not a side effect of browsing. The selected eye's presentation is separate from
the physical lens/exposure controls, which necessarily affect the whole camera.

The shell always has scope navigation, a view/title toolbar, a bounded image
workspace, and an accessible inspector. The inspector separates selection/view,
shared analysis, whole-camera controls, and diagnostics. Narrow windows use a
bottom inspector rather than drawing text over imagery or silently dropping it.
Comma (`,`) cycles inspector tabs without needing a function key; tabs can also
be clicked. Long diagnostic text wraps and scrolls inside its panel. Images preserve aspect
ratio; absent or disabled data gets an explicit placeholder, never a frozen
unlabelled tile. Global images remain labelled snapshots, not live sensor video.

Validation required before calling the redesign complete: all scopes rendered
and inspected at desktop and compact sizes; independent ROI settings and linked
views tested; navigation does not start capture; explicit object search still
works; prompt editing/cancel and camera controls remain usable; source-time and
missing-data labels are honest; calibration, lightbox and J cursor still work;
build and source-tree audit pass. Geometry algorithms are not changed by this
presentation rework.

## Verification (September 6)

- `viewer_ui` tests render all six views at 320x240, 640x480, 1200x850,
  800x1200, and 904x2048; test panel bounds/non-overlap, independent ROI view
  state, missing/disabled data, object crops, word wrapping, and editor hotkeys.
  The desktop, compact, and actual tall-window images were visually inspected.
  Portrait ROI/context proportions and card-local scale clipping were corrected
  from those inspections. Fixtures are synthetic presentation data, not anatomy
  accuracy measurements (`outputs/ui-layout-review`).
- The live control-interface run selected left/right ROIs independently, changed
  only the left overlay, visited linked/timing/global/object views, and verified
  that autofocus reference and acquisition state were unchanged by navigation.
- A real `hat` object prompt compiled through the native encoder. Explicit search
  captured a fresh global image and returned no candidate in an empty-room scene;
  explicit stop resumed SAM fine-eye processing. Iris prompt text and generation
  remained unchanged throughout (`outputs/ui-live-acceptance.log`). This verifies
  the no-match/retry path, not generic object-detection accuracy. Positive mask
  cropping is covered by the synthetic renderer test.
- Focused overlay, SAM presentation, calibration-thumbnail, completed monitor
  wireframe, cursor, keyboard-map, and scoped control/prompt-isolation tests pass.
  Logs: `outputs/viewer-ui-tests.log`, `outputs/viewer-scoped-control-tests.log`,
  `outputs/viewer-ui-regressions.log`, and
  `outputs/viewer-ui-geometry-regressions.log`.
- The SAM-enabled live build runs on the camera stream. Source-tree audit and
  diff whitespace checks pass. Existing compiler warnings remain; this is not
  a claim that the full repository/corpus test suite is green.

Calibration remains a separate distraction-free modal screen. Its geometric
solver and acceptance gates are unchanged. Main-screen lightbox/cursor
compositing and source-aligned eye-overlay helpers are reused. The new view
layer does not claim a stereo solve, fresh observations from held data, or an
improvement to SN-FEIDA/localization accuracy.
