# Rust physics and mathematics naming audit

The 2026-09-07 audit inventoried all 60 checked Rust source files then present:
168,261 lines and 45,449 explicit declarations (40,437 bindings, 4,115 fields,
870 constants, 23 associated constants, 3 statics and 1 const parameter).
Function arguments, local patterns, closure parameters, loop/match bindings,
fields and constants were collected using a Rust syntax visitor. Macro-expanded
or compiler/dependency-generated bindings are outside that census. This is a
bulk inventory plus targeted semantic review, not a claim that every declaration
has a formally verified physical interpretation.

The review prioritized names implying measured physics, metric units,
probabilities, derivatives, projected/deprojected geometry or coordinate frames.
Conventional coordinates, actual velocities/angles, genuine sum-of-squares
signal energy, and unambiguous short algebraic variables were retained.

## Important corrections

| Previous name/concept | Replacement and actual meaning |
| --- | --- |
| `angular_acceleration` in sign scoring | `angular_speed_change_step_rad`: absolute angular-speed change multiplied by the current interval, with units of angle; not acceleration |
| Similarity `rotation` | `rotation_coefficient`: off-diagonal matrix coefficient, not radians |
| Similarity `scale_delta` | `diagonal_coefficient_delta`: diagonal coefficient minus one, not exact scale change |
| `curvature_jerk` | `mean_abs_turn_angle_change_rad`: adjacent **spatial** outline turn variation, not a temporal third derivative |
| `velocity_center`, `velocity_globe_center` | `extrapolated_center_sensor_px`, `extrapolated_pivot_sensor_px`: predicted positions, not velocities |
| `saccade_likelihood`, `micro_motion_likelihood` | `saccade_score`, `micro_motion_score`: heuristic scores, not calibrated likelihoods |
| `green_relative_to_cyan` | `pupil_relative_to_general`: pupil motion in the general-image reference frame |
| `slice_depth` | `limbus_plane_offset_px`: model-space limbus plane offset, not a measured camera range |
| `rectified_area_px2`, surface-area buckets | `frontal_equivalent_disk_area_px2`, frontal-disk area bins: planar iris disk area, not curved anatomical surface area |
| Outer-edge force/power | Outer-edge localization/normalized edge amplitude: image contrast localization, not physical force or optical power |
| Texture energy | Texture-difference RMS where the calculation actually takes a square root |
| `variance_ratio` | `standard_deviation_ratio` where the calculation uses square-root eigenvalues |
| Display ray `depths` | `ray_ranges_inches`: distance along unit gaze rays, not optical-axis depth |
| Synthetic RAW iris `focal` | `gnomonic_projection_scale_px`: chosen projection scale, not calibrated focal length |

For the similarity matrix `[[1+d,-b],[b,1+d]]`, the exact rotation angle is
`atan2(b,1+d)` and scale is `hypot(1+d,b)`. Consumers that still use a small-angle
approximation have not been silently changed to those exact formulas;
`gross_small_angle_rotation_degrees` explicitly identifies one such estimate.
Physical angular acceleration would divide angular-velocity change by elapsed
time, and angular jerk differentiates acceleration again. Neither is obtained
merely by renaming the existing sign regularizer.

Constants now identify frontal-disk area bin ratios, scale-relock updates,
contact-sign confirmation updates, camera-facing normal minimum Z and edge
amplitude thresholds. Pixel-space pivots, near-surface points and normalized
position residuals identify their actual coordinate domain. An effective eye
pivot remains a defeasible, movable anatomical approximation rather than an
exact rigid hinge. See [geometry architecture](geometry-architecture.md).

SN-FEIDA remains the scale-normalized frontal-equivalent iris disk area defined
in [the area and motion reference](flat-tire-area-and-motion.md). This audit
does not reclassify pupil aperture area or visible mask area as SN-FEIDA.

## Equivalence evidence and limits

The reviewed mechanical pass changed 795 identifier occurrences across 17
files. All 1,127,055 non-comment tokens were compared before/after: only approved
identifier substitutions differed. Numeric values, string literals, operators
and control flow were unchanged. Existing serialized/JSON field names remain
compatibility aliases even where the Rust field now uses clearer terminology.
The retained audit artifacts are under `outputs/name-audit-20260907/`:
`before.json`, `after.json`, `renames.json`, `edits.json`, `edits-second.json`,
`verification.json`, and matched baseline/candidate test logs.

Both matched test runs had exactly 848 passing, 41 failing and 19 ignored tests,
with identical per-test outcomes. The 41 failures predate this rename pass;
this is **not** an all-green suite. No separate corpus replay is claimed for
the rename-only pass: the token comparison directly checks unchanged
calculations. Subsequent viewer-recording features are separately tested and
are not covered by that alpha-equivalence claim.

Actual model issues remain distinct work: symmetric sign branches can preserve
angular-speed/acceleration/jerk magnitudes; independent motion/translation
coupling is needed to distinguish them. A consistently poor pair of hypotheses
also needs an explicit inadequacy response rather than winning by relative
score alone. Naming changes do not fix those mechanisms or turn heuristic
fidelity into calibrated probability. Future geometry changes need matched
corpus evaluation, timing/coverage/localization checks and independent scale
support alongside synthetic tests, not area stability alone.
