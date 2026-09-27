# Eye anatomy, sclera support and iris sign

Visible sclera can constrain the position of a hypothetical globe and sometimes eliminate one of two iris poses. The current corpus experiments do not yet supply a reliable anatomical sclera mask. They must retain uncertain results and keep the observed evidence identical for both hypotheses.

The [multi-recording RAW video](../outputs/corpus-sign-video-20260917/corpus-sign-review.mp4) shows the frozen sign solver and a separate eyelid/white-gap heuristic. [Its report](../outputs/corpus-sign-video-20260917/README.md) documents coverage, disagreements, source identity and missing evidence. Orlosky investigation is separate; the anatomy below does not validate that method.

## What the anatomy supports

The six globe-moving muscles are the medial, lateral, superior and inferior rectus muscles and the superior and inferior oblique muscles. The four rectus muscles attach around the globe; the obliques approach it by different paths. Dissection measurements show appreciable variation in their insertions. Their attachments are distributed force application sites, not six lines that must intersect at a mechanical hinge. [Athavale et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4582163/)

The horizontal rectus pair mainly controls inward/outward gaze; the vertical rectus pair and obliques jointly contribute to vertical and torsional movement. MRI and tissue studies show that connective-tissue pulleys redirect muscle paths and can move with gaze. A force model would therefore need muscle activation, pulley geometry and orbital support as well as insertion locations. Those variables are absent from our RAW recordings. [Demer, Oh and Poukens, 2000](https://pubmed.ncbi.nlm.nih.gov/10798641/), [Demer and Clark, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7392398/)

The geometric globe centre, an effective rotation centre and the projected iris centre are separate quantities:

| Quantity | Evidence | Consequence for this viewer |
|---|---|---|
| Globe dimensions | A CT study of 250 adults reported about 24.2 mm transverse and 23.7 mm vertical, with transverse values spanning 21–27 mm. | A 12 mm sphere is a useful scale example, not Rob's measured eye shape. |
| Horizontal effective rotation centre | In 59 right eyes making ±11.9° movements, mean depth was 15.3 ± 1.5 mm behind the corneal apex. | Use a depth prior with an explicit reference plane, not a fixed pupil-to-pivot distance. |
| Vertical effective rotation centre | The same study reported 12.5 ± 1.4 mm behind the corneal apex in its abstract. | One scalar pivot depth does not represent all movements. |
| Sideways placement | MRI of 38 orbits found effective centres about 1.0 mm lateral during abduction and 0.8 mm medial during adduction, and more than 1 mm anterior to globe centroid. | Do not force the pivot to be exactly horizontally centred. |

Sources: [Bekerman et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4238270/), [Ohlendorf et al., 2022](https://onlinelibrary.wiley.com/doi/full/10.1111/opo.12940), [Demer and Clark, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7392398/). These studies use different movements and measurement methods; their summaries are not a single joint population distribution or hard bounds on Rob. Ohlendorf's setup restricted head movement and did not establish transverse pivot coordinates. The MRI results show gaze-dependent motion, so exact vertical/horizontal centring is not justified either.

Depth behind corneal apex must not silently become depth behind the fitted limbus. With distances positive posteriorly, `d_from_iris = d_from_apex - z_iris_from_apex`; the second term needs independent measurement or stated uncertainty. A camera-observed iris, entrance pupil, external limbus and geometric globe section are not interchangeable. OCT measurements also show that external topographic limbus dimensions and visible white-to-white boundaries differ. [Llorens-Quintana et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37827941/)

The existing 10–50 mm candidate-depth search is an engineering range, not established eye anatomy. The reviewed measurements do not justify treating a centre 50 mm behind the iris as a normal fixed ocular hinge. A far inferred centre may instead absorb head motion, registration error, bad conics or a mismatch between effective and geometric centres.

## What 2D eyelid overlap can and cannot determine

Let both 3D iris hypotheses project to the same ellipse `E`, and let the observed eyelid opening be the same pixel set `L`. The visible iris set is `E ∩ L` for both. Upper/lower occlusion alone therefore does not distinguish the two poses without an additional relationship between lid position, globe position and gaze. It can provide an empirical cue, but a rule such as bottom occlusion implies downward gaze is not a geometric proof.

The lid is also mobile. Measurements distinguish lid movements associated with saccades from blinks; an eye-opening boundary is not fixed facial geometry. [Evinger, Manning and Sibony, 1991](https://pubmed.ncbi.nlm.nih.gov/1993591/)

In the video, mask-overlap preference and temporal choice agree on only 15 of the 250 exposures where both choose. This is disagreement, not accuracy: these clips lack independent sign labels, and masks can include skin. The preference is strongly biased toward image-down. Its short history is explicitly dated and never counted as fresh evidence.

## The additional constraint from exposed sclera

For camera-facing iris hypotheses `(C_j, n_j)`, a spherical model with iris radius `r` and globe radius `R > r` gives

```text
d = sqrt(R^2 - r^2)
G_j = C_j - d n_j
```

Here `G_j` is the geometric sphere centre. It is not the effective rotation pivot from the studies above. Both spheres contain their associated iris circles, but their projected outer silhouettes usually differ.

Let `q(p)` be the unit forward camera ray through a pixel independently identified as exposed sclera. Assuming the camera is outside the sphere, a necessary containment condition is

```text
q(p) · G_j > 0
||G_j - (q(p) · G_j) q(p)|| <= R
```

This condition uses actual visible surface support. A globe silhouette may extend behind skin or lids; those hidden regions are not evidence against it. Conversely, a reliable sclera pixel outside every allowed silhouette contradicts the candidate under the stated shape and imaging model. One-sided exposed sclera can suffice; visibility on both sides is not required.

For uncertain parameters `theta`, the joint question is whether there exists one parameter setting that contains all supported pixels, allowing their stated uncertainty:

```text
exists theta in allowed_support:
    for all reliable sclera pixels p, contains(G_j(theta), p)
```

Do not give each pixel a different radius or centre. Do not extract a different mask for each sign. A finite nuisance grid is a sensitivity experiment, not proof that every continuous shape or camera calibration has been excluded. Probabilities require an observation/noise model beyond counting included pixels.

An illustrative orthographic example makes the distinction concrete. Let `r=6 mm`, `R=12 mm`, and tilt be ±30° horizontally. Both iris projections have semi-axes 5.196 and 6 mm. The two globe centres lie at projected x=−5.196 and +5.196 mm. A visible sclera point at x=14 mm lies 19.196 mm from the first centre and 8.804 mm from the second: the first 12 mm sphere cannot contain it, and the second can. Even permitting radii from 11 to 13.5 mm and a 2 mm lateral centre allowance, the first silhouette extends right only to 9.453 mm in this example. These are declared model assumptions and a synthetic demonstration, not measurements of Rob. [Diagram](../outputs/sclera-anatomy-20260918/containment-example.svg)

## Implementation and validation boundaries

`src/bin/buttercup_sclera_visibility.rs` evaluates common RAW-color sample pixels against both native globe hypotheses over shared shape settings. It preserves the input conic; it does not manufacture area stability. Results and source-matched pictures live in [the sclera visibility report](../outputs/sclera-visibility-20260918/README.md).

The first 100-exposure check uses two recordings, both eyes and five five-exposure neighborhoods per recording (four recording/eye streams). Strict color evidence with an iris-axis sampling window supports 69 exposures but excludes neither sign across the sampled shapes; 31 lack enough evidence. Looser thresholds produce nine conditional preferences that disappear under strict thresholds. Visual inspection finds eyelid/corner contamination and missed pink-looking sclera.

A matched camera-horizontal window removes the upper-lid contamination at source 1034. Its strict variant has 14 conditional A preferences, 51 unresolved exposures and 35 with insufficient evidence. Source 849 eye 2 has best-grid containment A=99.86% versus B=86%; source 876 eye 1 has A=100% versus B=71.64%. The newly selected patches look more consistent with exposed sclera, but loose thresholds still admit false corner regions. These are useful conditional contrasts, not measured sign accuracy. All 100 RAW hashes and fitted ellipse centres/major radii match exposures in the original video exactly; no conic was changed to improve the comparison. [New conditional cases](../outputs/sclera-visibility-20260918/horizontal-conditional-cases.png), [remaining failures](../outputs/sclera-visibility-20260918/horizontal-loose-failures.png).

The [54-second sclera comparison video](../outputs/sclera-visibility-20260918/video/sclera-hypotheses.mp4) shows all 100 exposures in twenty explicitly separated neighborhoods at four times their source duration. It places the same selected points over both candidates, displays geometric centres and labels conditional outcomes. The displayed silhouette uses a nominal 12 mm sphere; reported containment also considers the other stated shape settings. [Timing and provenance](../outputs/sclera-visibility-20260918/video/timing-provenance.json) distinguish observed intervals, terminal display holds and montage cuts.

On these same 100 exposures the frozen temporal method chooses 41 times. The strict horizontal sclera cue chooses 14 times: nine occur when the temporal method abstains, and the five overlaps agree. All 14 preferences also persist under the looser horizontal color threshold. This is a conditional support comparison, not evidence of nine correct recoveries. The [comparison audit](../outputs/sclera-anatomy-20260918/matched-choice-audit.json) matches physical normals after camera-Z convention conversion and branch permutation; comparing the literal A/B labels would be unsafe. Maximum paired-normal coordinate discrepancy is 3.02e-14. All 80 adjacent pairs within the five-exposure neighborhoods preserve their branch identities.

SN-FEIDA remains sign-invariant for these unchanged input ellipses. This work adds no independent scale or reviewed sclera/sign labels. Any eventual shared pipeline cue should retain both candidates when evidence is weak, distinguish geometric globe support from movable rotation-pivot support, keep source clocks and current/retained evidence separate, and propagate conic, camera, surface and segmentation uncertainty before selecting a sign.
