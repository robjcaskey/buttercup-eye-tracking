# Eyewear context and reflection supervision

`buttercup_eyewear_review` saves assistant visual suggestions for recordings,
with archive/session hashes and five exact fresh sensor-band sources. Outputs
go under each capture's `annotator/labels/eyewear.assistant-review.json`.
They are **not human labels** or reflection masks. `probably-on`, `probably-off`,
`mixed` and `unknown` describe eyewear context; no calibrated probability is
claimed. Sampling cannot exclude a transition in an uninspected interval.

Initial review: six recordings from September 12, 2026, starting at 14:02:06,
14:02:12, 14:03:35, 14:04:03 (probably-off), 14:04:35 and 14:05:06
(probably-on), local time. Thirty fresh sensor-band samples were inspected.
Cached global thumbnails were excluded as evidence for the contemporaneous
state. The entire sitting is one development group, not six independent
training/evaluation groups.

## Existing non-neural candidate generators

Read-only inspection located the historical `osbot_onlens_reflection_pairs.py`
and `osbot_interlens_reflection_check.py` in the older cleanroom tools. They are
not dependencies of this repository and have not been run across this corpus.

The former enhances half-resolution RAW, self-matches SIFT descriptors,
checks corner support, clusters copy displacement through time, and seeks
consistent displacement families across eyes and crop epochs. Its output
explicitly says **front/back surface assignment is not inferred**. The latter
fits an inter-eye motion transform on one epoch and evaluates on another,
including shuffled temporal-block controls. Its motion layers remain anonymous.

Useful weak supervision is therefore *repeated-reflection candidate support*,
not true front/rear surface identity. Endpoints are sparse candidate locations,
not dense pixel masks or proof of intersection with an iris. A zero-match
result is unknown: blur, clipping, weak reflections, a missing eye or short
epochs can all remove support. Repeated iris texture and mosaic aliases are
confounders, not new positive labels. Glasses-off images can still contain
corneal glints and screen reflections.

## Gate before a full sweep or neural training

1. Port the required RAW-only matcher into maintained native Rust, without
   runtime imports from another checkout. Retain bounded temporal windows,
   exact source identities, explicit missing-eye/empty-candidate handling and
   resolution-aware displacements. No installed Student weights or recorded
   Student/SAM fits may silently supply iris masks or sample selection.
2. Compare against the historical algorithm on identical native inputs, then
   inspect candidate locations on the recent glasses-on/off controls and older
   independent sessions. Include time-shuffled and CFA/ROI-reframe controls.
   Do not tune and report final accuracy on the same six-recording sitting.
3. Sweep compatible corpus sources, reporting eligibility, support, abstention,
   malformed/missing inputs and source-time intervals. Unknown stays unknown;
   an archive-level probable state must not paint every frame as positive.
4. Define separate localized targets for paired glare, unpaired bright
   reflections, ordinary corneal glints and unknown regions. Use canonical
   human review or explicitly versioned weak-target recipes. Existing
   possibly-occluded limbus labels do not identify the occluding cause.
5. Run the bootstrap DAG preflight and cold training from permitted roots.
   Hold sessions/subjects out, keep both eyes and nearby frames together, and
   evaluate iris-overlap false positives and localization, not just eyewear
   classification. Train a shared offline model in Rust; any personal or
   scenario refinement remains CPU-only. No live fitting change before these
   checks. No reflection network has yet been trained by this review.
