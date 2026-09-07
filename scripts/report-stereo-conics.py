#!/usr/bin/env python3
"""Summarize matched conic replays, including regressions and scope shortfalls.

This evaluator does not equate cross-eye agreement or smoothness with true gaze
accuracy. Withheld points are conditional on upstream SAM segmentation and its
initial conic hypotheses. External label scoring is a separate, post-fit step.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"count": 0}
    return {"count": len(values), "median": statistics.median(values),
            "p95": values[round((len(values)-1)*0.95)], "maximum": values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()
    methods = ("joint", "monocular_right", "monocular_left")
    summary = {m: {"available": 0, "unavailable": collections.Counter(),
                   "contributing_eyes": collections.Counter(), "costs": [],
                   "elapsed_ms": [], "alternative_margins": []} for m in methods}
    heldout = [[], []]
    matched_supported = [[], []]
    arc_coverage = [collections.Counter(), collections.Counter()]
    matched_costs = []
    ids, duplicates, capture_entries = set(), [], set()
    both_raw = both_evidence = rows = 0
    independent_scale = collections.Counter()
    worst = []
    with args.evaluation.open() as source:
        for line in source:
            row = json.loads(line)
            rows += 1
            inputs = row.get("inputs", [None, None])
            both_raw += all(inputs)
            both_evidence += all(row.get("observed_arc_groups", [0, 0]))
            for item in inputs:
                if item:
                    key = item["index"]
                    if key in ids:
                        duplicates.append(key)
                    ids.add(key)
                    capture_entries.add(item["capture_entry"])
            for eye, provenance in enumerate(row.get("scale_provenance", [])):
                independent_scale[f"eye{eye}:{provenance}"] += 1
            for name in methods:
                result = row.get(name, {})
                aggregate = summary[name]
                aggregate["elapsed_ms"].append(result.get("elapsed_ms"))
                if not result.get("available"):
                    aggregate["unavailable"][result.get("reason", row.get("error", "absent-result"))] += 1
                    continue
                aggregate["available"] += 1
                aggregate["contributing_eyes"][str(result["contributing_eyes"])] += 1
                aggregate["costs"].append(result["cost"])
                aggregate["alternative_margins"].append(result["alternative_cost_margin"])
            joint = row.get("joint", {})
            if joint.get("available") and all(row.get(name, {}).get("available") for name in methods[1:]):
                matched_costs.append(joint["cost"] - sum(row[m]["cost"] for m in methods[1:]))
            for eye, name in enumerate(methods[1:]):
                mono = row.get(name, {})
                if not (joint.get("available") and mono.get("available")):
                    continue
                a = joint["withheld_sample_residuals"][eye]
                b = mono["withheld_sample_residuals"][eye]
                if not a or not b or a["rms_px"] is None or b["rms_px"] is None:
                    continue
                pair = a["rms_px"], b["rms_px"]
                heldout[eye].append(pair)
                joint_groups = {g["arc"]: g for g in a.get("groups", [])}
                mono_groups = {g["arc"]: g for g in b.get("groups", [])}
                shared = []
                for key in joint_groups.keys() & mono_groups.keys():
                    ga, gb = joint_groups[key], mono_groups[key]
                    arc_coverage[eye][f"joint:{ga['used']},monocular:{gb['used']}"] += 1
                    if ga["used"] and gb["used"] and ga["points"] == gb["points"]:
                        shared.append((ga, gb))
                if shared:
                    count = sum(ga["points"] for ga, gb in shared)
                    matched_supported[eye].append(tuple(math.sqrt(sum(pair[i]["rms_px"]**2*pair[i]["points"] for pair in shared)/count) for i in (0, 1)))
                worst.append({"delta_rms_px": pair[0]-pair[1], "eye": eye,
                              "joint_rms_px": pair[0], "monocular_rms_px": pair[1],
                              "indices": [v["index"] if v else None for v in inputs]})
    for aggregate in summary.values():
        for key in ("costs", "elapsed_ms", "alternative_margins"):
            aggregate[key] = distribution(aggregate[key])
    def comparisons_for(samples):
        return [{"eye": eye, "matched_reads": len(pairs),
            "joint_rms_px": distribution([a for a, b in pairs]),
            "monocular_rms_px": distribution([b for a, b in pairs]),
            "joint_minus_monocular_rms_px": distribution([a-b for a, b in pairs]),
            "improved": sum(a < b-1e-6 for a, b in pairs),
            "regressed": sum(a > b+1e-6 for a, b in pairs),
            "regressed_over_1px": sum(a > b+1 for a, b in pairs)} for eye, pairs in enumerate(samples)]
    comparisons = comparisons_for(heldout)
    expected = None
    if args.expected_manifest:
        expected = json.loads(args.expected_manifest.read_text())["summary"]["unique_raw_frames"]
    unexpected = sorted(i for i in ids if expected is not None and (not isinstance(i, int) or not 0 <= i < expected))
    missing_count = expected-len(ids-set(unexpected)) if expected is not None else None
    report = {"schema": "buttercup-joint-conic-comparison-v1", "evaluation": str(args.evaluation),
        "scope": {"reads": rows, "both_raw_reads": both_raw, "both_evidence_reads": both_evidence,
                  "unique_input_indices": len(ids), "capture_entries": len(capture_entries),
                  "expected_input_indices": expected,
                  "complete_corpus": expected is not None and missing_count == 0 and not unexpected and not duplicates,
                  "missing_input_count": missing_count, "unexpected_input_indices": unexpected,
                  "duplicate_input_indices": duplicates},
        "methods": summary, "withheld_sample_comparisons": comparisons,
        "matched_supported_arc_comparisons": comparisons_for(matched_supported),
        "matched_arc_acceptance": arc_coverage,
        "matched_joint_minus_sum_monocular_objective": distribution(matched_costs),
        "scale_provenance": independent_scale,
        "largest_withheld_sample_regressions": sorted(worst, key=lambda r: -r["delta_rms_px"])[:30],
        "limitations": ["No independent gaze/visual-axis ground truth in this comparison.",
                        "Common fixation, pinhole optics, coarse origin/scale and axis-alignment support are defeasible assumptions.",
                        "Withheld contour samples do not make upstream segmentation/search independent of that image.",
                        "SN-FEIDA is absent when external coarse scale is unavailable; no self-radius normalization.",
                        "Comparison against human native-RAW localization labels and sequence continuity remains separately required."]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scope": report["scope"], "withheld_sample_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
