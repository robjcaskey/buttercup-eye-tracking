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
import itertools
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"count": 0}
    return {"count": len(values), "median": statistics.median(values),
            "p95": values[round((len(values)-1)*0.95)], "maximum": values[-1]}


def adjacent_area_steps(timeline):
    """Reports emit pairs before pending singletons, not in exposure order."""
    pixel_steps,normalized_steps=[],[]
    previous=None
    for clock,timestamp,sequence,dimensions,radii,normalized in sorted(timeline,key=lambda r:(r[0],r[1],r[2])):
        current=(clock,timestamp,sequence,dimensions,radii,normalized)
        if previous and radii and previous[4] and clock==previous[0] and dimensions==previous[3] \
                and 0<timestamp-previous[1]<=500_000_000 and sequence!=previous[2]:
            pixel_steps.append(tuple(abs(2*math.log(r/previous_r)) for r,previous_r in zip(radii,previous[4])))
            if all(v is not None and v>0 for v in normalized+previous[5]):
                normalized_steps.append(tuple(abs(math.log(v/previous_v)) for v,previous_v in zip(normalized,previous[5])))
        # Missing/rejected observations explicitly break the chain, including
        # when that single-ROI read was emitted after the complete pairs.
        previous=current
    return pixel_steps,normalized_steps


def check_shared_target_contract(result):
    """Check emitted geometry consistency, not anatomical/gaze accuracy.

    Older exports did not declare modeled ROIs; do not infer that declaration
    from contribution flags (an outlier-only modeled ROI is different).
    """
    if not result.get("available") or "modeled_eyes" not in result:
        return
    modeled=result["modeled_eyes"]
    penalties=result["unlocalized_eye_cost"]
    if len(modeled)!=2 or any(type(v) is not bool for v in modeled) or len(penalties)!=2 \
            or any(not math.isfinite(v) or v<0 for v in penalties):
        raise ValueError("invalid modeled-ROI declaration or unlocalized evidence cost")
    target=result["target_camera_mm"]
    if len(target)!=3 or not all(math.isfinite(v) for v in target):
        raise ValueError("shared target is not a finite 3D point")
    for eye in range(2):
        center=result["eye_centers_camera_mm"][eye]
        ray=result["eye_gaze_directions"][eye]
        ellipse=result["outer_ellipses"][eye]
        if not modeled[eye]:
            if center is not None or ray is not None or ellipse is not None or result["contributing_eyes"][eye]:
                raise ValueError("unlocalized ROI exported fabricated geometry or an admitted observation")
            continue
        if penalties[eye]!=0 or center is None or ray is None or len(center)!=3 or len(ray)!=3:
            raise ValueError("modeled ROI has missing geometry or an unlocalized penalty")
        delta=[t-c for t,c in zip(target,center)]
        distance=math.sqrt(sum(v*v for v in delta))
        if not math.isfinite(distance) or distance<=0 or not all(math.isfinite(v) for v in ray) \
                or max(abs(v/distance-r) for v,r in zip(delta,ray))>1e-7:
            raise ValueError("exported gaze ray does not point to the single shared target")


def matched_algorithm_report(baseline, candidate, index_ranges):
    """Stream exact source-matched rows; never silently compare different probes.

    This contract is for optimizer-only A/B trials with the same extraction.
    Arc index/type/count equality is checked; callers must also keep extraction
    and withheld coordinates unchanged (the old exports predate probe hashes).
    """
    coverage=[collections.Counter(),collections.Counter()]
    residuals=[[],[]]
    supported=[[],[]]
    area_steps=[[],[]]
    normalized_steps=[[],[]]
    timeline=[[],[]]
    regressions=[]
    probe_verification=collections.Counter()
    rows=0
    def in_scope(row):
        return not index_ranges or all(any(lo<=item["index"]<hi for lo,hi in index_ranges)
            for item in row["inputs"] if item)
    with baseline.open() as first,candidate.open() as second:
        for line_a,line_b in itertools.zip_longest(first,second):
            if line_a is None or line_b is None:
                raise ValueError("baseline/candidate source-read counts differ")
            a,b=json.loads(line_a),json.loads(line_b)
            if a["inputs"]!=b["inputs"]:
                raise ValueError("baseline/candidate source identity or row order differs")
            if not in_scope(b):
                continue
            rows+=1
            for eye,source in enumerate(b["inputs"]):
                if source is None:
                    continue
                fits=[row.get("joint",{}) for row in (a,b)]
                admitted=[fit.get("available",False) and fit["contributing_eyes"][eye] for fit in fits]
                coverage[eye][f"baseline:{admitted[0]},candidate:{admitted[1]}"]+=1
                frame=source["frame"]
                dimensions=tuple(frame.get(k) for k in ("width","height","stride"))
                observed_outer=all(admitted) and all(any(a["roi"]==eye+1 and a["kind"]=="OuterLimbus" and a["used"]
                    for a in fit.get("support",[])) for fit in fits)
                radii=[fit["outer_ellipses"][eye]["major_radius"] for fit in fits] if observed_outer else None
                normalized=[fit["sn_feida_mm2"][eye] for fit in fits] if observed_outer else None
                timeline[eye].append((source["clock_lineage"],int(frame["timestamp_ns"]),int(frame["sequence"]),dimensions,radii,normalized))
                if not all(admitted):
                    continue
                probes=[fit["withheld_sample_residuals"][eye] for fit in fits]
                if all(p and p["rms_px"] is not None for p in probes):
                    keys=lambda p:[(g["arc"],g["group"],g["kind"],g["points"]) for g in p["groups"]]
                    if keys(probes[0])!=keys(probes[1]):
                        raise ValueError("A/B withheld probe structure changed; optimizer-only comparison is invalid")
                    for ga,gb in zip(probes[0]["groups"],probes[1]["groups"]):
                        if ga.get("sample_fingerprint") and gb.get("sample_fingerprint"):
                            if ga["sample_fingerprint"]!=gb["sample_fingerprint"]:
                                raise ValueError("A/B withheld coordinates changed despite matching counts")
                            probe_verification["coordinate_fingerprints_matched"]+=1
                        else:
                            probe_verification["legacy_structure_only_checks"]+=1
                    residuals[eye].append(tuple(p["rms_px"] for p in probes))
                    common=[(ga,gb) for ga,gb in zip(probes[0]["groups"],probes[1]["groups"])
                        if ga["used"] and gb["used"]]
                    if common:
                        count=sum(ga["points"] for ga,gb in common)
                        supported[eye].append(tuple(math.sqrt(sum(pair[i]["rms_px"]**2*pair[i]["points"] for pair in common)/count) for i in (0,1)))
                    regressions.append({"eye":eye,"index":source["index"],"baseline_rms_px":probes[0]["rms_px"],
                        "candidate_rms_px":probes[1]["rms_px"],"delta_rms_px":probes[1]["rms_px"]-probes[0]["rms_px"]})
    for eye in range(2):
        area_steps[eye],normalized_steps[eye]=adjacent_area_steps(timeline[eye])
    def comparison(pairs):
        return {"matched":len(pairs),"baseline":distribution([a for a,b in pairs]),
            "candidate":distribution([b for a,b in pairs]),"candidate_minus_baseline":distribution([b-a for a,b in pairs]),
            "improved_over_1px":sum(b<a-1 for a,b in pairs),"regressed_over_1px":sum(b>a+1 for a,b in pairs)}
    return {"baseline":str(baseline),"candidate":str(candidate),"matched_reads":rows,"index_ranges":index_ranges,
        "probe_verification":probe_verification,
        "eye_admission":coverage,"all_withheld_samples":[comparison(p) for p in residuals],
        "common_accepted_arc_samples":[comparison(p) for p in supported],
        "frontal_equivalent_pixel_area_log_steps":[{"matched":len(p),"baseline":distribution([a for a,b in p]),
            "candidate":distribution([b for a,b in p])} for p in area_steps],
        "independent_SN_FEIDA_log_steps":[{"matched":len(p),"baseline":distribution([a for a,b in p]),
            "candidate":distribution([b for a,b in p])} for p in normalized_steps],
        "largest_regressions":sorted(regressions,key=lambda r:-r["delta_rms_px"])[:30],
        "limitations":["Optimizer-only comparison: unchanged extraction/withheld coordinates must be verified separately for old exports lacking probe hashes.",
            "Pixel-area steps are unnormalized, include genuine motion, and do not establish physical area stability.",
            "Short adjacent same-clock intervals only; no bridging absent fits or treating held source exposures as new.",
            "Limbus localization and gaze/pose ground truth remain separate post-fit validations."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--index-range", type=int, nargs=2, action="append", default=[], metavar=("START", "END"),
        help="optional source-index intervals for the separate optimizer A/B report; END is exclusive")
    args = parser.parse_args()
    methods = ("joint", "monocular_right", "monocular_left")
    summary = {m: {"available": 0, "unavailable": collections.Counter(),
                   "contributing_eyes": collections.Counter(), "costs": [],
                   "modeled_eyes": collections.Counter(), "unlocalized_eye_costs": [], "hypotheses": [],
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
                check_shared_target_contract(result)
                aggregate["contributing_eyes"][str(result["contributing_eyes"])] += 1
                aggregate["modeled_eyes"][str(result.get("modeled_eyes","legacy-unspecified"))] += 1
                aggregate["unlocalized_eye_costs"].append(sum(result["unlocalized_eye_cost"]) if "unlocalized_eye_cost" in result else None)
                aggregate["hypotheses"].append(result.get("hypotheses"))
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
        for key in ("costs", "elapsed_ms", "alternative_margins", "unlocalized_eye_costs", "hypotheses"):
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
    if args.baseline_evaluation:
        report["matched_algorithm_comparison"]=matched_algorithm_report(args.baseline_evaluation,args.evaluation,args.index_range)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scope": report["scope"], "withheld_sample_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
