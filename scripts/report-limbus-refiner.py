#!/usr/bin/env python3
"""Summarize matched source-native limbus-refiner evaluations, never impute truth.

cv: each exposure appears once, exclusively from its held-out source group.
sequence: fresh adjacent corpus samples on an attested source clock, <=500ms.
SAM-only auxiliary agreement is not human localization accuracy.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95": values[math.ceil(.95 * len(values)) - 1] if values else None,
            "max": max(values) if values else None}


def summarize_cv(reports):
    frames, identities = [], set()
    for report in reports:
        train = set(report["model_training"]["train_raw_sha256"])
        for row in report["frames"]:
            if row["split"] != "test":
                continue
            identity = row["source"]["raw_sha256"]
            if identity in train or identity in identities:
                raise ValueError("duplicate held-out source or training leakage")
            identities.add(identity)
            frames.append(row)
    admitted = [r for r in frames if r["baseline_raw_admitted"]]
    matched = [r for r in admitted if r["candidate"] and r["unmodified_refit"]]
    errors = lambda rows, key: [r[key]["rim"]["rms_px"] for r in rows if (r.get(key) or {}).get("rim")]
    pairs = []
    for r in matched:
        b, c, n = [r[k]["rim"]["rms_px"] for k in
                   ("baseline_errors", "unmodified_refit_errors", "candidate_errors")]
        pairs.append({"source": r["source"], "baseline_rms_px": b,
                      "unmodified_refit_rms_px": c, "candidate_rms_px": n,
                      "delta_from_control_px": n - c})
    return {"scope": "development grouped cross-validation; whole-frame contour fixed before label scoring",
            "frames": len(frames), "groups": dict(collections.Counter(r["group"] for r in frames)),
            "baseline_raw_admitted": len(admitted), "accepted_refinement": len(matched),
            "baseline_retained_without_refinement": len(admitted) - len(matched),
            "matched_baseline_rms_px": stats(errors(matched, "baseline_errors")),
            "matched_control_rms_px": stats(errors(matched, "unmodified_refit_errors")),
            "matched_refined_rms_px": stats(errors(matched, "candidate_errors")),
            "all_admitted_delta_including_unchanged_fallback_px": stats([
                (r["candidate_errors"] or r["baseline_errors"])["rim"]["rms_px"] -
                r["baseline_errors"]["rim"]["rms_px"] for r in admitted]),
            "refine_cpu_ms": stats([r["elapsed_ms"] for r in admitted]), "matched": pairs,
            "failures": [{"source": r["source"], "status": r["status"],
                          "baseline_raw_admitted": r["baseline_raw_admitted"]}
                         for r in frames if not r["candidate"]],
            "limitations": ["Generic rim includes lower-weight band midpoints, not exclusively surface apex.",
                            "Ten RAW-admitted labels are not ten independent people or sessions.",
                            "Fallback is the original baseline, not a fresh refined observation.",
                            "Model comparisons informed development; this is not a sealed final test."]}


def sequence_pairs(rows):
    lineages = collections.defaultdict(list)
    for r in rows:
        src = r["source"]
        if not src.get("clock_attested") and src.get("clock_basis") != "exact-RAW-matched-native-motion-chain":
            continue
        f = src["frame"]
        key = (src["clock_lineage"], f["eye_id"], (f.get("region") or {}).get("session"))
        lineages[key].append(r)
    pairs = []
    for key, group in lineages.items():
        group.sort(key=lambda r: r["source"]["frame"]["timestamp_ns"])
        for before, after in zip(group, group[1:]):
            a, b = before["source"]["frame"], after["source"]["frame"]
            delta = b["timestamp_ns"] - a["timestamp_ns"]
            # Do not bridge a missing candidate or unsupported scale sample.
            if not (0 < delta <= 500_000_000 and a["sequence"] < b["sequence"]
                    and before["baseline_raw_admitted"] and after["baseline_raw_admitted"]
                    and before["candidate"] and after["candidate"]
                    and before["sn_feida"] and after["sn_feida"]):
                continue
            if (before["sn_feida"].get("scale_reference") != after["sn_feida"].get("scale_reference")
                    or before["sn_feida"].get("units") != after["sn_feida"].get("units")):
                continue
            values = {}
            modes = ["baseline", "candidate"]
            if before["sn_feida"].get("unmodified_refit") and after["sn_feida"].get("unmodified_refit"):
                modes.append("unmodified_refit")
            for mode in modes:
                p, q = [r["sn_feida"].get(mode, r["sn_feida"].get(mode + "_mm2")) for r in (before, after)]
                if not (p and q and p > 0 and q > 0):
                    break
                values[mode + "_absolute_log_step"] = abs(math.log(q / p))
            else:
                pairs.append({"lineage": key, "before": before["source"]["raw_sha256"],
                              "after": after["source"]["raw_sha256"], "dt_ms": delta / 1e6,
                              "roi_reframed": (a["sensor_x"], a["sensor_y"]) != (b["sensor_x"], b["sensor_y"]),
                              **values})
    return pairs


def summarize_sequence(report):
    frames = report["frames"]
    pairs = sequence_pairs(frames)
    refined = [r for r in frames if r["candidate"] and r["baseline_raw_admitted"]]
    return {"scope": "conditional archived-contour diagnostic; includes development sources, not a sealed final test",
            "physical_context_ablated": report["physical_context_ablated"],
            "frames": len(frames), "raw_admitted": sum(r["baseline_raw_admitted"] for r in frames),
            "refined": len(refined), "independent_scale_frames": sum(bool(r["sn_feida"]) for r in frames),
            "scale_units": dict(collections.Counter(r["sn_feida"].get("units", "coarse_mm2") for r in frames if r["sn_feida"])),
            "matched_area_pairs": len(pairs), "reframed_pairs": sum(p["roi_reframed"] for p in pairs),
            "baseline_abs_log_sn_feida_step": stats([p["baseline_absolute_log_step"] for p in pairs]),
            "refined_abs_log_sn_feida_step": stats([p["candidate_absolute_log_step"] for p in pairs]),
            "control_abs_log_sn_feida_step": stats([p.get("unmodified_refit_absolute_log_step") for p in pairs]),
            "refine_cpu_ms": stats([r["elapsed_ms"] for r in frames]), "pairs": pairs,
            "limitations": ["Coarse scale is independent of candidate radius but is not calibrated metric truth.",
                            "Scale-only bounds omit fitted-radius and optical/model uncertainty.",
                            "Sparse samples, not an exhaustive sequential video replay; no missing frames filled.",
                            "Constant or wrong ellipses are not validated by area stability."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["cv", "sequence"])
    p.add_argument("output", type=Path)
    p.add_argument("reports", nargs="+", type=Path)
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(Path("outputs").resolve()):
        p.error("output must be below outputs")
    reports = [json.loads(path.read_text()) for path in args.reports]
    if args.mode == "sequence" and len(reports) != 1:
        p.error("sequence expects one evaluation")
    result = summarize_cv(reports) if args.mode == "cv" else summarize_sequence(reports[0])
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("matched", "pairs", "failures")}))


if __name__ == "__main__":
    main()
