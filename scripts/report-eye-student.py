#!/usr/bin/env python3
"""Summarize matched SAM/student diagnostics without calling SAM ground truth."""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def summary(values):
    values = sorted(x for x in values if math.isfinite(x))
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "p95": values[min(len(values) - 1, math.ceil(len(values) * .95) - 1)] if values else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--replay", action="append", type=Path, default=[])
    args = parser.parse_args()
    with args.evaluation.open() as source:
        rows = [json.loads(line) for line in source]
    report = {"scope": "730-frame pilot if using the initial dataset; actual counts below",
              "human_label_localization": "not measured; SAM agreement is not human-label accuracy",
              "timing_caution": "student_ms excludes cached input preprocessing; teacher_ms queries six prompts, not the normal two-head live path",
              "area_caution": "SN-FEIDA uses source-provided independent pixels_per_10mm only; sparse frames are not a temporal-stability evaluation",
              "splits": {}}
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["input"]["student_split"] == split]
        counts = collections.Counter()
        center = []
        pupil_center = []
        major = []
        minor = []
        areas = {"teacher": [], "student": []}
        for row in subset:
            t, s = row["teacher"], row["student"]
            ta, sa = bool(t["raw_admitted"]), bool(s["raw_admitted"])
            counts["teacher_admitted"] += ta
            counts["student_admitted"] += sa
            counts["both_admitted"] += ta and sa
            counts["teacher_only"] += ta and not sa
            counts["student_only_unverified"] += sa and not ta
            counts["teacher_pupil"] += ta and t["pupil_ellipse"] is not None
            counts["student_pupil"] += sa and s["pupil_ellipse"] is not None
            if ta and sa:
                te, se = t["outer_ellipse"], s["outer_ellipse"]
                center.append(math.dist(te["center"], se["center"]))
                major.append(abs(te["major_radius"] - se["major_radius"]))
                minor.append(abs(te["minor_radius"] - se["minor_radius"]))
                if t["pupil_ellipse"] is not None and s["pupil_ellipse"] is not None:
                    pupil_center.append(math.dist(t["pupil_ellipse"]["center"],
                                                 s["pupil_ellipse"]["center"]))
            scale = (row["input"].get("scale_hint") or {}).get("pixels_per_10mm")
            if isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0:
                for name in areas:
                    result = row[name]
                    if result["raw_admitted"] and result["outer_ellipse"]:
                        areas[name].append(math.pi * (result["outer_ellipse"]["major_radius"] / (scale / 10)) ** 2)
        report["splits"][split] = {"frames": len(subset), **counts,
            "sessions": len({r["input"]["clock_lineage"] for r in subset}),
            "center_disagreement_with_teacher_px": summary(center),
            "pupil_center_disagreement_with_teacher_px": summary(pupil_center),
            "major_radius_disagreement_with_teacher_px": summary(major),
            "minor_radius_disagreement_with_teacher_px": summary(minor),
            "sn_feida_mm2_heuristic_scale": {k: summary(v) for k, v in areas.items()},
            "student_cached_input_mask_and_geometry_ms": summary(r["student_ms"] for r in subset),
            "teacher_six_prompt_export_ms": summary(r["teacher"]["teacher_ms"] for r in subset)}
    report["completion_paced_live_worker_replays"] = {}
    by_backend = {}
    for path in args.replay:
        with path.open() as source:
            replay = [json.loads(line) for line in source]
        if replay:
            by_backend[replay[0]["backend"]] = {r["input"]["raw_sha256"]: r for r in replay}
        # Exclude the first use of each ROI lane (lazy model/CUDA warmup).
        seen = set()
        warm = []
        for row in replay:
            eye = row["input"]["frame"]["eye_id"]
            if eye in seen:
                warm.append(row)
            seen.add(eye)
        report["completion_paced_live_worker_replays"][str(path)] = {
            "frames": len(replay), "accepted": sum(r["accepted"] for r in replay),
            "source_identity_verified": all(r["source_identity_verified"] for r in replay),
            "elapsed_ms_excluding_first_per_eye": summary(r["elapsed_ms"] for r in warm),
            "encode_ms_excluding_first_per_eye": summary(r["encode_ms"] for r in warm),
            "note": "includes live preprocessing and shared geometry, not camera/display latency or offered-load frame drops"}
    if "sam" in by_backend and "student" in by_backend:
        sam, student = by_backend["sam"], by_backend["student"]
        if sam.keys() != student.keys():
            raise ValueError("replay sources differ; cannot report matched coverage")
        pairs = [(sam[key], student[key]) for key in sam]
        report["matched_live_worker_coverage"] = {
            "frames": len(pairs),
            "both_admitted": sum(a["accepted"] and b["accepted"] for a, b in pairs),
            "sam_only": sum(a["accepted"] and not b["accepted"] for a, b in pairs),
            "student_only_unverified": sum(b["accepted"] and not a["accepted"] for a, b in pairs),
            "sam_pupil": sum(a["accepted"] and a["pupil_present"] for a, _ in pairs),
            "student_pupil": sum(b["accepted"] and b["pupil_present"] for _, b in pairs)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
