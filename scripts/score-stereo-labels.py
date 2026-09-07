#!/usr/bin/env python3
"""Post-fit native-RAW label join/scoring; labels never enter the gaze solver.

Join by exact packed RAW SHA256 AND native dimensions/sensor origin. Sequence
number or filename similarity alone is not image identity. Assistant labels
and superseded backups are excluded. Visibility remains a separate stratum.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def local(point, ellipse):
    x, y = point[0]-ellipse["center"][0], point[1]-ellipse["center"][1]
    sine, cosine = math.sin(ellipse["angle"]), math.cos(ellipse["angle"])
    return cosine*x+sine*y, -sine*x+cosine*y


def distance(point, ellipse):
    # Same bracket + golden-section Euclidean metric as flat-tire label audit.
    x, y = map(abs, local(point, ellipse))
    a, b = ellipse["major_radius"], ellipse["minor_radius"]
    def squared(t):
        return (a*math.cos(t)-x)**2+(b*math.sin(t)-y)**2
    best = min(range(65), key=lambda i: squared(math.pi*i/128))
    lo, hi = math.pi*max(0, best-1)/128, math.pi*min(64, best+1)/128
    for _ in range(32):
        p, q = lo+(hi-lo)*0.38196601125, lo+(hi-lo)*0.61803398875
        if squared(p) < squared(q):
            hi = q
        else:
            lo = p
    return math.sqrt(squared((lo+hi)/2))


def band_intersects(inner, outer, ellipse):
    x, y = local(inner, ellipse)
    u, v = local(outer, ellipse)
    a, b = ellipse["major_radius"], ellipse["minor_radius"]
    x, y, dx, dy = x/a, y/b, (u-x)/a, (v-y)/b
    aa, bb, cc = dx*dx+dy*dy, 2*(x*dx+y*dy), x*x+y*y-1
    disc = bb*bb-4*aa*cc
    if aa <= 1e-20:
        return abs(cc) <= 1e-12
    if disc < 0:
        return False
    return any(0 <= (-bb+s*math.sqrt(disc))/(2*aa) <= 1 for s in (-1, 1))


def metrics(label, ellipse):
    if ellipse is None:
        return None
    strata = collections.defaultdict(list)
    bands = collections.defaultdict(list)
    for point in label.get("annotation_points", []):
        if point.get("kind") != "iris_edge" or point.get("x") is None or point.get("y") is None:
            continue
        visibility = point.get("visibility", "unknown")
        strata[visibility].append(distance((point["x"], point["y"]), ellipse))
        if point.get("band_inner") and point.get("band_outer"):
            bands[visibility].append(band_intersects(point["band_inner"], point["band_outer"], ellipse))
    return {key: {"points": len(values), "rms_px": math.sqrt(sum(v*v for v in values)/len(values)),
                  "maximum_px": max(values), "band_count": len(bands[key]),
                  "bands_intersected": sum(bands[key])} for key, values in strata.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("evaluations", type=Path, nargs="*")
    parser.add_argument("--replay-index", type=Path, help="write prediction/annotation-free source receipts for matched labels and four neighboring receipts on each side")
    args = parser.parse_args()
    labels, unavailable, excluded = [], [], []
    by_identity = collections.defaultdict(list)
    for path in map(Path, json.loads(args.inventory.read_text())["labels"]):
        if "assistant" in str(path) or "backup" in path.name:
            excluded.append(str(path))
            continue
        label = json.loads(path.read_text())
        if label.get("reviewed") is not True:
            excluded.append(str(path))
            continue
        raw = Path(label["source_raw"])
        if not raw.is_file():
            unavailable.append({"label": str(path), "reason": "label-native-RAW-missing", "raw": str(raw)})
            continue
        with raw.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        identity = (digest, label["frame_width"], label["frame_height"], *label["sensor_origin"])
        item = {"label": str(path), "native_identity": identity, "source_indices": [], "scores": [],
                "provenance": label.get("provenance"), "document": label}
        labels.append(item)
        by_identity[identity].append(item)
    by_index = collections.defaultdict(list)
    with args.frames.open() as source:
        for line in source:
            row = json.loads(line)
            frame = row["frame"]
            identity = (row["raw_sha256"], frame["width"], frame["height"], frame["sensor_x"], frame["sensor_y"])
            for item in by_identity.get(identity, []):
                item["source_indices"].append(row["index"])
                by_index[row["index"]].append(item)
    for evaluation in args.evaluations:
        with evaluation.open() as source:
            for line in source:
                row = json.loads(line)
                for eye, receipt in enumerate(row.get("inputs", [])):
                    for item in by_index.get(receipt["index"] if receipt else None, []):
                        score = {"evaluation": str(evaluation), "index": receipt["index"]}
                        score["SAM"] = {"accepted": row["raw_admitted"][eye],
                            "metrics": metrics(item["document"], row["baseline_sam_outer"][eye])}
                        for name in ("joint", "monocular_right" if eye == 0 else "monocular_left"):
                            fit = row.get(name, {})
                            ellipse = fit.get("outer_ellipses", [None, None])[eye]
                            score[name] = {"accepted": fit.get("contributing_eyes", [False, False])[eye],
                                "metrics": metrics(item["document"], ellipse)}
                        item["scores"].append(score)
    for item in labels:
        del item["document"]
    replay_count = 0
    if args.replay_index:
        indices = {i for index in by_index for i in range(max(0, index-4), index+5)}
        # Receipts contain RAW provenance and independent scale only, not label
        # coordinates, recorded predictions, or target locations. Selecting a
        # validation subset is separate from supplying geometry to its solver.
        with args.frames.open() as source, args.replay_index.open("x") as destination:
            for line in source:
                if json.loads(line)["index"] in indices:
                    destination.write(line)
                    replay_count += 1
    report = {"schema": "buttercup-stereo-postfit-labels-v1", "labels": labels,
        "unavailable": unavailable, "excluded": excluded,
        "targeted_replay": {"path": str(args.replay_index) if args.replay_index else None, "source_receipts": replay_count},
        "summary": {"reviewed_native_labels": len(labels), "labels_in_stereo_RAW": sum(bool(r["source_indices"]) for r in labels),
                    "labels_with_completed_fits": sum(bool(r["scores"]) for r in labels)},
        "limitations": ["A source-native limbus label is localization evidence, not independent gaze or 3D anatomy ground truth.",
                        "Guessed, visible and unknown landmarks are separate; rejected conics remain diagnostics, not accepted coverage.",
                        "No filename-only, sequence-only or approximate image joins."]}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
