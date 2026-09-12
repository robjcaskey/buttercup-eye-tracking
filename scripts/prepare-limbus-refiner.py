#!/usr/bin/env python3
"""Prepare source-native human landmark supervision and a separate SAM input.

No completed human ellipse, gaze prediction or target location is a feature.
Unknown/possibly-occluded landmarks remain unknown/occluded respectively.
All artifacts are runtime data. Older reviewed human bands are retained, but
assistant annotations and backups never become training examples.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROLES = ["rim", "band_inner", "band_outer", "iris_onset", "surface_apex", "subsurface_limit"]


def visible(point):
    return (isinstance(point, dict) and point.get("visibility") == "visible"
            and isinstance(point.get("x"), (int, float))
            and isinstance(point.get("y"), (int, float)))


def observations(label):
    result = []
    for p in label.get("annotation_points", []):
        if p.get("kind") != "iris_edge" or not visible(p):
            continue
        xy = [p["x"], p["y"]]
        targets = {"rim": xy}
        weight = {"rim": 0.3 if p.get("source") == "paired_midpoint" else 1.0}
        for field, role in [("band_inner", "band_inner"), ("band_outer", "band_outer"),
                            ("iris_side_onset", "iris_onset"),
                            ("subsurface_visibility_limit", "subsurface_limit")]:
            if p.get(field) is not None:
                targets[role] = p[field]
        if "visibility_triplet_apex" in p.get("source", ""):
            targets["surface_apex"] = xy
        occluded = ["subsurface_limit"] if "submerged" in p.get("possibly_occluded", []) else []
        result.append({"anchor": xy, "targets": targets, "weights": weight,
                       "occluded": occluded, "kind": p.get("source", "legacy_single")})
    # Partials with an apex were already included above. An inner-only partial
    # supplies no invented rim/apex coordinate, but does supervise abstention.
    for partial in label.get("limbus_partial_observations", []):
        marks = partial.get("landmarks", {})
        if marks.get("apex") is not None or not visible(marks.get("inner")):
            continue
        inner = [marks["inner"]["x"], marks["inner"]["y"]]
        triplet = partial.get("mode") == "triplet"
        result.append({"anchor": inner,
                       "targets": {"iris_onset" if triplet else "band_inner": inner},
                       "weights": {}, "kind": "possibly_occluded",
                       "occluded": ["rim", "surface_apex", "subsurface_limit"] if triplet else ["band_outer"]})
    return result


def prepare(inventory):
    rows, excluded, seen = [], [], set()
    for name in inventory["labels"]:
        path = Path(name)
        if "assistant" in str(path) or "backup" in path.name:
            excluded.append({"path": name, "reason": "assistant-or-backup"})
            continue
        label = json.loads(path.read_text())
        if not label.get("reviewed"):
            excluded.append({"path": name, "reason": "not-reviewed"})
            continue
        raw = Path(label["source_raw"])
        metadata = json.loads(raw.with_suffix(".json").read_text())
        width, height = int(label["frame_width"]), int(label["frame_height"])
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        identity = (digest, width, height, *label["sensor_origin"])
        if identity in seen:
            raise ValueError("duplicate reviewed native label; resolve precedence explicitly")
        seen.add(identity)
        if (width != metadata["width"] or height != metadata["height"]
                or label["sensor_origin"] != [metadata["sensor_x"], metadata["sensor_y"]]
                or raw.stat().st_size != width // 4 * 5 * height):
            raise ValueError(f"native RAW/label identity mismatch: {path}")
        frame = {key: metadata[key] for key in
                 ("sequence", "timestamp_ns", "sensor_x", "sensor_y", "width", "height", "stride", "pixel_format")}
        frame["eye_id"] = metadata.get("eye_id", 1)
        frame["label"] = "subject-right" if frame["eye_id"] == 1 else "subject-left"
        source = {"index": len(rows), "raw_file": str(raw.resolve()), "raw_offset": 0,
                  "raw_length": raw.stat().st_size, "raw_sha256": digest, "frame": frame,
                  "scale_hint": None, "clock_attested": False}
        rows.append({"source": source, "label": str(path.resolve()),
                     "observations": observations(label)})
    # Conservative grouping: overlapping/nearby sensor times from these native
    # captures share a fold, even if annotation tasks/directory names differ.
    # This deliberately over-merges uncertain lineages; never random-split patches.
    rows.sort(key=lambda r: int(r["source"]["frame"]["timestamp_ns"]))
    group, previous = -1, None
    for row in rows:
        timestamp = int(row["source"]["frame"]["timestamp_ns"])
        if previous is None or timestamp - previous > 300_000_000_000:
            group += 1
        row["group"] = group
        row["source"]["clock_lineage"] = f"limbus-human-conservative-group:{group}"
        previous = timestamp
    return {"schema": "buttercup-limbus-patch-dataset-v1", "roles": ROLES,
            "frames": rows, "excluded": excluded,
            "grouping": "sensor-timestamp-neighborhoods <=300s conservatively merged; clock identity not independently attested",
            "distance_scale_support": "missing; do not normalize area by candidate radius",
            "landmark_semantics": "optical image landmarks; no anatomical 3D depth or signed gaze labels"}


def prepare_sequence(outline_path, area_path):
    """Import the existing motion experiment, rechecking exact RAW identity.

    Never convert candidate-independent relative image scale into metric depth.
    Preserve gaps, unfit frames and individual scale-chain references.
    """
    outlines = json.loads(outline_path.read_text())
    area = json.loads(area_path.read_text())
    rows, references, motion_reports = [], [], {}
    for case in outlines["cases"]:
        f = case["frame"]
        raw = Path(f["source_raw"])
        payload = raw.read_bytes()
        if len(payload) != f["length"]:
            raise ValueError("archived RAW length mismatch")
        src = {"raw_file": str(raw.resolve()), "raw_offset": 0, "raw_length": len(payload),
               "raw_sha256": hashlib.sha256(payload).hexdigest(), "frame": f,
               "clock_attested": False, "clock_lineage": f["lineage"], "scale_hint": None}
        matched = [r for r in area["frames"] if Path(r["source"]).resolve() == outline_path.resolve()
                   and r["sequence"] == f["sequence"] and r["timestamp_ns"] == f["timestamp_ns"]
                   and r["eye"] == f["label"] and r["width"] == f["width"] and r["height"] == f["height"]]
        if len(matched) != 1:
            raise ValueError("ambiguous outline/motion source join")
        hint = matched[0]["independent_scale"]
        if hint:
            name = hint["report"]
            if name not in motion_reports:
                motion_reports[name] = json.loads(Path(name).read_text())
            motion = motion_reports[name]
            matched_motion = [m for m in motion["frames"] if m["sequence"] == f["sequence"]
                              and m["timestamp_ns"] == f["timestamp_ns"] and m["width"] == f["width"]
                              and m["height"] == f["height"]
                              and m["sensor_origin"] == [f["sensor_x"], f["sensor_y"]]]
            if len(matched_motion) != 1 or motion["source"]["label"] != f["label"]:
                raise ValueError("ambiguous native-motion metadata")
            m = matched_motion[0]
            with Path(motion["source"]["stream"]).open("rb") as stream:
                stream.seek(m["source_offset"])
                if stream.read(m["source_length"]) != payload:
                    raise ValueError("native-motion bytes differ from contour exposure")
            src["relative_scale_hint"] = hint
            src["clock_basis"] = "exact-RAW-matched-native-motion-chain"
            src["clock_lineage"] = str(Path(name).resolve())
        # Select exactly the first RAW-admitted candidate, never the best label
        # match or most stable area. Existing contour exporter owns this order.
        selected = next((c for c in case["candidates"] if c["baseline_raw_admitted"]
                         and c.get("baseline_ellipse")), None)
        label_path = f.get("canonical_label")
        label = json.loads(Path(label_path).read_text()) if label_path else {}
        obs = observations(label) if label.get("reviewed") else []
        rows.append({"source": src, "group": -4, "label": label_path or str(raw),
                     "observations": obs, "supervision": "evaluation-only-sequence"})
        references.append({"input": src, "accepted": selected is not None,
                           "source_identity_verified": True, "candidates": [selected] if selected else []})
    return {"schema": "buttercup-limbus-patch-dataset-v1", "roles": ROLES, "frames": rows,
            "evaluation_only": True, "scope": "archived contour replay; relative scale is not camera range"}, references


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sequence-area-report", type=Path,
                   help="inventory is an existing native outline export; import matched motion/area evidence")
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(Path("outputs").resolve()):
        p.error("output must be beneath outputs")
    args.output.mkdir(exist_ok=False)
    if args.sequence_area_report:
        data, reference = prepare_sequence(args.inventory, args.sequence_area_report)
        with (args.output / "sam-baseline.jsonl").open("x") as f:
            for row in reference:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
    else:
        data = prepare(json.loads(args.inventory.read_text()))
    with (args.output / "dataset.json").open("x") as f:
        json.dump(data, f, indent=2)
    with (args.output / "sam-inputs.jsonl").open("x") as f:
        for row in data["frames"]:
            f.write(json.dumps(row["source"], separators=(",", ":")) + "\n")
    print(json.dumps({"frames": len(data["frames"]),
                      "groups": dict(collections.Counter(r["group"] for r in data["frames"])),
                      "landmarks": dict(collections.Counter(role for r in data["frames"]
                          for o in r["observations"] for role in o["targets"])),
                      "explicitly_occluded": dict(collections.Counter(role for r in data["frames"]
                          for o in r["observations"] for role in o["occluded"]))}))


if __name__ == "__main__":
    main()
