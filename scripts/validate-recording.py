#!/usr/bin/env python3
"""Read-only OIM1/archive handoff checks. Never infer missing gaze or clocks."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import tarfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def framed_records(stream):
    while True:
        prefix = stream.read(24)
        if not prefix:
            return
        assert len(prefix) == 24, "truncated OIM1 prefix"
        magic, version, size, length, native, payload, reserved = struct.unpack("<4sHHIIII", prefix)
        assert (magic, version, size, native, payload, reserved) == (b"OIM1", 1, 24, 0, 0, 0)
        assert 0 < length <= 1024 * 1024, "unbounded metadata"
        data = stream.read(length)
        assert len(data) == length, "truncated OIM1 object"
        value = json.loads(data)
        assert isinstance(value, dict)
        yield value


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), "non-finite metadata"
    elif isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_ns") and child is not None:
                assert isinstance(child, str) and child.isdecimal(), (key, child)
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def validate(path):
    with tarfile.open(path, "r:") as archive:
        members = {m.name: m for m in archive.getmembers()}
        def read(name):
            return archive.extractfile(name).read()
        def lines(name):
            return [json.loads(line) for line in read(name).splitlines() if line]
        manifest = json.loads(read("manifest.json"))
        assert manifest["viewer_events_index"] == manifest["scene_index"] == "metadata.oim1"
        assert manifest["viewer_events_schema"] == "buttercup-viewer-event-v2"
        assert manifest["scene_schema"] == "buttercup-scene-v1"
        for key in ("index", "prediction_index", "recovery_index", "thumbnail_index", "thumbnail_stream"):
            assert manifest[key] in members
        assert "viewer-events.jsonl" not in members
        rows = list(framed_records(archive.extractfile("metadata.oim1")))
        frames = lines("frames.jsonl")
        frame_keys = {canonical(f["source_clock"]["source_key"]) for f in frames if f.get("source_clock")}
        thumbnails = lines("thumbnails.jsonl")
        start = next(r for r in rows if r["event"] == "recording_started")
        assert rows[-1]["event"] == "recording_stopped" and rows[-1]["sidecars_finalized"]
        start_ns = int(start["host_monotonic_ns"])
        config, scenes, refs, dropped = {}, {}, {}, set()
        events, states, intersections, reference_classes = Counter(), Counter(), Counter(), Counter()
        drop_archive_classes = Counter()
        target_positions, predictions, eye_ids = set(), set(), set()
        held = 0
        ray_comparisons = 0
        source_gaps = 0
        last_sequence = None
        presentations = []
        for row in rows:
            finite(row)
            event = row["event"]
            events[event] += 1
            session = row.get("viewer_session_id")
            seq = row.get("metadata_sequence")
            if seq is not None:
                seq = int(seq)
                if last_sequence is not None:
                    assert seq > last_sequence, "metadata order reversed"
                    source_gaps += seq - last_sequence - 1
                last_sequence = seq
            if event == "configuration_changed":
                config[(session, row["configuration_revision"])] = row["data"]
            elif event == "scene_sample":
                scenes[(session, row["data"]["scene_revision"])] = row["data"]["sample"]
            elif event in ("recording_start_snapshot", "queue_recovery_snapshot", "live_checkpoint"):
                config[(session, row["configuration_revision"])] = row["configuration"]
                scenes[(session, row["scene_revision"])] = row["scene"]
                assert row["snapshot_is_new_observation"] is False
            elif event == "source_dropped":
                key = row["data"]["clock"]
                if key:
                    source_key = canonical(key["source_key"])
                    dropped.add(source_key)
                    if row["data"].get("archived") is True:
                        assert source_key in frame_keys, "saved-but-unanalyzed ROI lacks its exact native payload"
                        assert row["data"].get("drop_scope") == "analysis"
                        drop_archive_classes["saved-native-roi-analysis-skipped"] += 1
                    elif source_key in frame_keys:
                        drop_archive_classes["native-roi-present-resolved-by-index"] += 1
                    else:
                        drop_archive_classes["native-roi-not-archived"] += 1
                else:
                    drop_archive_classes["non-roi-resolve-thumbnail-index"] += 1
            elif event == "presentation":
                presentations.append(row)
                cfg = config[(session, row["configuration_revision"])]
                scene = scenes[(session, row["scene_revision"])]
                assert scene["configuration_revision"] == row["configuration_revision"]
                assert int(row["host_submit_begin_monotonic_ns"]) <= int(row["host_submit_end_monotonic_ns"])
                for target in row["active_targets"]:
                    assert target["visible"] is True
                    target_positions.add(tuple(target["normalized"]))
                    assert "appearance" in target
                gaze = row["gaze"]
                if gaze["predicted_normalized"] is not None:
                    predictions.add(tuple(gaze["predicted_normalized"]))
                if gaze["held_geometry"]:
                    held += 1
                    assert not gaze["source_advanced_for_basis"]
                if gaze["drawn_normalized"] is not None:
                    assert all(0 <= v <= 1 for v in gaze["drawn_normalized"])
                geometry = cfg["geometry"]
                assert geometry["frame"]["units"] == "inches"
                assert geometry["frame"]["handedness"] == "left"
                monitor = geometry["monitor"]
                for eye in scene["eyes"]:
                    eye_ids.add(eye["roi_id"])
                    states[(eye["roi_id"], eye["state"])] += 1
                    assert eye["visual_axis"] is None and eye["metric_radius"] is None
                    if eye["roi_id"] != geometry["frame"]["reference_roi_id"]:
                        assert eye["predicted_screen_uv"] is None and eye["ray_origin"] is None
                    source = eye["analysis_source"]
                    refs[canonical(source)] = source
                    hit = eye["monitor_intersection"]
                    intersections[hit["status"]] += 1
                    if hit["uv"] is not None:
                        uv, point = hit["uv"], hit["point"]
                        matrix = monitor["uv_to_scene_3x3"]
                        for dim in range(3):
                            expected = matrix[dim][0]*uv[0] + matrix[dim][1]*uv[1] + matrix[dim][2]
                            assert abs(expected-point[dim]) < 1e-5, "UV/world transform mismatch"
                        mapping = cfg["mapping"]
                        if mapping.get("affine") is None and eye["predicted_screen_uv"] is not None:
                            assert max(abs(a-b) for a,b in zip(uv,eye["predicted_screen_uv"])) < 1e-8
                            ray_comparisons += 1
        for source in refs.values():
            key = source.get("key")
            if key is None:
                reference_classes[source["status"]] += 1
            elif canonical(key) in frame_keys:
                reference_classes["archived-exact"] += 1
            elif int(source["clock"]["host_arrival_monotonic_ns"]) < start_ns:
                reference_classes["pre-recording-history"] += 1
            elif canonical(key) in dropped:
                reference_classes["explicitly-dropped-source"] += 1
            else:
                raise AssertionError(f"unexplained source reference: {key}")
        for entry in thumbnails:
            assert entry["offset"] + entry["length"] <= members["thumbnails.oic1"].size
        metadata_size = members["metadata.oim1"].size
        missing = []
        if len(target_positions) < 2:
            missing.append("two-or-more-successfully-submitted-target-positions")
        if {f["eye_id"] for f in frames} != {1, 2}:
            missing.append("both-raw-rois")
        if not predictions:
            missing.append("available-2d-gaze-prediction")
        if events["queue_gap"]:
            missing.append("gap-free-presentation-history")
        return {
            "archive": str(Path(path).resolve()), "format_and_references_validated": True,
            "live_acceptance_complete": not missing, "live_acceptance_missing": missing,
            "archive_bytes": Path(path).stat().st_size, "metadata_bytes": metadata_size,
            "metadata_percent_archive": round(100*metadata_size/Path(path).stat().st_size, 3),
            "raw_frames_by_roi": dict(Counter(f["eye_id"] for f in frames)),
            "native_thumbnails": len(thumbnails), "thumbnail_snapshots": sum(t["recording_start_snapshot"] for t in thumbnails),
            "events": dict(events), "target_positions": len(target_positions),
            "distinct_predicted_uv": len(predictions), "held_presentations": held,
            "source_references": dict(reference_classes), "metadata_sequence_gaps": source_gaps,
            "source_drop_archive_classes": dict(drop_archive_classes),
            "roi_states": {f"{roi}:{state}":n for (roi,state),n in states.items()},
            "intersections": dict(intersections), "plane_only_screen_ray_comparisons": ray_comparisons,
            "interrupted": rows[-1]["interrupted"],
            "fixation_verified": False, "scanout_sensor_clock_calibrated": False,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    args = parser.parse_args()
    print(json.dumps(validate(args.archive), indent=2))
