#!/usr/bin/env python3
"""Inventory every available capture without extracting or decoding image payloads.

JSON is a report/index format here, not a change to the recording wire format.
Equal frame-index digests identify potential copies; they do NOT attest equal RAW
bytes. The evaluator must verify payload identity before deduplicating exposures.
"""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import tarfile


def clock_key(row):
    source = row.get("source_clock") or {}
    key = source.get("source_key") or source
    epoch = key.get("stream_epoch")
    if epoch is None:
        epoch = (row.get("region") or {}).get("session")
    return epoch


def describe_frames(payload, stream_sizes):
    eyes = collections.Counter()
    paired_reads = collections.defaultdict(set)
    geometry = collections.defaultdict(set)
    clocks = set()
    unknown_clock = missing_payload = malformed = 0
    timestamp_min = timestamp_max = None
    rows = 0
    for line in payload.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            eye = int(row["eye_id"])
            stamp = int(row["timestamp_ns"])
            width, height = int(row["width"]), int(row["height"])
            stream = row.get("stream")
            offset, length = int(row["offset"]), int(row["length"])
        except (ValueError, TypeError, KeyError):
            malformed += 1
            continue
        rows += 1
        eyes[str(eye)] += 1
        geometry[str(eye)].add((width, height, row.get("pixel_format")))
        epoch = clock_key(row)
        if epoch is None:
            unknown_clock += 1
        else:
            clocks.add(str(epoch))
        # The archive identifies a shared service stream, not a measured
        # rolling-shutter row phase. Preserve unknown epoch separately.
        paired_reads[(epoch, stamp)].add(eye)
        if offset < 0 or length <= 0 or offset + length > stream_sizes.get(stream, 0):
            missing_payload += 1
        timestamp_min = stamp if timestamp_min is None else min(timestamp_min, stamp)
        timestamp_max = stamp if timestamp_max is None else max(timestamp_max, stamp)
    stereo = sum(1 for es in paired_reads.values() if 1 in es and 2 in es)
    attested = sum(1 for (epoch, _), es in paired_reads.items()
                   if epoch is not None and 1 in es and 2 in es)
    return {
        "frame_index_sha256": hashlib.sha256(payload).hexdigest(),
        "frames": rows, "frames_by_eye": dict(eyes),
        "same_timestamp_stereo_reads": stereo,
        "same_epoch_timestamp_stereo_reads": attested,
        "clock_epochs": sorted(clocks), "unknown_clock_frames": unknown_clock,
        "invalid_or_missing_payload_frames": missing_payload,
        "malformed_frame_rows": malformed,
        "timestamp_ns_range": [timestamp_min, timestamp_max],
        "geometry_by_eye": {e: sorted(gs) for e, gs in geometry.items()},
    }


def inventory_archive(path):
    entries = []
    with tarfile.open(path, "r:*") as archive:
        members = {m.name.removeprefix("./"): m for m in archive if m.isfile()}
        indices = sorted(n for n in members if Path(n).name == "frames.jsonl")
        for name in indices:
            prefix = name[:-len("frames.jsonl")]
            sizes = {n[len(prefix):]: m.size for n, m in members.items()
                     if n.startswith(prefix)}
            row = describe_frames(archive.extractfile(members[name]).read(), sizes)
            row.update({"path": str(path), "kind": "tar", "prefix": prefix,
                        "archive_bytes": path.stat().st_size,
                        "streams": {n: s for n, s in sizes.items() if n.endswith(".raw10")},
                        "has_predictions": "predictions.jsonl" in sizes,
                        "has_metadata_oim1": "metadata.oim1" in sizes,
                        "has_manifest": "manifest.json" in sizes})
            entries.append(row)
    return entries


def inventory_directory(path):
    sizes = {p.name: p.stat().st_size for p in path.parent.iterdir() if p.is_file()}
    row = describe_frames(path.read_bytes(), sizes)
    row.update({"path": str(path.parent), "kind": "directory", "prefix": "",
                "streams": {n: s for n, s in sizes.items() if n.endswith(".raw10")},
                "has_predictions": "predictions.jsonl" in sizes,
                "has_metadata_oim1": "metadata.oim1" in sizes,
                "has_manifest": "manifest.json" in sizes})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    archives, indices, labels = [], [], []
    # Runtime libraries/build graphs are not capture stores. Do not follow
    # links back into the same bulk-data root or another source checkout.
    excluded = {".git", "target", "runtime", "models", "weights", "node_modules"}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded
                         and not Path(directory, d).is_symlink())
        for name in files:
            p = Path(directory, name)
            if p.is_symlink():
                continue
            if name.endswith((".tar", ".tar.gz", ".tgz")):
                archives.append(p)
            elif name == "frames.jsonl":
                indices.append(p)
            elif name.endswith(".labels.json"):
                labels.append(str(p))
    entries, errors, noncapturing = [], [], []
    for i, path in enumerate(sorted(archives)):
        try:
            rows = inventory_archive(path)
            entries.extend(rows)
            if not rows:
                noncapturing.append(str(path))
        except (OSError, tarfile.TarError, ValueError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
        if (i + 1) % 20 == 0:
            print(f"indexed {i + 1}/{len(archives)} archives", flush=True)
    for path in sorted(indices):
        try:
            entries.append(inventory_directory(path))
        except (OSError, ValueError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    digest_groups = collections.defaultdict(list)
    for index, row in enumerate(entries):
        digest_groups[row["frame_index_sha256"]].append(index)
    # Prefer original archives; all copies remain explicitly listed.
    representatives = [indices[0] for indices in digest_groups.values()]
    summary = {
        "archives_scanned": len(archives), "extracted_indices_scanned": len(indices),
        "capture_entries": len(entries), "unique_frame_index_groups": len(representatives),
        "stereo_frame_index_groups": sum(entries[i]["same_timestamp_stereo_reads"] > 0
                                         for i in representatives),
        "candidate_unique_frames": sum(entries[i]["frames"] for i in representatives),
        "candidate_unique_stereo_reads": sum(entries[i]["same_timestamp_stereo_reads"]
                                            for i in representatives),
        "labels_files": len(labels), "errors": len(errors),
    }
    report = {"schema": "buttercup-stereo-inventory-v1", "root": str(root),
              "summary": summary, "entries": entries,
              "frame_index_groups": list(digest_groups.values()),
              "representative_entry_indices": representatives,
              "labels": sorted(labels), "errors": errors,
              "archives_without_frame_index": noncapturing,
              "limitations": [
                  "Metadata-only inventory; RAW-byte identity and inference replay are separate checks.",
                  "Equal wire timestamps do not measure rolling-shutter row exposure offsets.",
                  "Unknown clock epochs must not become attested cross-session synchronization.",
                  "Labels are listed, not assumed to be paired or gaze ground truth."]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
