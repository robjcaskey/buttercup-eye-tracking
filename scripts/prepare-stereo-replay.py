#!/usr/bin/env python3
"""Make a byte-verified, source-keyed replay index; never copy image payloads.

Every capture index with two recorded ROI identities is accounted for, including
incomplete archives. Missing RAW stays missing. Shared source reads are paired
by explicit stream/region clock and exact sensor timestamp, never host arrival.
Unknown epochs are scoped to one capture, not silently joined across sessions.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
import tarfile


def number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clock(row, fallback):
    source = (row.get("source_clock") or {}).get("source_key") or {}
    if source.get("stream_epoch") is not None:
        return "stream:" + str(source["stream_epoch"]), True
    region = (row.get("region") or {}).get("session")
    if region is not None:
        return "region:" + str(region), True
    return "capture:" + fallback, False


def metadata_records(stream):
    while True:
        prefix = stream.read(24)
        if not prefix:
            return
        if len(prefix) != 24:
            raise ValueError("truncated metadata prefix")
        magic, version, size, length, native, payload, reserved = struct.unpack("<4sHHIIII", prefix)
        if (magic, version, size, native, payload, reserved) != (b"OIM1", 1, 24, 0, 0, 0) or not 0 < length <= 1048576:
            raise ValueError("invalid metadata prefix")
        data = stream.read(length)
        if len(data) != length:
            raise ValueError("truncated metadata object")
        yield json.loads(data)


class Capture:
    def __init__(self, entry):
        self.entry = entry
        self.archive = None
        self.members = {}
        if entry["kind"] == "tar":
            # Random-access offsets are only meaningful for uncompressed tar.
            if not entry["path"].endswith(".tar"):
                raise ValueError("compressed source needs a separately verified uncompressed copy")
            self.archive = tarfile.open(entry["path"], "r:")
            self.members = {m.name.removeprefix("./"): m for m in self.archive if m.isfile()}

    def stream(self, name):
        if self.archive:
            member = self.members.get(self.entry["prefix"] + name)
            return self.archive.extractfile(member) if member else None
        path = Path(self.entry["path"], name)
        return path.open("rb") if path.is_file() else None

    def raw_location(self, row):
        name = row["stream"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("unsafe RAW stream member")
        offset, length = int(row["offset"]), int(row["length"])
        if self.archive:
            member = self.members.get(self.entry["prefix"] + name)
            if not member or offset < 0 or length <= 0 or offset + length > member.size:
                return None
            return self.entry["path"], member.offset_data + offset, length
        path = Path(self.entry["path"], name)
        if not path.exists() or offset < 0 or length <= 0 or offset + length > path.stat().st_size:
            return None
        return str(path), offset, length

    def close(self):
        if self.archive:
            self.archive.close()


def scene_scales(capture):
    stream = capture.stream("metadata.oim1")
    scales = {}
    if stream is None:
        return scales
    with stream:
        for record in metadata_records(stream):
            event = record.get("event")
            if event == "scene_sample":
                sample = (record.get("data") or {}).get("sample") or {}
            else:
                scene = record.get("scene") or {}
                sample = scene.get("sample", scene)
            for eye in sample.get("eyes", []):
                hint = eye.get("scale_hint")
                source = (eye.get("raw_clock") or {}).get("source_key") or {}
                stamp, roi = number(source.get("sensor_timestamp_ns")), number(source.get("roi_id"))
                if hint and stamp is not None and roi is not None:
                    scales[(roi, stamp)] = hint
    return scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    args.output_directory.mkdir(parents=True, exist_ok=False)
    entries = inventory["entries"]
    # Process ALL entries, not merely index-hash representatives. Byte hashes
    # prove duplicates even if copied indexes have differing whitespace/paths.
    ids = [i for i, e in enumerate(entries) if e["same_timestamp_stereo_reads"] > 0]
    seen = {}
    reports, errors = [], []
    read_groups = collections.defaultdict(dict)
    files = {}
    frames_path = args.output_directory / "frames.jsonl"
    with frames_path.open("x") as writer:
        for progress, entry_id in enumerate(ids):
            entry = entries[entry_id]
            report = {"entry_id": entry_id, "path": entry["path"], "frames_indexed": 0,
                      "new_frames": 0, "duplicate_frames": 0, "missing_raw_frames": 0,
                      "clock_unknown_frames": 0, "scale_hint_frames": 0}
            capture = None
            try:
                capture = Capture(entry)
                try:
                    scale = scene_scales(capture)
                except (ValueError, OSError) as exc:
                    # A partial auxiliary stream cannot erase intact RAW.
                    # Missing scale is explicitly unavailable in the replay.
                    scale = {}
                    report["metadata_warning"] = str(exc)
                source = capture.stream("frames.jsonl")
                with source:
                    rows = [json.loads(line) for line in source if line.strip()]
                # Keep source ordering within each ROI. Primary/secondary
                # frames from one read are consecutive in the exported index.
                rows.sort(key=lambda r: (clock(r, entry["frame_index_sha256"])[0], int(r["timestamp_ns"]), int(r["eye_id"])))
                for row in rows:
                    eye, stamp = int(row["eye_id"]), int(row["timestamp_ns"])
                    if eye not in (1, 2):
                        continue
                    report["frames_indexed"] += 1
                    location = capture.raw_location(row)
                    if location is None:
                        report["missing_raw_frames"] += 1
                        continue
                    path, offset, length = location
                    if path not in files:
                        files[path] = open(path, "rb")
                    stream = files[path]
                    stream.seek(offset)
                    payload = stream.read(length)
                    if len(payload) != length:
                        report["missing_raw_frames"] += 1
                        continue
                    digest = hashlib.sha256(payload).hexdigest()
                    lineage, attested = clock(row, entry["frame_index_sha256"])
                    key = (lineage, stamp, eye, int(row["sensor_x"]), int(row["sensor_y"]),
                           int(row["width"]), int(row["height"]), int(row["stride"]), digest)
                    report["clock_unknown_frames"] += not attested
                    scale_hint = scale.get((eye, stamp))
                    report["scale_hint_frames"] += scale_hint is not None
                    if key in seen:
                        report["duplicate_frames"] += 1
                        continue
                    index = len(seen)
                    seen[key] = index
                    read_groups[(lineage, stamp)][eye] = index
                    report["new_frames"] += 1
                    writer.write(json.dumps({"index": index, "capture_entry": entry_id,
                        "clock_lineage": lineage, "clock_attested": attested,
                        "raw_file": path, "raw_offset": offset, "raw_length": length,
                        "raw_sha256": digest, "frame": row, "scale_hint": scale_hint}, separators=(",", ":")) + "\n")
                # Bound open file count across the complete collection.
                for stream in files.values():
                    stream.close()
                files.clear()
            except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
                errors.append({"entry_id": entry_id, "path": entry["path"], "error": str(exc)})
            finally:
                if capture:
                    capture.close()
            reports.append(report)
            if (progress + 1) % 10 == 0:
                print(f"verified {progress + 1}/{len(ids)} capture entries; {len(seen)} distinct RAW exposures", flush=True)
    with (args.output_directory / "pairs.jsonl").open("x") as writer:
        for (lineage, stamp), eyes in read_groups.items():
            writer.write(json.dumps({"clock_lineage": lineage, "timestamp_ns": stamp,
                                    "eyes": [eyes.get(1), eyes.get(2)]}, separators=(",", ":")) + "\n")
    summary = {"capture_entries_considered": len(ids), "unique_raw_frames": len(seen),
        "same_read_pairs_with_both_raw": sum(1 in e and 2 in e for e in read_groups.values()),
        "single_roi_reads": sum(not (1 in e and 2 in e) for e in read_groups.values()),
        "duplicate_raw_receipts": sum(r["duplicate_frames"] for r in reports),
        "missing_raw_receipts": sum(r["missing_raw_frames"] for r in reports), "errors": len(errors)}
    report = {"schema": "buttercup-byte-verified-stereo-replay-v1", "inventory": str(args.inventory),
        "frames_index": str(frames_path), "summary": summary, "captures": reports, "errors": errors,
        "contract": "Every available valid native RAW exposure from every indexed stereo capture; SHA256 plus source-clock/geometry identity deduplicates copies. No prediction or label seeds. Unknown clock epochs stay capture-local. Missing or truncated RAW is explicitly counted, never fabricated."}
    (args.output_directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
