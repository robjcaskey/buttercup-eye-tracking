#!/usr/bin/env python3
"""Select source-native, session-disjoint SAM distillation examples.

Input is the byte-addressed index from prepare-stereo-replay.py. This never
uses recorded gaze, fitted geometry, display targets, or human annotations.
Exact RAW duplicates and entire source-clock sessions stay in one partition.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def partition(key):
    value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
    return "test" if value == 0 else "validation" if value == 1 else "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-eye-session", type=int, default=20)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.per_eye_session <= 200:
        parser.error("per-eye-session must be 1..200")
    root = Path("outputs").resolve()
    if not args.output.parent.resolve().is_relative_to(root):
        parser.error("output must be under outputs")
    groups = collections.defaultdict(list)
    seen = set()
    unavailable = 0
    paths = {}
    with args.index.open() as source:
        for line in source:
            row = json.loads(line)
            path = row["raw_file"]
            if path not in paths:
                paths[path] = Path(path).stat().st_size if Path(path).is_file() else 0
            if int(row["raw_offset"]) + int(row["raw_length"]) > paths[path]:
                unavailable += 1
                continue
            digest = row["raw_sha256"]
            if digest in seen:
                continue
            seen.add(digest)
            groups[(row["clock_lineage"], row["frame"]["eye_id"])].append(row)
    selected = []
    lineages = sorted({key[0] for key in groups}, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    allowed = set(lineages[:args.max_sessions] if args.max_sessions > 0 else lineages)
    for (lineage, eye), rows in sorted(groups.items()):
        if lineage not in allowed:
            continue
        rows.sort(key=lambda r: int(r["frame"]["timestamp_ns"]))
        count = min(len(rows), args.per_eye_session)
        for i in range(count):
            row = rows[(i * (len(rows) - 1)) // max(1, count - 1)]
            row["student_split"] = partition(lineage)
            selected.append(row)
    # Hash all selected native bytes, not merely their index declarations.
    handles = {}
    try:
        with args.output.open("x") as dest:
            for index, row in enumerate(selected):
                stream = handles.setdefault(row["raw_file"], None)
                if stream is None:
                    stream = handles[row["raw_file"]] = open(row["raw_file"], "rb")
                stream.seek(int(row["raw_offset"]))
                payload = stream.read(int(row["raw_length"]))
                if hashlib.sha256(payload).hexdigest() != row["raw_sha256"]:
                    raise ValueError("RAW hash mismatch")
                row["student_index"] = index
                dest.write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        for stream in handles.values():
            if stream is not None:
                stream.close()
    print(json.dumps({"frames": len(selected), "sessions": len(allowed),
                      "splits": dict(collections.Counter(r["student_split"] for r in selected)),
                      "unavailable_source_rows": unavailable, "output": str(args.output)}))


if __name__ == "__main__":
    main()
