#!/usr/bin/env python3
"""Small CPU-only tests of source selection and partition leakage defenses."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class StudentIndexTests(unittest.TestCase):
    def test_deduplicates_raw_and_keeps_both_eyes_in_the_same_session_split(self):
        with tempfile.TemporaryDirectory(prefix="student-index-test-", dir="outputs") as directory:
            directory = Path(directory)
            raw = directory / "synthetic.raw10"
            payload = bytes(range(72))
            raw.write_bytes(payload)
            rows = []
            for session in range(12):
                for eye in (1, 2):
                    for frame in range(3):
                        offset = len(rows)
                        rows.append({"raw_file": str(raw.resolve()), "raw_offset": offset,
                                     "raw_length": 1, "raw_sha256": hashlib.sha256(payload[offset:offset + 1]).hexdigest(),
                                     "clock_lineage": f"session:{session}",
                                     "frame": {"eye_id": eye, "timestamp_ns": frame}})
            rows.append(dict(rows[0], clock_lineage="copied-session"))
            index, output = directory / "input.jsonl", directory / "selected.jsonl"
            index.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = subprocess.run([sys.executable, "scripts/prepare-eye-student.py", str(index), str(output),
                                     "--per-eye-session", "2"], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)["frames"], 48)
            selected = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len({row["raw_sha256"] for row in selected}), 48)
            for session in range(12):
                group = [r for r in selected if r["clock_lineage"] == f"session:{session}"]
                self.assertEqual({r["frame"]["eye_id"] for r in group}, {1, 2})
                self.assertEqual(len({r["student_split"] for r in group}), 1)
            self.assertEqual({r["student_split"] for r in selected}, {"train", "validation", "test"})

    def test_fails_if_source_bytes_no_longer_match_the_index(self):
        with tempfile.TemporaryDirectory(prefix="student-index-test-", dir="outputs") as directory:
            directory = Path(directory)
            raw = directory / "synthetic.raw10"
            raw.write_bytes(b"changed")
            row = {"raw_file": str(raw.resolve()), "raw_offset": 0, "raw_length": 7,
                   "raw_sha256": hashlib.sha256(b"correct").hexdigest(),
                   "clock_lineage": "session", "frame": {"eye_id": 1, "timestamp_ns": 1}}
            index = directory / "input.jsonl"
            index.write_text(json.dumps(row) + "\n")
            result = subprocess.run([sys.executable, "scripts/prepare-eye-student.py", str(index),
                                     str(directory / "selected.jsonl")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RAW hash mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
