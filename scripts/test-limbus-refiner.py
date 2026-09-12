#!/usr/bin/env python3
"""Optical landmark semantics: no guessed or imputed anatomical supervision."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).with_name("prepare-limbus-refiner.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
report_spec = importlib.util.spec_from_file_location("report", Path(__file__).with_name("report-limbus-refiner.py"))
report = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(report)


class Landmarks(unittest.TestCase):
    def test_pair_midpoint_is_not_surface_apex(self):
        sample = prepare.observations({"annotation_points": [{"kind": "iris_edge", "visibility": "visible",
            "x": 10, "y": 20, "source": "paired_midpoint", "band_inner": [8, 20], "band_outer": [12, 20]}]})[0]
        self.assertNotIn("surface_apex", sample["targets"])
        self.assertEqual(sample["weights"]["rim"], 0.3)

    def test_submerged_landmark_never_replaces_the_apex(self):
        sample = prepare.observations({"annotation_points": [{"kind": "iris_edge", "visibility": "visible",
            "x": 10, "y": 20, "source": "visibility_triplet_apex", "iris_side_onset": [8, 20],
            "subsurface_visibility_limit": [15, 20]}]})[0]
        self.assertEqual(sample["targets"]["rim"], [10, 20])
        self.assertEqual(sample["targets"]["surface_apex"], [10, 20])
        self.assertEqual(sample["targets"]["subsurface_limit"], [15, 20])

    def test_null_landmarks_are_not_imputed_coordinates(self):
        sample = prepare.observations({"limbus_partial_observations": [{"mode": "triplet", "landmarks": {
            "inner": {"x": 8, "y": 20, "visibility": "visible"}, "apex": None, "submerged": None}}]})[0]
        self.assertEqual(sample["targets"], {"iris_onset": [8, 20]})
        self.assertIn("surface_apex", sample["occluded"])

    def test_guessed_landmarks_do_not_train_the_visible_boundary(self):
        self.assertEqual(prepare.observations({"annotation_points": [{"kind": "iris_edge", "visibility": "guessed",
            "x": 10, "y": 20}]}), [])


class AreaEvidence(unittest.TestCase):
    def sample(self, sequence, timestamp):
        return {"source": {"raw_sha256": str(sequence), "clock_attested": True, "clock_lineage": "sensor-a",
                           "frame": {"eye_id": 0, "sequence": sequence, "timestamp_ns": timestamp,
                                     "sensor_x": 200, "sensor_y": 800}},
                "baseline_raw_admitted": True, "candidate": {"major_radius": 80},
                "sn_feida": {"baseline": 100, "candidate": 100, "scale_reference": 12, "units": "reference_px2"}}

    def test_area_pairs_never_bridge_missing_candidate_or_reference_reset(self):
        a, b, c = [self.sample(i, i * 100_000_000) for i in range(1, 4)]
        self.assertEqual(len(report.sequence_pairs([a, b, c])), 2)
        b["candidate"] = None
        self.assertEqual(report.sequence_pairs([a, b, c]), [])
        b["candidate"] = {"major_radius": 80}
        b["sn_feida"]["scale_reference"] = 13
        self.assertEqual(report.sequence_pairs([a, b, c]), [])

    def test_crop_move_is_not_area_or_clock_change(self):
        a, b = self.sample(1, 100_000_000), self.sample(2, 200_000_000)
        b["source"]["frame"]["sensor_y"] += 24
        pair = report.sequence_pairs([a, b])[0]
        self.assertTrue(pair["roi_reframed"])
        self.assertEqual(pair["candidate_absolute_log_step"], 0)
        b["source"]["clock_lineage"] = "sensor-b"
        self.assertEqual(report.sequence_pairs([a, b]), [])

    def test_held_out_scoring_rejects_training_leakage(self):
        row = self.sample(1, 100_000_000)
        row.update(split="test", group=0)
        with self.assertRaises(ValueError):
            report.summarize_cv([{"model_training": {"train_raw_sha256": ["1"]}, "frames": [row]}])


if __name__ == "__main__":
    unittest.main()
