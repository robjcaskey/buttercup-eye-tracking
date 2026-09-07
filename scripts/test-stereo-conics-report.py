#!/usr/bin/env python3
"""Small independent checks of the stereo report's matching and source clocks."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("stereo_report",Path(__file__).with_name("report-stereo-conics.py"))
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
motion_spec=importlib.util.spec_from_file_location("stereo_motion_report",Path(__file__).with_name("report-stereo-motion.py"))
motion_report=importlib.util.module_from_spec(motion_spec)
motion_spec.loader.exec_module(motion_report)


class Rows:
    def __init__(self,rows):
        self.rows=rows

    def open(self):
        return io.StringIO("".join(json.dumps(row)+"\n" for row in self.rows))


def row(index=1):
    source={"index":index,"raw_sha256":"same-native-bytes","clock_lineage":"sensor-epoch-1",
        "frame":{"timestamp_ns":index*10_000_000,"sequence":index,"width":420,"height":280,"stride":525}}
    probe={"rms_px":2.0,"groups":[{"arc":0,"group":0,"kind":"OuterLimbus","points":8,"rms_px":2.0,"used":True}]}
    fit={"available":True,"contributing_eyes":[True,False],"withheld_sample_residuals":[probe,None],
        "outer_ellipses":[{"major_radius":80.0},None],"sn_feida_mm2":[None,None]}
    return {"inputs":[source,None],"joint":fit}


class MatchingTests(unittest.TestCase):
    def test_unlocalized_roi_exports_no_geometry_but_keeps_its_rejection_cost(self):
        result={"available":True,"modeled_eyes":[True,False],"unlocalized_eye_cost":[0.0,3.0],
            "target_camera_mm":[0.0,0.0,100.0],"eye_centers_camera_mm":[[0.0,0.0,-100.0],None],
            "eye_gaze_directions":[[0.0,0.0,1.0],None],"outer_ellipses":[{"major_radius":20.0},None],
            "contributing_eyes":[True,False]}
        report.check_shared_target_contract(result)
        for field,value in [("eye_centers_camera_mm",[0.0,0.0,-100.0]),("eye_gaze_directions",[0.0,0.0,1.0]),
                            ("outer_ellipses",{"major_radius":20.0}),("contributing_eyes",True)]:
            invalid=copy.deepcopy(result);invalid[field][1]=value
            with self.assertRaises(ValueError):report.check_shared_target_contract(invalid)

    def test_exported_eye_rays_really_share_the_declared_fixation(self):
        result={"available":True,"modeled_eyes":[True,True],"unlocalized_eye_cost":[0.0,0.0],
            "target_camera_mm":[0.0,0.0,100.0],"eye_centers_camera_mm":[[-3.0,0.0,96.0],[3.0,0.0,96.0]],
            "eye_gaze_directions":[[0.6,0.0,0.8],[-0.6,0.0,0.8]],"outer_ellipses":[{},{}],
            "contributing_eyes":[True,True]}
        report.check_shared_target_contract(result)
        result["eye_gaze_directions"][1]=[0.6,0.0,0.8]
        with self.assertRaises(ValueError):report.check_shared_target_contract(result)

    def test_scale_uses_determinant_and_never_the_candidate_radius(self):
        a={"width":420,"height":280,"timestamp_ns":10,"sequence":1}
        b=dict(a,timestamp_ns=20,sequence=2,shared_global_scale={"reliable":True,"motion_support":16,
            "motion_residual":1.0,"stable_frames":3,"occupied_quadrants":4,"scale_delta":0.0,"rotation":0.1})
        scale,allowance=motion_report.scale_link(a,b)
        self.assertAlmostEqual(scale,(1.0+0.1**2)**0.5)
        self.assertNotEqual(scale,1.0)
        self.assertAlmostEqual(motion_report.normalized_log_change(100,100*scale,scale),0.0)
        self.assertAlmostEqual(motion_report.normalized_log_change(100,120,1.0),2*__import__("math").log(1.2))
        for field,value in [("reliable",False),("motion_support",0),("occupied_quadrants",1),("scale_delta",0.5)]:
            invalid=copy.deepcopy(b)
            invalid["shared_global_scale"][field]=value
            self.assertIsNone(motion_report.scale_link(a,invalid))

    def test_matches_sources_and_does_not_score_a_rejected_diagnostic_ellipse(self):
        a,b=row(),row()
        b["joint"]["contributing_eyes"][0]=False
        result=report.matched_algorithm_report(Rows([a]),Rows([b]),[])
        self.assertEqual(result["eye_admission"][0]["baseline:True,candidate:False"],1)
        self.assertEqual(result["all_withheld_samples"][0]["matched"],0)

    def test_source_identity_and_probe_changes_fail_explicitly(self):
        for change in ("raw_sha256","points","kind"):
            a,b=row(),row()
            if change=="raw_sha256":
                b["inputs"][0][change]="different-native-bytes"
            else:
                b["joint"]["withheld_sample_residuals"][0]["groups"][0][change]=3 if change=="points" else "PupillaryBoundary"
            with self.assertRaises(ValueError):
                report.matched_algorithm_report(Rows([a]),Rows([b]),[])

    def test_missing_source_read_does_not_silently_shrink_comparison(self):
        with self.assertRaises(ValueError):
            report.matched_algorithm_report(Rows([row()]),Rows([]),[])

    def test_equal_point_counts_do_not_hide_changed_validation_coordinates(self):
        a,b=row(),row()
        a["joint"]["withheld_sample_residuals"][0]["groups"][0]["sample_fingerprint"]="coordinates-A"
        b["joint"]["withheld_sample_residuals"][0]["groups"][0]["sample_fingerprint"]="coordinates-B"
        with self.assertRaises(ValueError):
            report.matched_algorithm_report(Rows([a]),Rows([b]),[])

    def test_index_ranges_are_explicit_and_end_exclusive(self):
        rows=[row(i) for i in (1,2,3)]
        result=report.matched_algorithm_report(Rows(rows),Rows(copy.deepcopy(rows)),[(2,3)])
        self.assertEqual(result["matched_reads"],1)

    def test_late_emitted_missing_singleton_breaks_source_time_area_chain(self):
        dimensions=(420,280,525)
        timeline=[("clock",10,1,dimensions,[10.0,10.0],[1.0,1.0]),
            ("clock",30,3,dimensions,[20.0,20.0],[4.0,4.0]),
            ("clock",20,2,dimensions,None,None)]
        self.assertEqual(report.adjacent_area_steps(timeline),([],[]))
        timeline[2]=("clock",20,2,dimensions,[15.0,15.0],[2.25,2.25])
        pixel,normalized=report.adjacent_area_steps(timeline)
        self.assertEqual(len(pixel),2)
        self.assertEqual(len(normalized),2)

    def test_clock_geometry_and_repeated_source_do_not_create_scale_samples(self):
        a=("clock",10,1,(420,280,525),[10.0,10.0],[None,None])
        for b in [("different",20,2,a[3],a[4],a[5]),("clock",20,2,(384,256,480),a[4],a[5]),
            ("clock",10,1,a[3],a[4],a[5])]:
            self.assertEqual(report.adjacent_area_steps([a,b]),([],[]))


if __name__=="__main__":
    unittest.main()
