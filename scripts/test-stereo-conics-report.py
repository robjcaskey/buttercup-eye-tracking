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


def replay_row(index=1):
    source=row(index)["inputs"][0]
    source["frame"].update(eye_id=1,sensor_x=100,sensor_y=200)
    time=source["frame"]["timestamp_ns"]
    return {"schema":"buttercup-joint-source-replay-v1","event":index-1,"input":source,
        "arrival_delay_ns":["0","0"],"logical_arrival_timestamp_ns":str(time),"source_now_ns":str(time),
        "native_roi_reframe":False,"duplicate_suppressed":True,"latest":[None,None],
        "joint":{"available":False,"reason":"Conic(NoBoundaryEvidence)","elapsed_ms":0.1}}


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
        result.update(hypotheses=16,hypotheses_by_association=[8,4,4])
        report.check_shared_target_contract(result)
        result["hypotheses_by_association"]=[16,4,4]
        with self.assertRaises(ValueError):report.check_shared_target_contract(result)
        result["hypotheses_by_association"]=[8,4,4]
        report.check_shared_target_contract(result)
        result["eye_gaze_directions"][1]=[0.6,0.0,0.8]
        with self.assertRaises(ValueError):report.check_shared_target_contract(result)

    def test_source_replay_counts_raw_exposures_and_native_reframes_not_redraws(self):
        a,b=replay_row(1),replay_row(2)
        b["input"]["frame"]["sensor_x"]+=8;b["native_roi_reframe"]=True
        result=report.summarize_source_replay(Rows([a,b]),10)
        self.assertEqual(result["scope"]["unique_input_indices"],2)
        self.assertEqual(result["scope"]["missing_input_count"],8)
        self.assertFalse(result["scope"]["complete_corpus"])
        self.assertEqual(result["counts"]["native_roi_reframes"],1)
        self.assertEqual(result["counts"]["duplicate_suppression_checks"],2)
        self.assertEqual(result["counts"]["fresh_reframe_fits"],0)

    def test_source_replay_rejects_stale_geometry_after_new_empty_evidence(self):
        a,b=replay_row(1),replay_row(2)
        source=a["input"]
        b["latest"][0]={"source":{"roi_id":1,"clock_domain":"1",
            "clock_epoch":str(report.source_clock_epoch(source["clock_lineage"])),
            "timestamp_ns":str(source["frame"]["timestamp_ns"]),"sequence":"1"},
            "contributing":True,"target_camera_mm":[0,0,100]}
        with self.assertRaises(ValueError):report.summarize_source_replay(Rows([a,b]))

    def test_source_replay_rejects_changed_native_time_and_fabricated_reframes(self):
        for field,value in [("source_now_ns","1"),("logical_arrival_timestamp_ns","2"),
                            ("native_roi_reframe",True),("duplicate_suppressed",False)]:
            invalid=replay_row();invalid[field]=value
            with self.assertRaises(ValueError):report.summarize_source_replay(Rows([invalid]))

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
