#!/usr/bin/env python3
"""Small independent checks of the stereo report's matching and source clocks."""
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("stereo_report",Path(__file__).with_name("report-stereo-conics.py"))
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
motion_spec=importlib.util.spec_from_file_location("stereo_motion_report",Path(__file__).with_name("report-stereo-motion.py"))
motion_report=importlib.util.module_from_spec(motion_spec)
motion_spec.loader.exec_module(motion_report)
label_spec=importlib.util.spec_from_file_location("stereo_label_report",Path(__file__).with_name("score-stereo-labels.py"))
label_report=importlib.util.module_from_spec(label_spec)
label_spec.loader.exec_module(label_report)


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


def paired_replay(left_first=False,target=(0.0,0.0,100.0)):
    sources=[replay_row(1)["input"],replay_row(2)["input"]]
    sources[1]["frame"].update(eye_id=2,timestamp_ns=sources[0]["frame"]["timestamp_ns"])
    fit=copy.deepcopy(row()["joint"])
    centers=[[-3.0,0.0,-100.0],[3.0,0.0,-100.0]]
    rays=[]
    for center in centers:
        delta=[t-c for t,c in zip(target,center)];length=math.sqrt(sum(v*v for v in delta))
        rays.append([v/length for v in delta])
    fit.update(modeled_eyes=[True,True],contributing_eyes=[True,True],unlocalized_eye_cost=[0.0,0.0],
        target_camera_mm=list(target),eye_centers_camera_mm=centers,eye_gaze_directions=rays,
        cost=2.0,alternative_cost_margin=0.1)
    fit["outer_ellipses"][1]=copy.deepcopy(fit["outer_ellipses"][0])
    fit["withheld_sample_residuals"][0]["groups"][0]["sample_fingerprint"]="same-native-probe"
    fit["withheld_sample_residuals"][1]=copy.deepcopy(fit["withheld_sample_residuals"][0])
    fit["support"]=[{"roi":eye,"kind":"OuterLimbus","used":True} for eye in (1,2)]
    result=[]
    for event,eye in enumerate([1,0] if left_first else [0,1]):
        r=replay_row(event+1);r["input"]=sources[eye]
        if event==1:r.update(joint=fit,publication_inputs=copy.deepcopy(sources))
        result.append(r)
    return result


class MatchingTests(unittest.TestCase):
    def test_viewpoint_depth_metadata_is_complete_and_reconstructs_metric_target(self):
        fit=paired_replay(target=(60.0,-40.0,100.0))[-1]["joint"]
        fit["target_search_chart"]={"frame":"reference-to-camera-tangent-plane",
            "reference_camera_mm":[0.0,0.0,-100.0],"distance_axis":"reference-to-camera",
            "axial_distance_mm":200.0,"slopes":[0.3,-0.2],"slope_limit":1.5}
        report.check_shared_target_contract(fit)
        for field,value in [("distance_axis","camera-optical-z"),("axial_distance_mm",600.0),
                            ("slopes",[1.6,-0.2]),("reference_camera_mm",None)]:
            bad=copy.deepcopy(fit);bad["target_search_chart"][field]=value
            with self.assertRaises(ValueError):report.check_shared_target_contract(bad)
        del fit["target_search_chart"]["axial_distance_mm"]
        with self.assertRaises(ValueError):report.check_shared_target_contract(fit)

    def test_off_axis_near_horizon_depth_cannot_masquerade_as_bounded_axial_distance(self):
        origin=[270.0,160.0,-350.0]
        length=math.sqrt(sum(v*v for v in origin));forward=[-v/length for v in origin]
        direction=[-0.66,-0.75,1e-5]
        norm=math.sqrt(sum(v*v for v in direction));direction=[v/norm for v in direction]
        transverse=math.hypot(forward[0],forward[2]);right=[forward[2]/transverse,0.0,-forward[0]/transverse]
        down=[forward[1]*right[2],forward[2]*right[0]-forward[0]*right[2],-forward[1]*right[0]]
        dot=lambda a,b:sum(x*y for x,y in zip(a,b))
        slopes=[dot(direction,axis)/dot(direction,forward) for axis in [right,down]]
        target=[o+600.0*v/dot(direction,forward) for o,v in zip(origin,direction)]
        fit=paired_replay(target=target)[-1]["joint"]
        fit["target_search_chart"]={"reference_camera_mm":origin,"distance_axis":"reference-to-camera",
            "axial_distance_mm":600.0,"slopes":slopes,"slope_limit":1.5}
        report.check_shared_target_contract(fit)
        old_target=[o+600.0*v/direction[2] for o,v in zip(origin,direction)]
        bad=paired_replay(target=old_target)[-1]["joint"]
        bad["target_search_chart"]=fit["target_search_chart"]
        with self.assertRaises(ValueError):report.check_shared_target_contract(bad)

    def test_source_label_scoring_does_not_invent_missing_sam_or_independent_gaze_results(self):
        r=row();r["joint"]["outer_ellipses"][0].update(center=[0.0,0.0],minor_radius=60.0,angle=0.0)
        label={"annotation_points":[{"kind":"iris_edge","x":80.0,"y":0.0,"visibility":"visible"}]}
        score=label_report.prediction_metrics(label,r,0)
        self.assertEqual(set(score),{"joint"})
        self.assertTrue(score["joint"]["accepted"])
        self.assertLess(score["joint"]["metrics"]["visible"]["rms_px"],1e-6)
        r["joint"]["contributing_eyes"][0]=False
        score=label_report.prediction_metrics(label,r,0)
        self.assertFalse(score["joint"]["accepted"])
        self.assertIsNotNone(score["joint"]["metrics"],"rejected geometry remains labeled diagnostic output, not coverage")

    def test_existing_sam_label_reference_is_preserved_and_partial_export_is_not_silently_accepted(self):
        r={"raw_admitted":[False,False],"baseline_sam_outer":[None,None]}
        self.assertEqual(label_report.prediction_metrics({},r,0),{"SAM":{"accepted":False,"metrics":None}})
        del r["raw_admitted"]
        with self.assertRaises(ValueError):label_report.prediction_metrics({},r,0)

    def test_source_geometry_compares_final_same_read_solutions_independent_of_eye_arrival_order(self):
        result=report.matched_algorithm_report(Rows(paired_replay()),Rows(paired_replay(True)),[],source_order=True)
        self.assertEqual(result["matched_reads"],1)
        counts=result["source_geometry"]["paired"]["counts"]
        self.assertEqual(counts["same_RAW_reads"],1)
        self.assertEqual(counts["both_available_same_publication_evidence"],1)
        self.assertEqual(counts["eye_rays_changed_over_1_degree"],0)
        self.assertEqual(result["probe_verification"]["coordinate_fingerprints_matched"],2)

    def test_identical_source_identity_does_not_hide_changed_shared_gaze_geometry(self):
        result=report.matched_algorithm_report(Rows(paired_replay()),Rows(paired_replay(True,(0.0,250.0,100.0))),[],source_order=True)
        geometry=result["source_geometry"]["paired"]
        self.assertEqual(geometry["counts"]["target_changed_over_1e-6_mm"],1)
        self.assertEqual(geometry["counts"]["eye_rays_changed_over_5_degrees"],2)
        self.assertAlmostEqual(geometry["target_change_mm"]["maximum"],250.0)
        self.assertGreater(geometry["largest_ray_changes"][0]["maximum_common_eye_angle_degrees"],50.0)

    def test_source_geometry_rejects_changed_raw_missing_inputs_and_unseen_publications(self):
        a,b=paired_replay(),paired_replay(True)
        b[0]["input"]["raw_sha256"]="changed"
        b[1]["publication_inputs"][1]["raw_sha256"]="changed"
        with self.assertRaises(ValueError):report.matched_algorithm_report(Rows(a),Rows(b),[],source_order=True)
        with self.assertRaises(ValueError):report.matched_algorithm_report(Rows(a),Rows(a[:1]),[],source_order=True)
        future=paired_replay()
        future[0]["joint"]=future[1]["joint"]
        future[0]["publication_inputs"]=future[1]["publication_inputs"]
        with self.assertRaises(ValueError):list(report.source_read_rows(Rows(future)))

    def test_final_failed_pair_cannot_be_replaced_by_its_first_available_publication(self):
        replay=paired_replay()
        first=copy.deepcopy(replay[1]["joint"])
        first.update(modeled_eyes=[True,False],contributing_eyes=[True,False])
        for field in ("eye_centers_camera_mm","eye_gaze_directions","outer_ellipses","withheld_sample_residuals"):
            first[field][1]=None
        first["support"]=first["support"][:1]
        replay[0].update(joint=first,publication_inputs=[replay[0]["input"],None])
        replay[1]["joint"]={"available":False,"reason":"Conic(NoFeasibleHypothesis)"}
        result=report.matched_algorithm_report(Rows(replay),Rows(copy.deepcopy(replay)),[],source_order=True)
        self.assertEqual(result["matched_reads"],1)
        self.assertEqual(result["source_geometry"]["paired"]["counts"]["available_baseline:False,candidate:False"],1)
        self.assertEqual(result["all_withheld_samples"][0]["matched"],0)

    def test_source_geometry_keeps_unpaired_dropouts_in_the_source_time_area_chain(self):
        start=paired_replay();end=paired_replay()
        for r in end:
            r["input"]["index"]+=4;r["input"]["frame"]["timestamp_ns"]+=20_000_000
            r["input"]["frame"]["sequence"]+=4
        end[1]["publication_inputs"]=[copy.deepcopy(r["input"]) for r in end]
        for e in end[1]["joint"]["outer_ellipses"]:e["major_radius"]*=2.0
        missing=replay_row(3);missing["input"]["frame"]["timestamp_ns"]=20_000_000
        replay=start+[missing]+end
        result=report.matched_algorithm_report(Rows(replay),Rows(copy.deepcopy(replay)),[],source_order=True)
        self.assertEqual(result["matched_reads"],3)
        self.assertEqual(result["source_geometry"]["paired"]["counts"]["same_RAW_reads"],2)
        self.assertEqual(result["source_geometry"]["singleton"]["counts"]["same_RAW_reads"],1)
        self.assertEqual(result["frontal_equivalent_pixel_area_log_steps"][0]["matched"],0)
        self.assertEqual(result["frontal_equivalent_pixel_area_log_steps"][1]["matched"],1)

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
        for count,rms in [(0,None),(8,0.2),(16,__import__("math").pi)]:
            result["support"]=[{"boundary_normal_samples":count,"boundary_normal_rms_radians":rms}]
            report.check_shared_target_contract(result)
        for count,rms in [(0,0.0),(8,None),(17,0.2),(1,4.0),(1,float("nan"))]:
            result["support"]=[{"boundary_normal_samples":count,"boundary_normal_rms_radians":rms}]
            with self.assertRaises(ValueError):report.check_shared_target_contract(result)
        result["support"]=[]
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

    def test_extractor_comparison_skips_changed_eye_but_preserves_unchanged_partner(self):
        for field,value in [("sample_fingerprint","changed"),("points",7),("kind","PupillaryBoundary")]:
            a=row();a["inputs"][1]=copy.deepcopy(a["inputs"][0]);a["inputs"][1]["index"]=2
            a["joint"]["contributing_eyes"]=[True,True]
            a["joint"]["withheld_sample_residuals"][0]["groups"][0]["sample_fingerprint"]="unchanged"
            a["joint"]["withheld_sample_residuals"][1]=copy.deepcopy(a["joint"]["withheld_sample_residuals"][0])
            b=copy.deepcopy(a);b["joint"]["withheld_sample_residuals"][0]["groups"][0][field]=value
            result=report.matched_algorithm_report(Rows([a]),Rows([b]),[],allow_extractor_changes=True)
            self.assertEqual(result["comparison_kind"],"extractor_change")
            self.assertEqual(result["all_withheld_samples"][0]["matched"],0)
            self.assertEqual(result["all_withheld_samples"][1]["matched"],1)
            self.assertEqual(result["probe_verification"]["eye0:changed_probe_sets_skipped"],1)
            self.assertEqual(result["probe_verification"]["coordinate_fingerprints_matched"],1)
            for eye in range(2):self.assertEqual(result["eye_admission"][eye]["baseline:True,candidate:True"],1)

    def test_extractor_mode_cannot_weaken_source_identity_or_accept_unverified_probes(self):
        a,b=row(),row()
        result=report.matched_algorithm_report(Rows([a]),Rows([b]),[],allow_extractor_changes=True)
        self.assertEqual(result["all_withheld_samples"][0]["matched"],0)
        self.assertEqual(result["probe_verification"]["eye0:unverified_probe_sets_skipped"],1)
        b["inputs"][0]["raw_sha256"]="different-bytes"
        with self.assertRaises(ValueError):
            report.matched_algorithm_report(Rows([a]),Rows([b]),[],allow_extractor_changes=True)

    def test_source_replay_retains_identical_outer_probes_when_pupil_extraction_changes(self):
        a=paired_replay();b=copy.deepcopy(a)
        for value in (a,b):
            probe=value[-1]["joint"]["withheld_sample_residuals"][0]
            probe["groups"].append({"arc":1,"group":100,"kind":"PupillaryBoundary","points":4,
                "rms_px":20.0,"used":True,"sample_fingerprint":"old-reflection-edge"})
        candidate=b[-1]["joint"]["withheld_sample_residuals"][0]["groups"]
        candidate[0].update(rms_px=3.0,used=False)
        candidate.pop()
        result=report.matched_algorithm_report(Rows(a),Rows(b),[],allow_extractor_changes=True,source_order=True)
        self.assertEqual(result["all_withheld_samples"][0]["matched"],0)
        self.assertEqual(result["unchanged_outer_limbus_probe_samples"][0]["matched"],1)
        self.assertEqual(result["unchanged_outer_limbus_probe_samples"][0]["candidate_minus_baseline"]["median"],1.0)
        self.assertEqual(result["unchanged_common_accepted_outer_limbus_probe_samples"][0]["matched"],0)
        candidate[0]["sample_fingerprint"]="different-outer-coordinates"
        result=report.matched_algorithm_report(Rows(a),Rows(b),[],allow_extractor_changes=True,source_order=True)
        self.assertEqual(result["unchanged_outer_limbus_probe_samples"][0]["matched"],0)

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
