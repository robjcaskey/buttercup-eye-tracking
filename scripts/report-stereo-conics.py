#!/usr/bin/env python3
"""Summarize matched conic replays, including regressions and scope shortfalls.

This evaluator does not equate cross-eye agreement or smoothness with true gaze
accuracy. Withheld points are conditional on upstream SAM segmentation and its
initial conic hypotheses. External label scoring is a separate, post-fit step.
"""
import argparse
import collections
import contextlib
import json
import math
import itertools
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"count": 0}
    return {"count": len(values), "median": statistics.median(values),
            "p95": values[round((len(values)-1)*0.95)], "maximum": values[-1]}


def adjacent_area_steps(timeline):
    """Reports emit pairs before pending singletons, not in exposure order."""
    pixel_steps,normalized_steps=[],[]
    previous=None
    for clock,timestamp,sequence,dimensions,radii,normalized in sorted(timeline,key=lambda r:(r[0],r[1],r[2])):
        current=(clock,timestamp,sequence,dimensions,radii,normalized)
        if previous and radii and previous[4] and clock==previous[0] and dimensions==previous[3] \
                and 0<timestamp-previous[1]<=500_000_000 and sequence!=previous[2]:
            pixel_steps.append(tuple(abs(2*math.log(r/previous_r)) for r,previous_r in zip(radii,previous[4])))
            if all(v is not None and v>0 for v in normalized+previous[5]):
                normalized_steps.append(tuple(abs(math.log(v/previous_v)) for v,previous_v in zip(normalized,previous[5])))
        # Missing/rejected observations explicitly break the chain, including
        # when that single-ROI read was emitted after the complete pairs.
        previous=current
    return pixel_steps,normalized_steps


def check_shared_target_contract(result):
    """Check emitted geometry consistency, not anatomical/gaze accuracy.

    Older exports did not declare modeled ROIs; do not infer that declaration
    from contribution flags (an outlier-only modeled ROI is different).
    """
    if not result.get("available") or "modeled_eyes" not in result:
        return
    if "hypotheses_by_association" in result:
        work=result["hypotheses_by_association"]
        if len(work)!=3 or any(type(v) is not int or v<0 for v in work) \
                or sum(work)!=result["hypotheses"] or sum(work)>24:
            raise ValueError("association searches did not share the declared bounded work budget")
    for arc in result.get("support",[]):
        if "boundary_normal_samples" not in arc:
            continue
        count=arc["boundary_normal_samples"]
        rms=arc.get("boundary_normal_rms_radians")
        if type(count) is not int or not 0<=count<=16 or (count==0 and rms is not None) \
                or (count>0 and (not isinstance(rms,(int,float)) or not math.isfinite(rms) or not 0<=rms<=math.pi+1e-12)):
            raise ValueError("invalid image-boundary normal diagnostic; not a 3D surface normal or an unbounded sample vote")
    modeled=result["modeled_eyes"]
    penalties=result["unlocalized_eye_cost"]
    if len(modeled)!=2 or any(type(v) is not bool for v in modeled) or len(penalties)!=2 \
            or any(not math.isfinite(v) or v<0 for v in penalties):
        raise ValueError("invalid modeled-ROI declaration or unlocalized evidence cost")
    target=result["target_camera_mm"]
    if len(target)!=3 or not all(math.isfinite(v) for v in target):
        raise ValueError("shared target is not a finite 3D point")
    chart=result.get("target_search_chart") or {}
    if any(k in chart for k in ("reference_camera_mm","axial_distance_mm","distance_axis")):
        origin=chart.get("reference_camera_mm")
        axial=chart.get("axial_distance_mm")
        slopes=chart.get("slopes")
        limit=chart.get("slope_limit")
        if chart.get("distance_axis")!="reference-to-camera" \
                or not isinstance(origin,list) or len(origin)!=3 or not all(math.isfinite(v) for v in origin) or origin[2]>=0 \
                or not isinstance(axial,(int,float)) or not math.isfinite(axial) or axial<=0 \
                or not isinstance(slopes,list) or len(slopes)!=2 or not all(math.isfinite(v) for v in slopes) \
                or not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit<=0 \
                or any(abs(v)>limit+1e-7 for v in slopes):
            raise ValueError("invalid viewpoint axial-distance chart")
        length=math.sqrt(sum(v*v for v in origin))
        forward=[-v/length for v in origin]
        transverse=math.hypot(forward[0],forward[2])
        right=[forward[2]/transverse,0.0,-forward[0]/transverse]
        down=[forward[1]*right[2],forward[2]*right[0]-forward[0]*right[2],-forward[1]*right[0]]
        expected=[o+axial*(f+slopes[0]*r+slopes[1]*d) for o,f,r,d in zip(origin,forward,right,down)]
        if max(abs(a-b) for a,b in zip(target,expected))>1e-6*max(1.0,axial):
            raise ValueError("viewpoint slopes and axial distance do not reconstruct the bounded shared target")
    for eye in range(2):
        center=result["eye_centers_camera_mm"][eye]
        ray=result["eye_gaze_directions"][eye]
        ellipse=result["outer_ellipses"][eye]
        if not modeled[eye]:
            if center is not None or ray is not None or ellipse is not None or result["contributing_eyes"][eye]:
                raise ValueError("unlocalized ROI exported fabricated geometry or an admitted observation")
            continue
        if penalties[eye]!=0 or center is None or ray is None or len(center)!=3 or len(ray)!=3:
            raise ValueError("modeled ROI has missing geometry or an unlocalized penalty")
        delta=[t-c for t,c in zip(target,center)]
        distance=math.sqrt(sum(v*v for v in delta))
        if not math.isfinite(distance) or distance<=0 or not all(math.isfinite(v) for v in ray) \
                or max(abs(v/distance-r) for v,r in zip(delta,ray))>1e-7:
            raise ValueError("exported gaze ray does not point to the single shared target")


def source_clock_epoch(lineage):
    value=14695981039346656037
    for byte in lineage.encode():
        value=((value^byte)*1099511628211)&((1<<64)-1)
    return value


def summarize_source_replay(evaluation,expected=None):
    """Arrival-order validation. Publications are NOT additional exposures."""
    counts=collections.Counter()
    identities={}
    reads=collections.defaultdict(set)
    latest_observed={}
    previous_geometry={}
    completed_clocks=set()
    captures=set()
    current_clock=None
    previous_arrival=0
    delays=None
    elapsed=[]
    def identity(source):
        frame=source["frame"]
        return (source["clock_lineage"],source.get("raw_sha256"),
            *(int(frame[k]) for k in ("timestamp_ns","eye_id","sequence","sensor_x","sensor_y","width","height","stride")))
    with evaluation.open() as stream:
        for event,line in enumerate(stream):
            row=json.loads(line)
            if row.get("schema")!="buttercup-joint-source-replay-v1" or row["event"]!=event:
                raise ValueError("invalid source-replay event order/schema")
            source=row["input"];frame=source["frame"]
            index=source["index"];clock=source["clock_lineage"]
            eye=int(frame["eye_id"])-1;time=int(frame["timestamp_ns"])
            if eye not in (0,1) or index in identities or eye in reads[clock,time]:
                raise ValueError("duplicate or invalid source exposure in replay coverage")
            identities[index]=identity(source);reads[clock,time].add(eye)
            captures.add(source.get("capture_entry"))
            if clock!=current_clock:
                if clock in completed_clocks:raise ValueError("unrelated clock lineages interleaved")
                if current_clock is not None:completed_clocks.add(current_clock)
                current_clock=clock;previous_arrival=0
            current_delays=[int(v) for v in row["arrival_delay_ns"]]
            if delays is None:delays=current_delays
            if current_delays!=delays:raise ValueError("arrival scenario changed mid-replay")
            arrival=int(row["logical_arrival_timestamp_ns"])
            if arrival!=time+delays[eye] or arrival<previous_arrival:
                raise ValueError("arrival scheduling changed source time or went backwards")
            previous_arrival=arrival
            key=(clock,eye)
            old=latest_observed.get(key)
            if old is None or time>old[0]:latest_observed[key]=(time,int(frame["sequence"]))
            geometry=(time,frame["sensor_x"],frame["sensor_y"],frame["width"],frame["height"])
            prior=previous_geometry.get(key)
            reframe=bool(prior and 0<time-prior[0]<=500_000_000 and geometry[3:]==prior[3:]
                and geometry[1:3]!=prior[1:3])
            if reframe!=row["native_roi_reframe"]:raise ValueError("incorrect native ROI-reframe declaration")
            previous_geometry[key]=geometry
            counts["native_roi_reframes"]+=reframe
            fit=row["joint"];elapsed.append(fit["elapsed_ms"])
            counts["available_publications"]+=fit.get("available",False)
            if fit.get("available"):
                check_shared_target_contract(fit)
                inputs=row["publication_inputs"]
                for p in inputs:
                    if p is not None and (identities.get(p["index"])!=identity(p)
                            or p["clock_lineage"]!=clock or int(p["frame"]["timestamp_ns"])!=time):
                        raise ValueError("publication used altered, future, or cross-read evidence")
                counts["paired_publications"]+=all(inputs)
                counts["both_eyes_contributing_publications"]+=all(fit["contributing_eyes"])
                counts["fresh_reframe_fits"]+=reframe and fit["contributing_eyes"][eye]
            else:
                counts["unavailable:"+fit["reason"]]+=1
            duplicate=row["duplicate_suppressed"]
            if not duplicate and "OutsideSourceWindow" not in fit.get("reason",""):
                raise ValueError("a retained duplicate re-entered the solver")
            counts["duplicate_suppression_checks"]+=duplicate
            now=int(row["source_now_ns"])
            if now!=max(t for (c,_),(t,_) in latest_observed.items() if c==clock):
                raise ValueError("latest-state age used arrival time instead of native source time")
            for slot,latest in enumerate(row["latest"]):
                if latest is None:continue
                s=latest["source"]
                if int(s["clock_domain"])!=1 or int(s["clock_epoch"])!=source_clock_epoch(clock) \
                        or int(s["roi_id"])!=slot+1 \
                        or (int(s["timestamp_ns"]),int(s["sequence"]))!=latest_observed.get((clock,slot)) \
                        or not 0<=now-int(s["timestamp_ns"])<=500_000_000:
                    raise ValueError("late completion restored invalidated, foreign, future, or expired live geometry")
                counts["latest_current_source_checks"]+=1
    unexpected=[i for i in identities if expected is not None and not 0<=i<expected]
    missing=expected-len(identities)+len(unexpected) if expected is not None else None
    return {"schema":"buttercup-joint-source-replay-report-v1","evaluation":str(evaluation),
        "scope":{"unique_input_indices":len(identities),"reads":len(reads),
            "both_raw_reads":sum(len(eyes)==2 for eyes in reads.values()),"capture_entries":len(captures),
            "clock_lineages":len(completed_clocks)+(current_clock is not None),
            "expected_input_indices":expected,"missing_input_count":missing,"unexpected_indices":unexpected,
            "complete_corpus":expected is not None and missing==0 and not unexpected},
        "arrival_delay_ns":delays,"counts":counts,"tracker_elapsed_ms":distribution(elapsed),
        "limitations":["Synthetic arrival delays only; no measured detector latency or SAM video-memory reproduction.",
            "Repeated same-exposure publications are not new observations; coverage counts unique arriving RAW sources.",
            "Source freshness, pairing and duplicate checks do not establish limbus localization, sign or gaze accuracy.",
            "Native ROI reframes are counted, not certified accurate by source-continuity checks."]}


def evaluation_rows(evaluation):
    with evaluation.open() as stream:
        for line in stream:
            yield json.loads(line)


def source_read_rows(evaluation):
    """One final result per physical read, including failed pairs and singletons.

    With the replay's constant per-eye delay, second arrivals have a common
    source-time ordering. First publications are not additional observations.
    Pending genuine singletons are emitted last in deterministic source order;
    the area diagnostic independently sorts source time, preserving dropouts.
    This offline join is not memory retained by the real-time solver.
    """
    pending={}
    complete=set()
    indices=set()
    delays=None
    previous_pair=None
    with contextlib.closing(evaluation_rows(evaluation)) as rows:
        for row in rows:
            if row.get("schema")!="buttercup-joint-source-replay-v1":
                raise ValueError("geometry comparison requires native source-replay rows")
            source=row["input"];frame=source["frame"]
            key=(source["clock_lineage"],int(frame["timestamp_ns"]))
            eye=int(frame["eye_id"])-1
            current_delays=tuple(int(v) for v in row["arrival_delay_ns"])
            if delays is None:delays=current_delays
            if delays!=current_delays:
                raise ValueError("source-read joining requires constant per-eye arrival delays")
            if eye not in (0,1) or source["index"] in indices or key in complete:
                raise ValueError("duplicate/conflicting source in final-read geometry comparison")
            indices.add(source["index"])
            result=pending.setdefault(key,{"inputs":[None,None]})
            if result["inputs"][eye] is not None:
                raise ValueError("duplicate ROI in one physical source read")
            result["inputs"][eye]=source
            result["joint"]=row["joint"]
            result["publication_input_indices"]=None
            if row["joint"].get("available"):
                publication=row["publication_inputs"]
                if len(publication)!=2 or not any(publication):
                    raise ValueError("available final-read geometry has no source evidence")
                for slot,p in enumerate(publication):
                    if p is not None and p!=result["inputs"][slot]:
                        raise ValueError("final-read publication changed RAW identity or used an unseen partner")
                result["publication_input_indices"]=[p["index"] if p else None for p in publication]
            if all(result["inputs"]):
                order=(source_clock_epoch(key[0]),key[1])
                if previous_pair is not None and order<=previous_pair:
                    raise ValueError("paired completions are not source ordered under a constant arrival delay")
                previous_pair=order
                complete.add(key)
                yield pending.pop(key)
    for key in sorted(pending,key=lambda k:(source_clock_epoch(k[0]),k[1])):
        yield pending[key]


class JointGeometryComparison:
    """Compare each eye with itself across runs, never average two gaze points."""
    def __init__(self):
        self.counts=collections.Counter()
        self.target_changes=[]
        self.ray_angles=[[],[]]
        self.largest=[]

    def observe(self,a,b):
        self.counts["same_RAW_reads"]+=1
        fits=[r["joint"] for r in (a,b)]
        available=[bool(f.get("available")) for f in fits]
        self.counts[f"available_baseline:{available[0]},candidate:{available[1]}"]+=1
        for fit in fits:check_shared_target_contract(fit)
        if not all(available):return
        if a["publication_input_indices"]!=b["publication_input_indices"]:
            self.counts["different_publication_evidence_geometry_skipped"]+=1
            return
        self.counts["both_available_same_publication_evidence"]+=1
        self.counts["contributing_eyes_changed"]+=fits[0]["contributing_eyes"]!=fits[1]["contributing_eyes"]
        target_delta=math.dist(*(f["target_camera_mm"] for f in fits))
        self.target_changes.append(target_delta)
        self.counts["target_changed_over_1e-6_mm"]+=target_delta>1e-6
        angles=[]
        for eye in range(2):
            if not all(f["contributing_eyes"][eye] for f in fits):continue
            u,v=(f["eye_gaze_directions"][eye] for f in fits)
            cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
            angle=math.degrees(math.atan2(math.sqrt(sum(x*x for x in cross)),sum(x*y for x,y in zip(u,v))))
            self.ray_angles[eye].append(angle);angles.append(angle)
            self.counts["eye_rays_changed_over_1_degree"]+=angle>1.0
            self.counts["eye_rays_changed_over_5_degrees"]+=angle>5.0
        maximum=max(angles,default=0.0)
        if maximum>1e-7 or target_delta>1e-6:
            self.largest.append({"indices":[s["index"] if s else None for s in a["inputs"]],
                "maximum_common_eye_angle_degrees":maximum,"target_delta_mm":target_delta,
                "baseline_contributing":fits[0]["contributing_eyes"],"candidate_contributing":fits[1]["contributing_eyes"],
                "baseline_cost":fits[0]["cost"],"candidate_cost":fits[1]["cost"],
                "baseline_margin":fits[0]["alternative_cost_margin"],"candidate_margin":fits[1]["alternative_cost_margin"]})

    def report(self):
        return {"counts":self.counts,"target_change_mm":distribution(self.target_changes),
            "same_eye_ray_change_degrees":[distribution(v) for v in self.ray_angles],
            "largest_ray_changes":sorted(self.largest,key=lambda r:(-r["maximum_common_eye_angle_degrees"],-r["target_delta_mm"]))[:30],
            "limitations":["Same native RAW/extraction is necessary but not anatomical gaze ground truth.",
                "Changing the arrival order can legitimately change which past observations are available. Paired initialization must not depend on presentation slots when the same prior joint evidence is available.",
                "A competing-hypothesis cost margin is heuristic separation, not calibrated sign confidence."]}


def matched_algorithm_report(baseline, candidate, index_ranges, allow_extractor_changes=False, source_order=False):
    """Stream exact source-matched rows; never silently compare different probes.

    Default: optimizer-only A/B trials with the same extraction. The explicit
    extractor-change mode still requires exact RAW/source identity, but skips
    per-eye residual comparisons unless all probe coordinates are verified
    unchanged. Admission and source-time area comparisons retain both eyes.
    """
    coverage=[collections.Counter(),collections.Counter()]
    residuals=[[],[]]
    supported=[[],[]]
    unchanged_outer=[[],[]]
    unchanged_supported_outer=[[],[]]
    area_steps=[[],[]]
    normalized_steps=[[],[]]
    timeline=[[],[]]
    regressions=[]
    probe_verification=collections.Counter()
    geometry={kind:JointGeometryComparison() for kind in ("paired","singleton")} if source_order else {}
    rows=0
    def in_scope(row):
        return not index_ranges or all(any(lo<=item["index"]<hi for lo,hi in index_ranges)
            for item in row["inputs"] if item)
    read_rows=source_read_rows if source_order else evaluation_rows
    with contextlib.closing(read_rows(baseline)) as first,contextlib.closing(read_rows(candidate)) as second:
        for a,b in itertools.zip_longest(first,second):
            if a is None or b is None:
                raise ValueError("baseline/candidate source-read counts differ")
            if a["inputs"]!=b["inputs"]:
                raise ValueError("baseline/candidate source identity or row order differs")
            if not in_scope(b):
                continue
            rows+=1
            if source_order:geometry["paired" if all(a["inputs"]) else "singleton"].observe(a,b)
            for eye,source in enumerate(b["inputs"]):
                if source is None:
                    continue
                fits=[row.get("joint",{}) for row in (a,b)]
                admitted=[fit.get("available",False) and fit["contributing_eyes"][eye] for fit in fits]
                coverage[eye][f"baseline:{admitted[0]},candidate:{admitted[1]}"]+=1
                frame=source["frame"]
                dimensions=tuple(frame.get(k) for k in ("width","height","stride"))
                observed_outer=all(admitted) and all(any(a["roi"]==eye+1 and a["kind"]=="OuterLimbus" and a["used"]
                    for a in fit.get("support",[])) for fit in fits)
                radii=[fit["outer_ellipses"][eye]["major_radius"] for fit in fits] if observed_outer else None
                normalized=[fit["sn_feida_mm2"][eye] for fit in fits] if observed_outer else None
                timeline[eye].append((source["clock_lineage"],int(frame["timestamp_ns"]),int(frame["sequence"]),dimensions,radii,normalized))
                if not all(admitted):
                    continue
                probes=[fit["withheld_sample_residuals"][eye] for fit in fits]
                if all(probes):
                    # Pupil photometry can change while the independent SAM
                    # limbus samples remain byte-identical. Keep this explicit
                    # fixed outer-probe metric instead of either comparing new
                    # pupil probes with old ones or discarding the whole eye.
                    outer_sets=[{(g["group"],g["points"],g["sample_fingerprint"]):g for g in p["groups"]
                        if g["kind"]=="OuterLimbus" and g.get("sample_fingerprint")} for p in probes]
                    common_outer=[(g,outer_sets[1][key]) for key,g in outer_sets[0].items() if key in outer_sets[1]]
                    for pairs,destination in [(common_outer,unchanged_outer[eye]),
                            ([pair for pair in common_outer if all(g["used"] for g in pair)],unchanged_supported_outer[eye])]:
                        if pairs:
                            count=sum(pair[0]["points"] for pair in pairs)
                            destination.append(tuple(math.sqrt(sum(pair[i]["rms_px"]**2*pair[i]["points"]
                                for pair in pairs)/count) for i in (0,1)))
                if all(p and p["rms_px"] is not None for p in probes):
                    keys=lambda p:[(g["arc"],g["group"],g["kind"],g["points"]) for g in p["groups"]]
                    changed=keys(probes[0])!=keys(probes[1]) or any(
                        ga.get("sample_fingerprint") and gb.get("sample_fingerprint")
                        and ga["sample_fingerprint"]!=gb["sample_fingerprint"]
                        for ga,gb in zip(probes[0]["groups"],probes[1]["groups"]))
                    if changed:
                        if not allow_extractor_changes:
                            raise ValueError("A/B withheld probe structure or coordinates changed; optimizer-only comparison is invalid")
                        probe_verification[f"eye{eye}:changed_probe_sets_skipped"]+=1
                        continue
                    if allow_extractor_changes and any(not g.get("sample_fingerprint")
                            for p in probes for g in p["groups"]):
                        probe_verification[f"eye{eye}:unverified_probe_sets_skipped"]+=1
                        continue
                    for ga,gb in zip(probes[0]["groups"],probes[1]["groups"]):
                        if ga.get("sample_fingerprint") and gb.get("sample_fingerprint"):
                            probe_verification["coordinate_fingerprints_matched"]+=1
                        else:
                            probe_verification["legacy_structure_only_checks"]+=1
                    residuals[eye].append(tuple(p["rms_px"] for p in probes))
                    common=[(ga,gb) for ga,gb in zip(probes[0]["groups"],probes[1]["groups"])
                        if ga["used"] and gb["used"]]
                    if common:
                        count=sum(ga["points"] for ga,gb in common)
                        supported[eye].append(tuple(math.sqrt(sum(pair[i]["rms_px"]**2*pair[i]["points"] for pair in common)/count) for i in (0,1)))
                    regressions.append({"eye":eye,"index":source["index"],"baseline_rms_px":probes[0]["rms_px"],
                        "candidate_rms_px":probes[1]["rms_px"],"delta_rms_px":probes[1]["rms_px"]-probes[0]["rms_px"]})
                elif allow_extractor_changes:
                    probe_verification[f"eye{eye}:missing_probe_sets_skipped"]+=1
    for eye in range(2):
        area_steps[eye],normalized_steps[eye]=adjacent_area_steps(timeline[eye])
    def comparison(pairs):
        return {"matched":len(pairs),"baseline":distribution([a for a,b in pairs]),
            "candidate":distribution([b for a,b in pairs]),"candidate_minus_baseline":distribution([b-a for a,b in pairs]),
            "improved_over_1px":sum(b<a-1 for a,b in pairs),"regressed_over_1px":sum(b>a+1 for a,b in pairs)}
    return {"baseline":str(baseline),"candidate":str(candidate),"matched_reads":rows,"index_ranges":index_ranges,
        **({"source_geometry":{key:value.report() for key,value in geometry.items()}} if source_order else {}),
        "comparison_kind":"extractor_change" if allow_extractor_changes else "optimizer_only",
        "probe_verification":probe_verification,
        "eye_admission":coverage,"all_withheld_samples":[comparison(p) for p in residuals],
        "common_accepted_arc_samples":[comparison(p) for p in supported],
        "unchanged_outer_limbus_probe_samples":[comparison(p) for p in unchanged_outer],
        "unchanged_common_accepted_outer_limbus_probe_samples":[comparison(p) for p in unchanged_supported_outer],
        "frontal_equivalent_pixel_area_log_steps":[{"matched":len(p),"baseline":distribution([a for a,b in p]),
            "candidate":distribution([b for a,b in p])} for p in area_steps],
        "independent_SN_FEIDA_log_steps":[{"matched":len(p),"baseline":distribution([a for a,b in p]),
            "candidate":distribution([b for a,b in p])} for p in normalized_steps],
        "largest_regressions":sorted(regressions,key=lambda r:-r["delta_rms_px"])[:30],
        "limitations":[("Extractor-change comparison: changed/missing/unverified per-eye probes are explicitly skipped, never scored against each other; unchanged partner-eye probes remain comparable. Assess changed-eye localization with independent labels, not this selected probe intersection."
            if allow_extractor_changes else "Optimizer-only comparison: unchanged extraction/withheld coordinates must be verified separately for old exports lacking probe hashes."),
            "Acquisition scale is candidate-independent but held between MediaPipe updates under an assumed 12mm limbus; normalized steps are conditional on that prior, not fresh physical-scale measurements.",
            "Pixel-area steps are unnormalized, include genuine motion, and do not establish physical area stability.",
            "Short adjacent same-clock intervals only; no bridging absent fits or treating held source exposures as new.",
            "Limbus localization and gaze/pose ground truth remain separate post-fit validations."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--source-order-replay", action="store_true")
    parser.add_argument("--allow-extractor-changes", action="store_true",
        help="explicitly skip changed/unverified per-eye probes while retaining source-matched admission and area comparisons")
    parser.add_argument("--index-range", type=int, nargs=2, action="append", default=[], metavar=("START", "END"),
        help="optional source-index intervals for the separate optimizer A/B report; END is exclusive")
    args = parser.parse_args()
    if args.allow_extractor_changes and not args.baseline_evaluation:
        parser.error("allow-extractor-changes requires a baseline evaluation")
    if args.source_order_replay:
        if args.index_range:
            parser.error("source-order reports require the full matched replay scope")
        expected=json.loads(args.expected_manifest.read_text())["summary"]["unique_raw_frames"] if args.expected_manifest else None
        report=summarize_source_replay(args.evaluation,expected)
        if args.baseline_evaluation:
            report["baseline_source_audit"]=summarize_source_replay(args.baseline_evaluation,expected)
            report["matched_algorithm"]=matched_algorithm_report(args.baseline_evaluation,args.evaluation,[],
                allow_extractor_changes=args.allow_extractor_changes,source_order=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        return
    methods = ("joint", "monocular_right", "monocular_left")
    summary = {m: {"available": 0, "unavailable": collections.Counter(),
                   "contributing_eyes": collections.Counter(), "costs": [],
                   "modeled_eyes": collections.Counter(), "unlocalized_eye_costs": [], "hypotheses": [],
                   "association_work": collections.Counter(),
                   "elapsed_ms": [], "alternative_margins": []} for m in methods}
    heldout = [[], []]
    matched_supported = [[], []]
    arc_coverage = [collections.Counter(), collections.Counter()]
    matched_costs = []
    ids, duplicates, capture_entries = set(), [], set()
    both_raw = both_evidence = rows = 0
    independent_scale = collections.Counter()
    worst = []
    with args.evaluation.open() as source:
        for line in source:
            row = json.loads(line)
            rows += 1
            inputs = row.get("inputs", [None, None])
            both_raw += all(inputs)
            both_evidence += all(row.get("observed_arc_groups", [0, 0]))
            for item in inputs:
                if item:
                    key = item["index"]
                    if key in ids:
                        duplicates.append(key)
                    ids.add(key)
                    capture_entries.add(item["capture_entry"])
            for eye, provenance in enumerate(row.get("scale_provenance", [])):
                independent_scale[f"eye{eye}:{provenance}"] += 1
            for name in methods:
                result = row.get(name, {})
                aggregate = summary[name]
                aggregate["elapsed_ms"].append(result.get("elapsed_ms"))
                if not result.get("available"):
                    aggregate["unavailable"][result.get("reason", row.get("error", "absent-result"))] += 1
                    continue
                aggregate["available"] += 1
                check_shared_target_contract(result)
                aggregate["contributing_eyes"][str(result["contributing_eyes"])] += 1
                aggregate["modeled_eyes"][str(result.get("modeled_eyes","legacy-unspecified"))] += 1
                aggregate["unlocalized_eye_costs"].append(sum(result["unlocalized_eye_cost"]) if "unlocalized_eye_cost" in result else None)
                aggregate["hypotheses"].append(result.get("hypotheses"))
                aggregate["association_work"][str(result.get("hypotheses_by_association","legacy-unspecified"))] += 1
                aggregate["costs"].append(result["cost"])
                aggregate["alternative_margins"].append(result["alternative_cost_margin"])
            joint = row.get("joint", {})
            if joint.get("available") and all(row.get(name, {}).get("available") for name in methods[1:]):
                matched_costs.append(joint["cost"] - sum(row[m]["cost"] for m in methods[1:]))
            for eye, name in enumerate(methods[1:]):
                mono = row.get(name, {})
                if not (joint.get("available") and mono.get("available")):
                    continue
                a = joint["withheld_sample_residuals"][eye]
                b = mono["withheld_sample_residuals"][eye]
                if not a or not b or a["rms_px"] is None or b["rms_px"] is None:
                    continue
                pair = a["rms_px"], b["rms_px"]
                heldout[eye].append(pair)
                joint_groups = {g["arc"]: g for g in a.get("groups", [])}
                mono_groups = {g["arc"]: g for g in b.get("groups", [])}
                shared = []
                for key in joint_groups.keys() & mono_groups.keys():
                    ga, gb = joint_groups[key], mono_groups[key]
                    arc_coverage[eye][f"joint:{ga['used']},monocular:{gb['used']}"] += 1
                    if ga["used"] and gb["used"] and ga["points"] == gb["points"]:
                        shared.append((ga, gb))
                if shared:
                    count = sum(ga["points"] for ga, gb in shared)
                    matched_supported[eye].append(tuple(math.sqrt(sum(pair[i]["rms_px"]**2*pair[i]["points"] for pair in shared)/count) for i in (0, 1)))
                worst.append({"delta_rms_px": pair[0]-pair[1], "eye": eye,
                              "joint_rms_px": pair[0], "monocular_rms_px": pair[1],
                              "indices": [v["index"] if v else None for v in inputs]})
    for aggregate in summary.values():
        for key in ("costs", "elapsed_ms", "alternative_margins", "unlocalized_eye_costs", "hypotheses"):
            aggregate[key] = distribution(aggregate[key])
    def comparisons_for(samples):
        return [{"eye": eye, "matched_reads": len(pairs),
            "joint_rms_px": distribution([a for a, b in pairs]),
            "monocular_rms_px": distribution([b for a, b in pairs]),
            "joint_minus_monocular_rms_px": distribution([a-b for a, b in pairs]),
            "improved": sum(a < b-1e-6 for a, b in pairs),
            "regressed": sum(a > b+1e-6 for a, b in pairs),
            "regressed_over_1px": sum(a > b+1 for a, b in pairs)} for eye, pairs in enumerate(samples)]
    comparisons = comparisons_for(heldout)
    expected = None
    if args.expected_manifest:
        expected = json.loads(args.expected_manifest.read_text())["summary"]["unique_raw_frames"]
    unexpected = sorted(i for i in ids if expected is not None and (not isinstance(i, int) or not 0 <= i < expected))
    missing_count = expected-len(ids-set(unexpected)) if expected is not None else None
    report = {"schema": "buttercup-joint-conic-comparison-v1", "evaluation": str(args.evaluation),
        "scope": {"reads": rows, "both_raw_reads": both_raw, "both_evidence_reads": both_evidence,
                  "unique_input_indices": len(ids), "capture_entries": len(capture_entries),
                  "expected_input_indices": expected,
                  "complete_corpus": expected is not None and missing_count == 0 and not unexpected and not duplicates,
                  "missing_input_count": missing_count, "unexpected_input_indices": unexpected,
                  "duplicate_input_indices": duplicates},
        "methods": summary, "withheld_sample_comparisons": comparisons,
        "matched_supported_arc_comparisons": comparisons_for(matched_supported),
        "matched_arc_acceptance": arc_coverage,
        "matched_joint_minus_sum_monocular_objective": distribution(matched_costs),
        "scale_provenance": independent_scale,
        "largest_withheld_sample_regressions": sorted(worst, key=lambda r: -r["delta_rms_px"])[:30],
        "limitations": ["No independent gaze/visual-axis ground truth in this comparison.",
                        "Common fixation, pinhole optics, coarse origin/scale and axis-alignment support are defeasible assumptions.",
                        "Withheld contour samples do not make upstream segmentation/search independent of that image.",
                        "SN-FEIDA is absent when external coarse scale is unavailable; no self-radius normalization.",
                        "Comparison against human native-RAW localization labels and sequence continuity remains separately required."]}
    if args.baseline_evaluation:
        report["matched_algorithm_comparison"]=matched_algorithm_report(args.baseline_evaluation,args.evaluation,args.index_range,args.allow_extractor_changes)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scope": report["scope"], "withheld_sample_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
