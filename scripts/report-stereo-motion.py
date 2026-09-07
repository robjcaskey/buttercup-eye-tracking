#!/usr/bin/env python3
"""Post-fit SN-FEIDA changes using byte-verified, independent RAW motion links.

Each adjacent RAW-motion link defines its own previous-image scale reference.
No candidate radius calibrates that scale. This is neither absolute mm² nor a
claim of stable anatomical surface area, calibrated confidence, or true gaze.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics


def identity(frame,digest,eye,origin):
    return (digest,int(eye),int(frame["timestamp_ns"]),int(frame["sequence"]),
        int(frame["width"]),int(frame["height"]),*map(int,origin))


def scale_link(previous,current):
    motion=current.get("shared_global_scale",{})
    d,r=motion.get("scale_delta",math.nan),motion.get("rotation",math.nan)
    residual,support=motion.get("motion_residual",math.nan),motion.get("motion_support",0)
    if not (motion.get("reliable") is True and support>=9 and math.isfinite(residual) and residual<=2.0
        and motion.get("stable_frames",0)>=2 and motion.get("occupied_quadrants",0)>=3
        and math.isfinite(d) and abs(d)<=0.04 and math.isfinite(r) and abs(r)<=0.10
        and current["width"]==previous["width"] and current["height"]==previous["height"]
        and 0<int(current["timestamp_ns"])-int(previous["timestamp_ns"])<=500_000_000
        and int(current["sequence"])!=int(previous["sequence"])):
        return None
    # Same bounded heuristic as the earlier native flat-tire area experiment.
    # [[1+d,-r],[r,1+d]] has positive determinant (1+d)²+r².
    scale=math.hypot(1+d,r)
    allowance=min(0.045,max(0.012,0.012+max(residual,0)/180+0.025/math.sqrt(support)))
    return scale,allowance


def normalized_log_change(previous_radius,current_radius,scale):
    return 2*(math.log(current_radius/previous_radius)-math.log(scale))


def evaluations(path):
    index=collections.defaultdict(list)
    with path.open() as source:
        for line in source:
            row=json.loads(line)
            fit=row.get("joint",{})
            for eye,receipt in enumerate(row["inputs"]):
                if receipt is None:
                    continue
                frame=receipt["frame"]
                key=identity(frame,receipt["raw_sha256"],frame["eye_id"],[frame["sensor_x"],frame["sensor_y"]])
                observed=fit.get("available",False) and fit["contributing_eyes"][eye] and any(
                    a["used"] and a["roi"]==eye+1 and a["kind"]=="OuterLimbus" for a in fit["support"])
                radius=fit["outer_ellipses"][eye]["major_radius"] if observed else None
                index[key].append({"index":receipt["index"],"clock":receipt["clock_lineage"],"radius":radius})
    return index


def distribution(values):
    values=sorted(values)
    return {"count":0} if not values else {"count":len(values),"median":statistics.median(values),
        "mean":statistics.mean(values),"p95":values[round((len(values)-1)*0.95)],"maximum":values[-1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("candidate",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("scale_reports",type=Path,nargs="+")
    args=parser.parse_args()
    baseline,candidate=evaluations(args.baseline),evaluations(args.candidate)
    counts=collections.Counter()
    rows=[]
    seen=set()
    for path in args.scale_reports:
        report=json.loads(path.read_text())
        source=report["source"]
        if source.get("lossless_raw_source_by_offset") is not True or source.get("pixel_format")!="RAW10_LE40_1X1":
            raise ValueError(f"scale report does not attest lossless source RAW: {path}")
        eye={"subject-right":1,"subject-left":2}[source["label"]]
        digests={}
        with Path(source["stream"]).open("rb") as raw:
            def key_for(frame):
                offset,length=int(frame["source_offset"]),int(frame["source_length"])
                key=(offset,length)
                if key not in digests:
                    raw.seek(offset)
                    packed=raw.read(length)
                    if len(packed)!=length:
                        raise ValueError(f"missing native RAW bytes for {path}: {key}")
                    digests[key]=hashlib.sha256(packed).hexdigest()
                return identity(frame,digests[key],eye,frame["sensor_origin"])
            for previous,current in zip(report["frames"],report["frames"][1:]):
                counts["motion_links_considered"]+=1
                link=scale_link(previous,current)
                if link is None:
                    counts["unsupported_motion_links"]+=1
                    continue
                counts["supported_motion_links"]+=1
                keys=key_for(previous),key_for(current)
                if keys in seen:
                    counts["duplicate_links"]+=1
                    continue
                seen.add(keys)
                matches=[[index.get(key,[]) for key in keys] for index in (baseline,candidate)]
                if any(len(match)!=1 for arm in matches for match in arm):
                    counts["missing_or_ambiguous_exact_RAW_join"]+=1
                    continue
                matches=[[match[0] for match in arm] for arm in matches]
                if len({m["clock"] for arm in matches for m in arm})!=1:
                    counts["incompatible_source_clocks"]+=1
                    continue
                if any(m["radius"] is None for arm in matches for m in arm):
                    counts["missing_fresh_accepted_outer_boundary"]+=1
                    continue
                scale,allowance=link
                changes=[normalized_log_change(arm[0]["radius"],arm[1]["radius"],scale) for arm in matches]
                counts["matched_SN_FEIDA_links"]+=1
                reframe=previous["sensor_origin"]!=current["sensor_origin"]
                counts["matched_ROI_reframes"]+=int(reframe)
                rows.append({"scale_report":str(path),"indices":[m["index"] for m in matches[0]],
                    "timestamps_ns":[str(previous["timestamp_ns"]),str(current["timestamp_ns"])],
                    "scale_reference":"previous exposure's native pixel scale, one distinct reference per link",
                    "independent_linear_scale_ratio":scale,"fractional_scale_allowance":allowance,
                    "roi_reframe":reframe,"baseline_log_SN_FEIDA_change":changes[0],"candidate_log_SN_FEIDA_change":changes[1],
                    "scale_only_candidate_log_change_bounds":[changes[1]-2*math.log1p(allowance),changes[1]-2*math.log1p(-allowance)]})
    result={"schema":"buttercup-independent-stereo-motion-diagnostic-v1","baseline":str(args.baseline),
        "candidate":str(args.candidate),"counts":counts,"links":rows,
        "absolute_log_SN_FEIDA_changes":{name:distribution([abs(row[f"{name}_log_SN_FEIDA_change"]) for row in rows]) for name in ("baseline","candidate")},
        "limitations":["Independent means independent of candidate fitting, not statistically independent of all iris pixels.",
            "Local reference-pixel² area ratios, not absolute anatomical mm²; no normalization by the candidate's radius.",
            "Scale bounds are a heuristic allowance only and omit conic-fit uncertainty; not calibrated confidence intervals.",
            "Only heuristically supported RAW motion links with exact bytes, geometry, sequence, timestamp and source-clock joins count.",
            "An area-stable wrong ellipse still fails; pair this diagnostic with human labels, coverage and gaze/source alignment."]}
    with args.output.open("x") as destination:
        json.dump(result,destination,indent=2)
        destination.write("\n")
    print(json.dumps({"counts":counts,"absolute_log_SN_FEIDA_changes":result["absolute_log_SN_FEIDA_changes"]},indent=2))


if __name__=="__main__":
    main()
