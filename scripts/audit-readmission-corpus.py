#!/usr/bin/env python3
"""Inventory archived ROI clocks/residency without extracting RAW or trusting predictions as labels."""
import argparse
import collections
import json
import pathlib
import tarfile


def audit(path):
    with tarfile.open(path) as archive:
        stream = archive.extractfile("frames.jsonl")
        rows = [json.loads(line) for line in stream if line.strip()]
        masks = collections.Counter()
        exposures = collections.defaultdict(set)
        transitions = []
        prior = None
        for row in rows:
            region = row.get("region") or {}
            mask = region.get("active_mask", 3)
            masks[mask] += 1
            key = (region.get("session"), row["sequence"], row["timestamp_ns"])
            exposures[key].add(row["eye_id"])
            if prior is not None and prior != mask:
                transitions.append([row["sequence"], prior, mask])
            prior = mask
        raw_valid = all(
            row["offset"] + row["length"] <= archive.getmember(row["stream"]).size
            for row in rows
        )
        return dict(path=str(path), frames=len(rows), raw_offsets_valid=raw_valid,
                    exact_clock_pairs=sum(eyes == {1, 2} for eyes in exposures.values()),
                    eyes=sorted({r["eye_id"] for r in rows}), masks=dict(masks),
                    residency_transitions=transitions,
                    recovery_trace="recovery.jsonl" in archive.getnames())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--correlate", type=pathlib.Path, help="Run withheld-eye translation test on this archive")
    args = parser.parse_args()
    if args.correlate:
        print(json.dumps(correlate(args.correlate), indent=2))
        return
    paths = sorted({p for root in args.roots for p in root.rglob("*.tar")},
                   key=lambda p: p.stat().st_mtime, reverse=True)[:args.limit]
    results = []
    for path in paths:
        try:
            results.append(audit(path))
        except (OSError, KeyError, ValueError, tarfile.TarError) as error:
            results.append(dict(path=str(path), error=str(error)))
    print(json.dumps(dict(schema="readmission-corpus-inventory-v1", captures=results,
                         limitation="Availability inventory only; paired pixels are not labeled recovery truth."), indent=2))


def correlate(path):
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    with tarfile.open(path) as archive:
        rows = [json.loads(line) for line in archive.extractfile("frames.jsonl")]
        manifest = json.load(archive.extractfile("manifest.json"))
        groups = collections.defaultdict(dict)
        for r in rows:
            groups[((r.get("region") or {}).get("session"),r["sequence"],r["timestamp_ns"])][r["eye_id"]]=r
        pairs = [g for g in groups.values() if set(g)=={1,2}]
        streams = {eye:archive.extractfile(pairs[0][eye]["stream"]) for eye in [1,2]} if pairs else {}
        def raw(row):
            stream=streams[row["eye_id"]];stream.seek(row["offset"])
            data=np.frombuffer(stream.read(row["length"]),dtype=np.uint8).reshape(row["height"],row["stride"])
            packed=data[:,:row["width"]//4*5].reshape(row["height"],-1,5).astype(np.uint64)
            words=sum(packed[:,:,i] << (8*i) for i in range(5))
            pixels=np.stack([(words >> (10*i)) & 1023 for i in range(4)],axis=2).reshape(row["height"],row["width"]).astype(np.float32)
            # Native 4x4 Quad-Bayer cell mean: fixed phase-free luminance proxy,
            # no auto contrast or candidate-dependent geometric normalization.
            h,w=pixels.shape
            return pixels[:h//4*4,:w//4*4].reshape(h//4,4,w//4,4).mean(axis=(1,3))
        comparisons=[]; rejected=0
        # Bounded, disjoint before/after pairs; each eye supplies an independent
        # fixed template. The missing eye's match is scoring-only, not input.
        for i in range(0,min(len(pairs)-5,1000),10):
            before,after=pairs[i],pairs[i+5]
            if any(after[e]["timestamp_ns"]-before[e]["timestamp_ns"] not in range(1,1_000_000_001) for e in [1,2]):
                rejected+=1;continue
            estimates=[]
            for eye in [1,2]:
                a,b=raw(before[eye]),raw(after[eye]);h,w=a.shape
                template=a[h//4:3*h//4,w//4:3*w//4]
                if b.shape != a.shape or template.std()<2: break
                score=cv2.matchTemplate(b,template,cv2.TM_CCOEFF_NORMED)
                _,peak,_,pos=cv2.minMaxLoc(score)
                other=score.copy();x,y=pos
                other[max(0,y-2):y+3,max(0,x-2):x+3]=-1
                distinct=peak-float(other.max())
                if peak<0.65 or distinct<0.015: break
                origin=[before[eye]["sensor_x"]+w//4*4,before[eye]["sensor_y"]+h//4*4]
                moved=[after[eye]["sensor_x"]+x*4,after[eye]["sensor_y"]+y*4]
                estimates.append((np.array(moved)-origin,float(peak),distinct))
            if len(estimates)!=2: rejected+=1;continue
            for missing in [0,1]:
                truth=estimates[missing][0]; prediction=estimates[1-missing][0]
                comparisons.append(dict(sequence=after[1]["sequence"],missing_eye=missing+1,
                    source_timestamp_ns=after[1]["timestamp_ns"],
                    seed_eyes=[[before[e]["sensor_x"],before[e]["sensor_y"]] for e in [1,2]],
                    eye_size=[before[1]["width"],before[1]["height"]],
                    sensor_band=(after[1].get("region") or {}).get("sensor_window",manifest.get("sensor_window")),
                    sensor_band_provenance="source-region" if after[1].get("region") else "initial-manifest-assumption",
                    observed_translations_px=[e[0].tolist() for e in estimates],
                    held_crop_error_px=float(np.linalg.norm(truth)),
                    translated_pair_error_px=float(np.linalg.norm(truth-prediction)),
                    correlation_scores=[e[1] for e in estimates]))
        def stats(key):
            values=[c[key] for c in comparisons]
            return dict(median=float(np.median(values)),p90=float(np.percentile(values,90))) if values else None
        return dict(path=str(path),exact_clock_pairs=len(pairs),accepted_withheld_tests=len(comparisons),
                    rejected_pair_windows=rejected,baseline=stats("held_crop_error_px"),candidate=stats("translated_pair_error_px"),
                    comparisons=comparisons,limitations="Translation proxy, not live SAM pivot replay or human labels. Ambiguous correlations excluded; no rotation model, scale labels, or SN-FEIDA claim.")


if __name__ == "__main__":
    main()
