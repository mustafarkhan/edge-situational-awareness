"""Score a tracking run against a simulated flight (tools/sim_flight.py).

Tracking: standard CLEAR-MOT and identity metrics via `motmetrics`, class-agnostic,
IoU >= 0.5 for a match.
  MOTA        1 - (misses + false positives + ID switches) / ground-truth boxes
  IDF1        how well each true object keeps ONE track ID over time (identity accuracy)
  ID switches times a true object's matched track ID changed
Geolocation: for every matched box, distance in metres between the position the pipeline
reported and the object's true position. Static objects make this a clean test of the
pixel -> lat/lon chain under camera motion and altitude change.

  python benchmarks/tracking_eval.py --flight outputs/flight_slow --run outputs/track_slow_bytetrack.jsonl \
      --tag slow_bytetrack
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import motmetrics as mm
import numpy as np

REPO = Path(__file__).resolve().parents[1]
EARTH = 111_320.0


def metres(lat1, lon1, lat2, lon2):
    dn = (lat2 - lat1) * EARTH
    de = (lon2 - lon1) * EARTH * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dn, de)


def iou_distance(gt_boxes, hyp_boxes, min_iou=0.5):
    """1 - IoU for [x, y, w, h] boxes, NaN where IoU < min_iou (= not allowed to match).
    Written out because motmetrics 1.4's iou_matrix uses np.asfarray, removed in NumPy 2."""
    if not gt_boxes or not hyp_boxes:
        return np.empty((len(gt_boxes), len(hyp_boxes)))
    a, b = np.asarray(gt_boxes, float), np.asarray(hyp_boxes, float)
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    iou = inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter)
    d = 1.0 - iou
    d[iou < min_iou] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", required=True)
    ap.add_argument("--run", required=True, help="JSONL from tools/detect_video.py")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tracker", default="")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "tracking.csv"))
    args = ap.parse_args()

    flight = Path(args.flight)
    meta = json.loads((flight / "meta.json").read_text())
    objects = {int(k): v for k, v in json.loads((flight / "objects.json").read_text()).items()}
    gt = defaultdict(list)
    for line in (flight / "gt.txt").read_text().splitlines():
        f, oid, x, y, w, h, *_ = line.split(",")
        gt[int(f) - 1].append((int(oid), [float(x), float(y), float(w), float(h)]))
    recs = [json.loads(l) for l in Path(args.run).read_text().splitlines() if l.strip()]

    acc = mm.MOTAccumulator(auto_id=False)
    pos = {}  # (frame, track_id) -> (lat, lon)
    for rec in recs:
        f = rec["frame"]
        g = gt.get(f, [])
        hyps = [d for d in rec["detections"] if d["track_id"] is not None]
        hb = [[d["bbox"][0], d["bbox"][1], d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]] for d in hyps]
        dist = iou_distance([b for _, b in g], hb)
        acc.update([o for o, _ in g], [d["track_id"] for d in hyps], dist, frameid=f)
        for d in hyps:
            pos[(f, d["track_id"])] = (d["lat"], d["lon"])

    mh = mm.metrics.create()
    s = mh.compute(
        acc,
        metrics=[
            "mota",
            "idf1",
            "num_switches",
            "num_fragmentations",
            "recall",
            "precision",
            "mostly_tracked",
            "num_unique_objects",
            "num_objects",
        ],
        name="x",
    ).iloc[0]

    ev = acc.mot_events
    matches = ev[ev["Type"].isin(["MATCH", "SWITCH"])]
    errs, per_track = [], defaultdict(list)
    for (f, _), row in matches.iterrows():
        p = pos.get((f, row["HId"]))
        o = objects.get(int(row["OId"]))
        if p and o:
            errs.append(metres(o["lat"], o["lon"], *p))
            per_track[row["HId"]].append(p)
    jitter = [
        float(np.mean([metres(np.mean([a for a, _ in v]), np.mean([b for _, b in v]), a, b) for a, b in v]))
        for v in per_track.values()
        if len(v) >= 10
    ]
    fps = [r["fps"] for r in recs[3:]] or [0]

    row = {
        "tag": args.tag,
        "tracker": args.tracker,
        "frames": len(recs),
        "speed_m_per_frame": meta["speed_m_per_frame"],
        "gt_boxes": int(s["num_objects"]),
        "gt_objects": int(s["num_unique_objects"]),
        "MOTA": round(float(s["mota"]), 3),
        "IDF1": round(float(s["idf1"]), 3),
        "id_switches": int(s["num_switches"]),
        "fragmentations": int(s["num_fragmentations"]),
        "recall": round(float(s["recall"]), 3),
        "precision": round(float(s["precision"]), 3),
        "mostly_tracked": int(s["mostly_tracked"]),
        "geo_err_median_m": round(float(np.median(errs)), 2) if errs else "",
        "geo_err_p95_m": round(float(np.percentile(errs, 95)), 2) if errs else "",
        "track_jitter_median_m": round(float(np.median(jitter)), 2) if jitter else "",
        "cpu_fps_median": round(float(np.median(fps)), 1),
    }
    out = Path(args.out)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("\n".join(f"{k:>24}: {v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
