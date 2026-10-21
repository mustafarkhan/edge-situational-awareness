"""Operator load: how many alerts does the operator get per real swimmer, and does de-duplication help?

Replays the real tracking runs on SeaDronesSee-MOT clips through esa.alerts.AlertManager (same settings as the
pipeline: alert after 15 frames) with different de-duplication radii. Each raised alert is matched to a labelled
swimmer (IoU >= 0.3 between the alerting track's box and the ground-truth box in that frame), which separates:

  unique     labelled swimmers that got at least one alert (what we must not lose)
  duplicate  further alerts for a swimmer that already had one (operator load, caused by tracker ID changes)
  false      alerts matching no labelled swimmer

  python benchmarks/alert_load.py --clips outputs/sea_mot/DJI_0055 ... --runs outputs
Writes benchmarks/results/alert_load.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from esa.alerts import AlertManager  # noqa: E402

SWIMMER_MOT = {1, 2}
RADII = (0, 2, 3, 5, 8, 12)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--runs", required=True, help="folder with sea_track_<clip>_<tracker>.jsonl")
    ap.add_argument("--tracker", default="bytetrack")
    ap.add_argument("--min-frames", type=int, default=15)
    ap.add_argument("--out", default=str(REPO / "benchmarks/results/alert_load.csv"))
    a = ap.parse_args()

    rows = []
    for clip in map(Path, a.clips):
        gt: dict[int, list[tuple[int, tuple]]] = {}
        for line in (clip / "gt.txt").read_text().splitlines():
            f, tid, x, y, w, h, _, cls, _ = line.split(",")
            if int(cls) in SWIMMER_MOT:
                gt.setdefault(int(f) - 1, []).append(
                    (int(tid), (float(x), float(y), float(x) + float(w), float(y) + float(h)))
                )
        n_gt = len({t for v in gt.values() for t, _ in v})
        recs = [json.loads(x) for x in (Path(a.runs) / f"sea_track_{clip.name}_{a.tracker}.jsonl").open()]
        for r in RADII:
            am = AlertManager({"swimmer"}, a.min_frames, dedupe_radius_m=r, dedupe_s=60)
            got, false = [], 0
            for rec in recs:
                for al in am.update(rec["detections"], now=rec["frame"] / 30.0):
                    box = next(d["bbox"] for d in rec["detections"] if d["track_id"] == al["track_id"])
                    best = max(gt.get(rec["frame"], []), key=lambda g: iou(box, g[1]), default=None)
                    if best is not None and iou(box, best[1]) >= 0.3:
                        got.append(best[0])
                    else:
                        false += 1
            uniq = len(set(got))
            rows.append(
                {
                    "clip": clip.name,
                    "dedupe_radius_m": r,
                    "labelled_swimmers": n_gt,
                    "alerts": len(got) + false,
                    "unique_swimmers_alerted": uniq,
                    "duplicate_alerts": len(got) - uniq,
                    "false_alerts": false,
                    "alerts_per_swimmer_found": round((len(got) + false) / max(uniq, 1), 2),
                }
            )
    # all clips together
    for r in RADII:
        sel = [x for x in rows if x["dedupe_radius_m"] == r and x["clip"] != "all"]
        tot = {k: sum(x[k] for x in sel) for k in ("labelled_swimmers", "alerts", "unique_swimmers_alerted",
                                                     "duplicate_alerts", "false_alerts")}  # fmt: skip
        rows.append(
            {
                "clip": "all",
                "dedupe_radius_m": r,
                **tot,
                "alerts_per_swimmer_found": round(tot["alerts"] / max(tot["unique_swimmers_alerted"], 1), 2),
            }
        )
    out = Path(a.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for x in rows:
        if x["clip"] == "all":
            print(x)


if __name__ == "__main__":
    main()
