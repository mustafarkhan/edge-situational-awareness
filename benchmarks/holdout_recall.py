"""How many labelled objects does a land model find in the two VisDrone sequences removed before training?

VisDrone has no public test labels here, and the validation set also picks the best training epoch, so the two
training sequences 9999951 and 9999955 (768 frames) were set aside before any training. This scores a model on
them: an object counts as found when a detection of any class overlaps it with IoU >= 0.5 (greedy, highest
confidence first), at the confidence threshold the system alerts on.

  python benchmarks/holdout_recall.py --model weights/ft_y11n_ep3.pt --images datasets/demo_holdout/images \
      --tag ft_y11n_ep3

Appends one row per sequence plus "all" to benchmarks/results/visdrone_holdout.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def match(gt: np.ndarray, det: np.ndarray, conf: np.ndarray, thr: float = 0.5) -> int:
    """Objects found. Greedy, highest confidence first, each object found once."""
    if len(gt) == 0 or len(det) == 0:
        return 0
    free = np.ones(len(gt), bool)
    hits = 0
    for i in np.argsort(-conf):
        o = iou(det[i], gt)
        o[~free] = 0
        j = int(o.argmax())
        if o[j] >= thr:
            free[j] = False
            hits += 1
    return hits


def yolo_labels(path: Path, w: int, h: int) -> np.ndarray:
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []
    if not rows:
        return np.zeros((0, 4))
    a = np.array([[float(v) for v in r[1:5]] for r in rows])
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default=str(REPO / "datasets/demo_holdout/images"))
    ap.add_argument("--labels", default=None, help="YOLO label folder (default: labels/ next to the images)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(REPO / "benchmarks/results/visdrone_holdout.csv"))
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(a.model)
    imgs = sorted(Path(a.images).glob("*.jpg"))
    if not imgs:
        sys.exit(f"no images in {a.images}")
    labels = Path(a.labels) if a.labels else Path(a.images).parent / "labels"
    per = defaultdict(lambda: [0, 0, 0, 0])  # frames, objects, found, detections
    for p in imgs:
        r = model.predict(str(p), imgsz=a.imgsz, conf=a.conf, max_det=500, verbose=False)[0]
        h, w = r.orig_shape
        gt = yolo_labels(labels / f"{p.stem}.txt", w, h)
        det, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        found = match(gt, det, conf)
        for key in (p.name.split("_")[0], "all"):
            s = per[key]
            s[0] += 1
            s[1] += len(gt)
            s[2] += found
            s[3] += len(det)
    out = Path(a.out)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["tag", "model", "imgsz", "conf", "sequence", "frames", "objects", "found", "recall",
                         "detections", "precision"])  # fmt: skip
        for key in sorted(per, key=lambda k: (k == "all", k)):
            f, n, found, d = per[key]
            row = [a.tag, Path(a.model).name, a.imgsz, a.conf, key, f, n, found, round(found / max(n, 1), 4), d,
                   round(found / max(d, 1), 4)]  # fmt: skip
            wr.writerow(row)
            print(*row, sep="\t")
    return 0


if __name__ == "__main__":
    sys.exit(main())
