"""COCO-style evaluation shared by plain and SAHI inference.

Why COCO eval instead of only Ultralytics' built-in val? Because COCO eval reports AP
separately for small / medium / large objects (area < 32^2, 32^2-96^2, > 96^2 pixels).
For search and rescue at sea, AP_small is the number that matters most.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def load_gt(gt_path: str | Path) -> dict:
    return json.loads(Path(gt_path).read_text())


def predict_ultralytics(model_path, gt, root, imgsz=640, device=0, max_det=300, conf=0.001):
    """Run a YOLO model on every GT image. conf=0.001 on purpose: mAP integrates over the
    whole precision-recall curve, so you must keep low-confidence boxes for evaluation."""
    from ultralytics import YOLO

    model = YOLO(model_path) if isinstance(model_path, (str, Path)) else model_path  # or a ready model object
    preds, times = [], []
    for img in gt["images"]:
        t0 = time.perf_counter()
        r = model.predict(
            str(Path(root) / img["file_name"]), imgsz=imgsz, conf=conf, max_det=max_det, device=device, verbose=False
        )[0]
        times.append((time.perf_counter() - t0) * 1000)
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            preds.append(
                {
                    "image_id": img["id"],
                    "category_id": int(c) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                }
            )
    return preds, times


def predict_sahi(model_path, gt, root, slice_size=640, overlap=0.2, device="cuda:0", conf=0.05, imgsz=640):
    """Sliced inference: cut the image into overlapping tiles, detect on each, merge.
    Small objects get more pixels per tile, at the cost of N forward passes per image."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=model_path, confidence_threshold=conf, device=device, image_size=imgsz
    )
    preds, times = [], []
    for img in gt["images"]:
        t0 = time.perf_counter()
        res = get_sliced_prediction(
            str(Path(root) / img["file_name"]),
            model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )
        times.append((time.perf_counter() - t0) * 1000)
        for op in res.object_prediction_list:
            preds.append(
                {
                    "image_id": img["id"],
                    "category_id": int(op.category.id) + 1,
                    "bbox": [float(v) for v in op.bbox.to_xywh()],
                    "score": float(op.score.value),
                }
            )
    return preds, times


def coco_evaluate(gt_path, preds, max_dets=100) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(gt_path))
    if not preds:
        raise ValueError("No predictions -- check model, paths and class mapping.")
    dt = gt.loadRes(preds)
    E = COCOeval(gt, dt, "bbox")
    E.params.maxDets = [1, 10, max_dets]
    E.evaluate()
    E.accumulate()
    E.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    out = {n: float(E.stats[i]) for i, n in enumerate(names)}
    prec = E.eval["precision"]  # [T, R, K, A, M]
    # pycocotools' summarize() hard-codes maxDets=100 for the overall AP (stats[0]) and returns
    # -1 when 100 isn't in maxDets. Dense scenes (VisDrone: up to ~300 objects/image) need more,
    # so compute overall AP ourselves at the largest maxDets, like the other stats already are.
    p_all = prec[:, :, :, 0, -1]
    out["AP"] = float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float("nan")
    # per-class AP (IoU .5:.95, all areas, max dets)
    for k, cat_id in enumerate(E.params.catIds):
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        out[f"AP_{gt.cats[cat_id]['name']}"] = float(np.mean(p)) if p.size else float("nan")
    return out
