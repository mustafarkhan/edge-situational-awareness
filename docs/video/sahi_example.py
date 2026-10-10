"""Produce the real inputs for the video's data scenes:
* stats.json  -- object-size statistics of the val split (same maths as data/analyse_dataset.py)
* sahi.json   -- plain vs SAHI detections and timings on one hero image
* prints the chosen hero image (the val image with most labelled people, unless --hero given)

python docs/video/sahi_example.py --model weights/yolo11n.pt --outdir outputs/video_inputs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
COCO_SUPER = {0: "person", 1: "two_wheeler", 3: "two_wheeler", 2: "vehicle", 5: "vehicle", 7: "vehicle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(REPO / "configs/datasets/visdrone.yaml"))
    ap.add_argument("--hero", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--outdir", default=str(REPO / "outputs/video_inputs"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    names = cfg["names"]
    imgs = sorted((root / cfg["val"]).glob("*.jpg"))

    sizes, longest, best, best_n, n_boxes = [], [], None, -1, 0
    for p in imgs:
        w, h = Image.open(p).size
        lines = (root / "labels/val" / (p.stem + ".txt")).read_text().splitlines()
        n_people = 0
        for ln in lines:
            c, _, _, bw, bh = ln.split()
            sizes.append(np.sqrt(float(bw) * w * float(bh) * h))
            longest.append(max(w, h))
            n_people += names[int(c)] in ("pedestrian", "people")
        n_boxes += len(lines)
        if n_people > best_n:
            best, best_n = p, n_people
    sizes, longest = np.array(sizes), np.array(longest)
    eff = sizes * 640 / longest
    stats = {
        "n_images": len(imgs),
        "n_boxes": int(n_boxes),
        "median640": float(np.median(eff)),
        "lt16_640": float(np.mean(eff < 16)),
        "lt32_640": float(np.mean(eff < 32)),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))

    hero = Path(args.hero) if args.hero else best
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO

    m = YOLO(args.model)
    m.predict(str(hero), imgsz=640, verbose=False)  # warm-up
    t0 = time.perf_counter()
    r = m.predict(str(hero), imgsz=640, conf=args.conf, max_det=500, verbose=False)[0]
    ms_plain = (time.perf_counter() - t0) * 1000
    plain = [[COCO_SUPER[c], *b] for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()) if c in COCO_SUPER]
    sm = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.model, confidence_threshold=args.conf, device="cpu", image_size=640
    )
    get_sliced_prediction(str(hero), sm, slice_height=640, slice_width=640, verbose=0)  # warm-up
    t0 = time.perf_counter()
    res = get_sliced_prediction(
        str(hero), sm, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0
    )
    ms_sahi = (time.perf_counter() - t0) * 1000
    sahi = [
        [COCO_SUPER[op.category.id], *op.bbox.to_xyxy()]
        for op in res.object_prediction_list
        if op.category.id in COCO_SUPER
    ]
    (out / "sahi.json").write_text(
        json.dumps(
            {
                "hero": str(hero),
                "conf": args.conf,
                "plain": plain,
                "sahi": sahi,
                "ms_plain": ms_plain,
                "ms_sahi": ms_sahi,
            }
        )
    )
    print(
        json.dumps(stats),
        "\nhero:",
        hero,
        f"({best_n} labelled people)",
        f"\nplain: {len(plain)} dets {ms_plain:.0f} ms | sahi: {len(sahi)} dets {ms_sahi:.0f} ms",
    )


if __name__ == "__main__":
    sys.exit(main())
