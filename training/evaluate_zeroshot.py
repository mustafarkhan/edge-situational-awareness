"""Zero-shot baseline: how well does a COCO-pretrained YOLO11 detect objects in drone
imagery *before* any fine-tuning? This measures the domain gap the project has to close.

COCO and VisDrone use different class lists, so both are mapped to three shared classes:
  person      <- VisDrone pedestrian, people        | COCO person
  two_wheeler <- VisDrone bicycle, motor            | COCO bicycle, motorcycle
  vehicle     <- VisDrone car, van, truck, bus      | COCO car, bus, truck
VisDrone tricycle / awning-tricycle have no COCO equivalent and are left out of the ground
truth (a detection on one counts as a false positive, so scores are slightly pessimistic).

  python training/evaluate_zeroshot.py --model yolo11n.pt --data configs/datasets/visdrone.yaml \
      --imgsz 640 --tag y11n_640_zeroshot [--sahi --slice 640]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from esa.coco_utils import coco_evaluate, predict_sahi, predict_ultralytics  # noqa: E402

SUPER = ["person", "two_wheeler", "vehicle"]
VISDRONE_TO_SUPER = {"pedestrian": 0, "people": 0, "bicycle": 1, "motor": 1, "car": 2, "van": 2, "truck": 2, "bus": 2}
COCO_TO_SUPER = {0: 0, 1: 1, 3: 1, 2: 2, 5: 2, 7: 2}  # COCO ids: person, bicycle, motorcycle, car, bus, truck


def build_gt(data_yaml: str, split: str) -> Path:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    root, names = Path(cfg["path"]), cfg["names"]
    out = root / f"coco_{split}_super.json"
    if out.exists():
        return out
    coco = {
        "info": {"root": str(root), "split": split},
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(SUPER)],
    }
    aid = 1
    imgs = sorted(p for p in (root / cfg[split]).iterdir() if p.suffix.lower() in {".jpg", ".png"})
    for iid, p in enumerate(imgs, start=1):
        w, h = Image.open(p).size
        coco["images"].append({"id": iid, "file_name": str(p.relative_to(root)), "width": w, "height": h})
        lbl = root / "labels" / split / (p.stem + ".txt")
        for line in lbl.read_text().splitlines() if lbl.exists() else []:
            c, cx, cy, bw, bh = line.split()
            sc = VISDRONE_TO_SUPER.get(names[int(c)])
            if sc is None:
                continue
            bw, bh = float(bw) * w, float(bh) * h
            coco["annotations"].append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": sc + 1,
                    "bbox": [float(cx) * w - bw / 2, float(cy) * h - bh / 2, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            aid += 1
    out.write_text(json.dumps(coco))
    return out


VISDRONE_IDX_TO_SUPER = {0: 0, 1: 0, 2: 1, 9: 1, 3: 2, 4: 2, 5: 2, 8: 2}  # for models fine-tuned on VisDrone


def remap(preds, class_map="coco"):
    table = COCO_TO_SUPER if class_map == "coco" else VISDRONE_IDX_TO_SUPER
    out = []
    for p in preds:
        sc = table.get(p["category_id"] - 1)
        if sc is not None:
            out.append({**p, "category_id": sc + 1})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--sahi", action="store_true")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--max-dets", type=int, default=500)
    ap.add_argument(
        "--class-map",
        choices=["coco", "visdrone"],
        default="coco",
        help="'coco' for pretrained weights, 'visdrone' for weights fine-tuned on VisDrone",
    )
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all)")
    ap.add_argument("--device", default="cpu", help="'cpu', or 0 for the first CUDA GPU (e.g. on the Jetson)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "zeroshot.csv"))
    args = ap.parse_args()

    gt_path = build_gt(args.data, args.split)
    gt = json.loads(Path(gt_path).read_text())
    if args.limit:
        keep = {im["id"] for im in gt["images"][: args.limit]}
        gt["images"] = [im for im in gt["images"] if im["id"] in keep]
        gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
        gt_path = Path(gt_path).with_name(f"coco_{args.split}_super_first{args.limit}.json")
        gt_path.write_text(json.dumps(gt))
    root = gt["info"]["root"]

    if args.sahi:
        preds, times = predict_sahi(
            args.model,
            gt,
            root,
            slice_size=args.slice,
            overlap=args.overlap,
            device="cpu" if args.device == "cpu" else f"cuda:{args.device}",
            conf=args.conf,
            imgsz=args.imgsz,
        )
    else:
        preds, times = predict_ultralytics(
            args.model,
            gt,
            root,
            imgsz=args.imgsz,
            device="cpu" if args.device == "cpu" else int(args.device),
            max_det=args.max_dets,
            conf=args.conf,
        )
    mapped = remap(preds, args.class_map)
    if mapped:
        m = coco_evaluate(gt_path, mapped, max_dets=args.max_dets)
    else:
        # a model that outputs nothing scores zero; record it rather than crash (see INT8 experiment)
        print("WARNING: model produced no detections at all -> all AP = 0")
        m = {
            k: 0.0
            for k in [
                "AP",
                "AP50",
                "AP75",
                "AP_small",
                "AP_medium",
                "AP_large",
                "AP_person",
                "AP_two_wheeler",
                "AP_vehicle",
            ]
        }

    row = {
        "tag": args.tag,
        "split": args.split,
        "n_images": len(gt["images"]),
        "model": Path(args.model).name,
        "imgsz": args.imgsz,
        "sahi": args.sahi,
        "slice": args.slice if args.sahi else "",
        "conf": args.conf,
        "cpu_ms_per_image_median": round(float(np.median(times)), 1),
        **{k: round(v, 4) for k, v in m.items()},
    }
    out = Path(args.out)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("\n" + "\n".join(f"{k:>26}: {v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
