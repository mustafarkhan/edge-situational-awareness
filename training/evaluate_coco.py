"""COCO-style evaluation with size-bucketed AP, with or without SAHI sliced inference.

  # plain
  python training/evaluate_coco.py --model runs/sds/y11s_1280/weights/best.pt \
      --gt datasets/seadronessee_yolo/coco_val.json --imgsz 1280 --tag y11s_1280
  # sliced
  python training/evaluate_coco.py ... --sahi --slice 640 --tag y11s_1280_sahi

Results are appended to benchmarks/results/accuracy.csv.
RULE: tune on 'val'. Touch 'test' only for the final numbers you publish.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esa.coco_utils import coco_evaluate, load_gt, predict_sahi, predict_ultralytics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--gt", required=True, help="COCO json from data/yolo_to_coco.py")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sahi", action="store_true")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--max-dets", type=int, default=100, help="VisDrone scenes are dense: try 500")
    ap.add_argument("--sanity", action="store_true",
                    help="score the ground truth against itself: must give AP=1.0, else your pipeline is broken")
    ap.add_argument("--tag", required=True, help="short name for this row in the results table")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "accuracy.csv"))
    args = ap.parse_args()

    gt = load_gt(args.gt)
    root = gt["info"]["root"]
    split = gt["info"].get("split", "?")
    if split == "test":
        print("!! Evaluating on TEST. Only do this for final, already-chosen models. !!")

    if args.sanity:
        preds = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
                 for a in gt["annotations"]]
        print({k: round(v, 3) for k, v in coco_evaluate(args.gt, preds, max_dets=args.max_dets).items()})
        return
    if args.sahi:
        preds, times = predict_sahi(args.model, gt, root, slice_size=args.slice,
                                    overlap=args.overlap, imgsz=args.imgsz)
    else:
        preds, times = predict_ultralytics(args.model, gt, root, imgsz=args.imgsz, max_det=args.max_dets)

    m = coco_evaluate(args.gt, preds, max_dets=args.max_dets)
    row = {"tag": args.tag, "split": split, "model": args.model, "imgsz": args.imgsz,
           "sahi": args.sahi, "ms_per_image_median": round(float(np.median(times)), 1),
           **{k: round(v, 4) for k, v in m.items()}}
    out = Path(args.out)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("\n" + "\n".join(f"{k:>22}: {v}" for k, v in row.items()))
    print("NB: ms_per_image here is on the training machine, not the Jetson. "
          "Jetson speed comes from benchmarks/benchmark.py.")


if __name__ == "__main__":
    main()
