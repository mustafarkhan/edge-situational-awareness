"""COCO-style evaluation on the SeaDronesSee subset.

Two views:
  --view native     all 5 SeaDronesSee classes (models fine-tuned on SeaDronesSee only)
  --view two_class  swimmer + boat only, so a COCO-pretrained model (person -> swimmer, boat -> boat) and a
                    fine-tuned model can be compared on the same ground truth

  python training/evaluate_sea.py --model weights/sds_ft.pt --data configs/datasets/seadronessee.yaml \
      --split test --imgsz 1280 --view native --tag sds_ft_1280
Text prompts (open vocabulary, no sea training at all):
  python training/evaluate_sea.py --model weights/yoloe-11s-seg.pt --view two_class \
      --prompts "swimmer=swimmer|person in water;boat=boat" --tag yoloe11s_p2
  Each target class can have several phrases (|); a detection of any phrase counts as that class. Rows go to
  benchmarks/results/seadronessee_prompts.csv with the prompt text recorded.

Rows are appended to benchmarks/results/seadronessee.csv. Splits: val / test (videos never trained on) /
seen (official-val frames from training videos: measures how much frame-level leakage inflates scores).
"""

from __future__ import annotations

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

COCO_TO_TWO = {0: "swimmer", 8: "boat"}  # COCO class ids: person, boat


def parse_prompts(spec: str) -> tuple[list[str], list[str]]:
    """'swimmer=swimmer|person in water;boat=boat' -> phrases, and the target class of each phrase."""
    phrases, targets = [], []
    for part in spec.split(";"):
        target, _, alts = part.partition("=")
        for phrase in alts.split("|"):
            if phrase.strip():
                phrases.append(phrase.strip())
                targets.append(target.strip())
    if not phrases:
        raise ValueError(f"no prompts in {spec!r}")
    return phrases, targets


def load_prompted(model_path: str, phrases: list[str]):
    from ultralytics import YOLOE

    model = YOLOE(model_path)
    model.set_classes(phrases, model.get_text_pe(phrases))
    return model


def build_gt(data_yaml: str, split: str, keep: list[str] | None, limit: int) -> tuple[Path, dict, list[str]]:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    root, names = Path(cfg["path"]), cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    out_names = keep or names
    coco = {
        "info": {"root": str(root), "split": split},
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(out_names)],
    }
    aid = 1
    imgs = sorted((root / cfg[split]).glob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    for iid, p in enumerate(imgs, start=1):
        w, h = Image.open(p).size
        coco["images"].append({"id": iid, "file_name": str(p.relative_to(root)), "width": w, "height": h})
        lbl = root / "labels" / split / (p.stem + ".txt")
        for line in lbl.read_text().splitlines() if lbl.exists() else []:
            c, cx, cy, bw, bh = line.split()
            name = names[int(c)]
            if name not in out_names:
                continue
            bw, bh = float(bw) * w, float(bh) * h
            coco["annotations"].append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": out_names.index(name) + 1,
                    "bbox": [float(cx) * w - bw / 2, float(cy) * h - bh / 2, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            aid += 1
    tag = "two" if keep else "native"
    gt_path = root / f"coco_{split}_{tag}{'_' + str(limit) if limit else ''}.json"
    gt_path.write_text(json.dumps(coco))
    return gt_path, coco, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(REPO / "configs/datasets/seadronessee.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--view", choices=["native", "two_class"], default="native")
    ap.add_argument("--coco-model", action="store_true", help="model predicts COCO classes (pretrained weights)")
    ap.add_argument("--prompts", default=None, help='open-vocabulary model: "class=phrase|phrase;class=phrase"')
    ap.add_argument("--device", default="cpu", help="cpu, or 0 for the GPU / a TensorRT engine on the Jetson")
    ap.add_argument("--sahi", action="store_true")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--max-dets", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(REPO / "benchmarks/results/seadronessee.csv"))
    args = ap.parse_args()
    if args.coco_model and args.view != "two_class":
        sys.exit("a COCO-class model can only be scored with --view two_class")

    keep = ["swimmer", "boat"] if args.view == "two_class" else None
    gt_path, gt, ds_names = build_gt(args.data, args.split, keep, args.limit)
    root = gt["info"]["root"]
    phrases, targets = parse_prompts(args.prompts) if args.prompts else (None, None)
    if phrases:
        preds, times = predict_ultralytics(
            load_prompted(args.model, phrases),
            gt,
            root,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_dets,
            conf=args.conf,
        )
    elif args.sahi:
        preds, times = predict_sahi(
            args.model, gt, root, slice_size=args.slice, device="cpu", conf=args.conf, imgsz=args.imgsz
        )
    else:
        preds, times = predict_ultralytics(
            args.model, gt, root, imgsz=args.imgsz, device=args.device, max_det=args.max_dets, conf=args.conf
        )
    out_names = keep or ds_names
    mapped = []
    for p in preds:
        c = p["category_id"] - 1  # model class index
        name = targets[c] if phrases else COCO_TO_TWO.get(c) if args.coco_model else ds_names[c]
        if name in out_names:
            mapped.append({**p, "category_id": out_names.index(name) + 1})
    m = coco_evaluate(gt_path, mapped, max_dets=args.max_dets) if mapped else {}

    row = {
        "tag": args.tag,
        "split": args.split,
        "view": args.view,
        "n_images": len(gt["images"]),
        "n_objects": len(gt["annotations"]),
        "model": Path(args.model).name,
        "imgsz": args.imgsz,
        "sahi": args.sahi,
        **({"prompts": args.prompts} if phrases else {}),
        "cpu_ms_per_image_median": round(float(np.median(times)), 1),
    }
    for k in ["AP50", "AP", "AP_small", "AP_medium", "AP_large"] + [f"AP_{n}" for n in out_names]:
        v = m.get(k, 0.0)
        row[k] = round(v, 4) if v == v else ""
    # per-class columns differ by view, so each view gets its own file
    out = Path(args.out)
    stem = out.stem + ("_prompts" if phrases else "") + ("_two_class" if args.view == "two_class" else "")
    out = out.with_name(stem + out.suffix)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("\n".join(f"{k:>26}: {v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
