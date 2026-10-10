"""Build a COCO ground-truth JSON from a YOLO split, for COCO-style evaluation
(AP_small etc.). Image paths are stored relative to the dataset root.

  python data/yolo_to_coco.py --data configs/datasets/seadronessee.yaml --split test
"""

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    img_dir = root / cfg[args.split]

    coco = {
        "info": {"root": str(root), "split": args.split},
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
    }
    aid = 1
    for iid, p in enumerate(sorted(x for x in img_dir.iterdir() if x.suffix.lower() in IMG_EXT), start=1):
        w, h = Image.open(p).size
        coco["images"].append({"id": iid, "file_name": str(p.relative_to(root)), "width": w, "height": h})
        lbl = root / "labels" / args.split / (p.stem + ".txt")
        for line in lbl.read_text().splitlines() if lbl.exists() else []:
            c, cx, cy, bw, bh = line.split()
            bw, bh = float(bw) * w, float(bh) * h
            x, y = float(cx) * w - bw / 2, float(cy) * h - bh / 2
            coco["annotations"].append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": int(c) + 1,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            aid += 1

    out = Path(args.out or f"{root}/coco_{args.split}.json")
    out.write_text(json.dumps(coco))
    print(f"{len(coco['images'])} images, {len(coco['annotations'])} boxes -> {out}")


if __name__ == "__main__":
    main()
