"""Convert SeaDronesSee Object Detection v2 (COCO JSON) to YOLO format with a leak-free split.

Splits:
  * official 'val'  -> our TEST  (official test labels are withheld for the challenge server)
  * official 'train' -> our TRAIN + VAL, split in BLOCKS of consecutive image ids.
    Frames come from video; neighbouring frames are near-duplicates. A random per-image
    split would put almost the same frame in train and val, and your val score would lie.

Usage:
  python data/convert_seadronessee.py \
      --train-json /path/annotations/instances_train.json --train-images /path/images/train \
      --val-json   /path/annotations/instances_val.json   --val-images   /path/images/val \
      --out datasets/seadronessee_yolo
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _common import link_or_copy, write_dataset_yaml, write_label, yolo_row

IGNORE_NAMES = {"ignored"}


def load(json_path):
    d = json.loads(Path(json_path).read_text())
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    return d, anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--out", default="datasets/seadronessee_yolo")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--block-size", type=int, default=200, help="consecutive image ids per block")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)

    tr, tr_anns = load(args.train_json)
    va, va_anns = load(args.val_json)

    # class list from the file itself (never hard-code what you can read)
    cats = sorted((c for c in tr["categories"] if c["name"] not in IGNORE_NAMES), key=lambda c: c["id"])
    cat_to_cls = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    print("classes:", names)

    # block split of the official train set
    imgs = sorted(tr["images"], key=lambda im: im["id"])
    blocks = [imgs[i:i + args.block_size] for i in range(0, len(imgs), args.block_size)]
    random.Random(args.seed).shuffle(blocks)
    n_val = max(1, round(len(blocks) * args.val_fraction))
    split_imgs = {"val": [im for b in blocks[:n_val] for im in b],
                  "train": [im for b in blocks[n_val:] for im in b],
                  "test": va["images"]}
    sources = {"train": (Path(args.train_images), tr_anns), "val": (Path(args.train_images), tr_anns),
               "test": (Path(args.val_images), va_anns)}

    for split, images in split_imgs.items():
        img_dir, anns = sources[split]
        counts, dropped = Counter(), 0
        for im in images:
            src = img_dir / im["file_name"]
            if not src.exists():
                dropped += 1
                continue
            link_or_copy(src, out / "images" / split / src.name)
            rows = []
            for a in anns.get(im["id"], []):
                if a["category_id"] not in cat_to_cls:
                    continue
                r = yolo_row(cat_to_cls[a["category_id"]], *a["bbox"], im["width"], im["height"])
                if r:
                    rows.append(r)
                    counts[names[cat_to_cls[a["category_id"]]]] += 1
            write_label(out / "labels" / split / (src.stem + ".txt"), rows)
        print(f"{split:5s}: {len(images) - dropped} images, {sum(counts.values())} boxes, {dict(counts)}"
              + (f"  (missing image files: {dropped})" if dropped else ""))

    print("dataset yaml:", write_dataset_yaml("seadronessee", out, names))


if __name__ == "__main__":
    main()
