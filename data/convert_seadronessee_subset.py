"""Convert the video-disjoint SeaDronesSee subset (see data/sds_select_subset.py) to YOLO format.

Input: the extracted archives, i.e. a folder with test/, val/, seen/, train/ sub-folders of JPEGs (images may
have been downscaled; YOLO labels are normalised, so that is harmless as long as the aspect ratio is kept),
plus the official COCO annotation files.

  python data/convert_seadronessee_subset.py --images /path/extracted \
      --train-json instances_train.json --val-json instances_val.json --out datasets/seadronessee_yolo

Writes configs/datasets/seadronessee.yaml with splits train / val / test / seen.
'ignored' regions are dropped (detections there count as false positives; noted in LIMITATIONS).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from _common import link_or_copy, write_dataset_yaml, write_label, yolo_row
from PIL import Image

ROLES = ("train", "val", "test", "seen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--out", default="datasets/seadronessee_yolo")
    args = ap.parse_args()
    out = Path(args.out)

    meta, anns, cats = {}, defaultdict(list), None
    for path in (args.train_json, args.val_json):
        d = json.loads(Path(path).read_text())
        cats = cats or d["categories"]
        id2fn = {}
        for im in d["images"]:
            meta[im["file_name"]] = (im["width"], im["height"])
            id2fn[im["id"]] = im["file_name"]
        for a in d["annotations"]:
            anns[id2fn[a["image_id"]]].append(a)
    keep = sorted((c for c in cats if c["name"] != "ignored"), key=lambda c: c["id"])
    cat_to_cls = {c["id"]: i for i, c in enumerate(keep)}
    names = [c["name"] for c in keep]

    for role in ROLES:
        src_dir = Path(args.images) / role
        if not src_dir.exists():
            print(f"skip {role}: {src_dir} missing")
            continue
        counts, n, aspect_bad = Counter(), 0, 0
        for img in sorted(src_dir.glob("*.jpg")):
            w0, h0 = meta[img.name]
            w1, h1 = Image.open(img).size
            if abs(w0 / h0 - w1 / h1) > 0.01:
                aspect_bad += 1
            rows = []
            for a in anns.get(img.name, []):
                if a["category_id"] not in cat_to_cls:
                    continue
                r = yolo_row(cat_to_cls[a["category_id"]], *a["bbox"], w0, h0)
                if r:
                    rows.append(r)
                    counts[names[cat_to_cls[a["category_id"]]]] += 1
            link_or_copy(img, out / "images" / role / img.name)
            write_label(out / "labels" / role / (img.stem + ".txt"), rows)
            n += 1
        print(
            f"{role:5s}: {n} images, {sum(counts.values())} boxes {dict(counts)}"
            + (f"  !! {aspect_bad} images changed aspect ratio" if aspect_bad else "")
        )

    y = write_dataset_yaml("seadronessee", out, names)
    cfg = y.read_text().replace("test: images/test\n", "test: images/test\nseen: images/seen\n")
    y.write_text(cfg)
    print("dataset yaml:", y)


if __name__ == "__main__":
    main()
