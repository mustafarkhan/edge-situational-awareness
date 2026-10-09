"""Convert VisDrone2019-DET to YOLO format.

Expected layout (as downloaded):
  <src>/VisDrone2019-DET-train/{images,annotations}
  <src>/VisDrone2019-DET-val/{images,annotations}
  <src>/VisDrone2019-DET-test-dev/{images,annotations}

Splits: official train -> train, official val -> val, official test-dev -> test.
VisDrone annotation line: x,y,w,h,score,category,truncation,occlusion
  score == 0 means 'ignored region'; category 0 = ignored, 11 = others -> both skipped.
"""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from _common import link_or_copy, write_dataset_yaml, write_label, yolo_row

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="datasets/visdrone_yolo")
    args = ap.parse_args()
    out = Path(args.out)

    for split, folder in SPLITS.items():
        d = Path(args.src) / folder
        if not d.exists():
            print(f"skip {split}: {d} not found")
            continue
        counts, n = Counter(), 0
        for img_path in sorted((d / "images").glob("*.jpg")):
            iw, ih = Image.open(img_path).size
            rows = []
            ann = d / "annotations" / (img_path.stem + ".txt")
            for line in ann.read_text().strip().splitlines() if ann.exists() else []:
                v = [int(x) for x in line.strip().strip(",").split(",")[:6]]
                x, y, w, h, score, cat = v
                if score == 0 or cat in (0, 11):
                    continue
                r = yolo_row(cat - 1, x, y, w, h, iw, ih)
                if r:
                    rows.append(r)
                    counts[NAMES[cat - 1]] += 1
            link_or_copy(img_path, out / "images" / split / img_path.name)
            write_label(out / "labels" / split / (img_path.stem + ".txt"), rows)
            n += 1
        print(f"{split:5s}: {n} images, {sum(counts.values())} boxes")

    print("dataset yaml:", write_dataset_yaml("visdrone", out, NAMES))


if __name__ == "__main__":
    main()
