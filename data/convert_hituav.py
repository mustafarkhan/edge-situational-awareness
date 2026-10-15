"""HIT-UAV thermal dataset -> YOLO format, with a flight-disjoint split.

HIT-UAV (Suo et al., Scientific Data 2023; Zenodo 10.5281/zenodo.7633134, CC0 1.0): 2,898 thermal images, 640x512,
sampled every 15 frames from drone videos at 60-130 m, day and night. File names encode the flight condition:
  <0 day | 1 night>_<altitude m>_<camera angle deg>_<weather>_<serial>.jpg

The official split is random over frames: more than half of its test images have a training frame within 2
serial numbers (about a second apart in the same video). So this script groups images by flight condition
(day/night x altitude x angle; 105 groups, our proxy for "one flight") and keeps whole groups out of training:
  test   ~20% of images per day/night x altitude stratum, whole groups never trained on
  val    ~10%, whole groups, for choosing settings
  train  the rest, minus...
  seen   15% of the training groups' frames held out: same flights as training, never trained on. Scoring the same
         model on seen vs test (with a COCO model as control) measures how much frame-level leakage inflates scores.

  python data/convert_hituav.py --src /path/to/HIT-UAV --out datasets/hituav_yolo
Expects <src>/normal_xml/JPEGImages/*.jpg and <src>/normal_json/annotations/{train,val,test}.json.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from _common import REPO, write_dataset_yaml

CLASSES = ["person", "car", "bicycle", "other_vehicle"]  # HIT-UAV ids 0-3; id 4 "DontCare" is dropped
SEED = 0


def condition(fn: str) -> dict:
    dn, alt, angle, weather, serial = Path(fn).stem.split("_")
    return {"night": dn == "1", "alt": int(alt), "angle": int(angle), "weather": weather, "serial": int(serial)}


def make_split(files: list[str]) -> dict[str, str]:
    groups = defaultdict(list)
    for f in files:
        c = condition(f)
        groups[(c["night"], c["alt"], c["angle"])].append(f)
    strata = defaultdict(list)
    for g in groups:
        strata[g[:2]].append(g)
    rng = random.Random(SEED)
    role = {}
    for _, gs in sorted(strata.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        total = sum(len(groups[g]) for g in gs)
        if len(gs) < 3:  # too few flights in this stratum to hold any out: train only
            for g in gs:
                for f in groups[g]:
                    role[f] = "train"
            continue
        n_test = n_val = 0
        for g in gs:  # whole groups only; never overshoot a target by more than a little
            k = len(groups[g])
            if n_test + k <= 0.22 * total or (n_test == 0 and k <= 0.35 * total):
                r, n_test = "test", n_test + k
            elif n_val + k <= 0.12 * total:
                r, n_val = "val", n_val + k
            else:
                r = "train"
            for f in groups[g]:
                role[f] = r
    train = sorted(f for f, r in role.items() if r == "train")
    for f in rng.sample(train, round(0.15 * len(train))):
        role[f] = "seen"
    return role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=str(REPO / "datasets/hituav_yolo"))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)

    images, anns = {}, defaultdict(list)
    for s in ("train", "val", "test"):
        d = json.loads((src / f"normal_json/annotations/{s}.json").read_text())
        id2fn = {im["id"]: im["filename"] for im in d["images"]}
        for im in d["images"]:
            images[im["filename"]] = (im["width"], im["height"], s)
        for x in d["annotation"]:
            anns[id2fn[x["image_id"]]].append(x)

    role = make_split(sorted(images))
    counts = defaultdict(lambda: defaultdict(int))
    for fn, (w, h, official) in images.items():
        r = role[fn]
        (out / "images" / r).mkdir(parents=True, exist_ok=True)
        (out / "labels" / r).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "normal_xml/JPEGImages" / fn, out / "images" / r / fn)
        lines = []
        for x in anns[fn]:
            c = x["category_id"]
            if c >= len(CLASSES):
                continue
            bx, by, bw, bh = x["bbox"]
            lines.append(f"{c} {(bx + bw / 2) / w:.6f} {(by + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out / "labels" / r / (Path(fn).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[r]["images"] += 1
        counts[r]["night"] += condition(fn)["night"]
        counts[r][f"official_{official}"] += 1

    sel = {"description": __doc__.split("\n\n")[0], "seed": SEED, "roles": {fn: role[fn] for fn in sorted(role)}}
    (REPO / "data/hituav_selection.json").write_text(json.dumps(sel, indent=0))
    yaml_path = write_dataset_yaml("hituav", out, CLASSES)
    yaml_path.write_text(yaml_path.read_text().replace("test: images/test\n", "test: images/test\nseen: images/seen\n"))
    for r in ("train", "val", "test", "seen"):
        print(r, dict(counts[r]))
    print("yaml ->", yaml_path)


if __name__ == "__main__":
    main()
