"""Know your data before you train. Prints class balance and object-size statistics,
and shows how small objects get after the image is resized to the model's input size.

  python data/analyse_dataset.py --data configs/datasets/seadronessee.yaml --split train --imgsz 640
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from _common import REPO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280])
    ap.add_argument("--max-images", type=int, default=5000, help="sample for speed")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    root, names = Path(cfg["path"]), cfg["names"]
    imgs = sorted(p for p in (root / cfg[args.split]).iterdir() if p.suffix.lower() in IMG_EXT)
    if len(imgs) > args.max_images:
        imgs = list(np.random.default_rng(0).choice(imgs, args.max_images, replace=False))

    cls_count, sizes, longest, per_img = Counter(), [], [], []
    for p in imgs:
        w, h = Image.open(p).size
        lbl = root / "labels" / args.split / (p.stem + ".txt")
        lines = lbl.read_text().splitlines() if lbl.exists() else []
        per_img.append(len(lines))
        for line in lines:
            c, _, _, bw, bh = line.split()
            cls_count[names[int(c)]] += 1
            sizes.append(np.sqrt(float(bw) * w * float(bh) * h))  # sqrt(area) in original pixels
            longest.append(max(w, h))

    sizes, longest = np.array(sizes), np.array(longest)
    name = Path(args.data).stem
    print(f"\n## {name} / {args.split}: {len(imgs)} images, {len(sizes)} boxes\n")
    print("| class | boxes | share |\n|---|---|---|")
    for k, v in cls_count.most_common():
        print(f"| {k} | {v} | {v / len(sizes):.1%} |")
    print(f"\nObjects per image: mean {np.mean(per_img):.1f}, max {np.max(per_img)}, "
          f"images with none: {sum(n == 0 for n in per_img)}")
    print(f"\nsqrt(area) in original pixels: median {np.median(sizes):.1f}, p10 {np.percentile(sizes, 10):.1f}")
    print("\n| model input | median object size | < 8 px | < 16 px | < 32 px (COCO 'small') |\n|---|---|---|---|---|")
    for s in args.imgsz:
        eff = sizes * s / longest  # letterbox scales the longest side to s
        print(f"| {s} | {np.median(eff):.1f} px | {np.mean(eff < 8):.1%} | {np.mean(eff < 16):.1%} | {np.mean(eff < 32):.1%} |")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(*zip(*cls_count.most_common()))
    ax[0].set_title("Class balance"); ax[0].tick_params(axis="x", rotation=45)
    for s in args.imgsz:
        ax[1].hist(sizes * s / longest, bins=np.logspace(0, 3, 50), alpha=0.5, label=f"input {s}")
    ax[1].set_xscale("log"); ax[1].axvline(32, ls="--", c="k"); ax[1].legend()
    ax[1].set_title("Object size after resize (sqrt area, px)")
    fig.tight_layout()
    out = REPO / "docs" / "figures" / f"{name}_{args.split}_stats.png"
    fig.savefig(out, dpi=120)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
