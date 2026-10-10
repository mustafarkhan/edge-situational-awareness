"""Where does the detector fail? Recall broken down by object size, occlusion, class and
image brightness, plus pictures of the worst images.

Uses the RAW VisDrone annotations because they carry occlusion and truncation levels that the
YOLO conversion drops. Matching is class-agnostic, greedy by confidence, IoU >= 0.5, so "found"
means "there is a box on it", whatever class it was given.

  python data/failure_analysis.py --model weights/ft_y11n_ep3.pt --imgsz 1280 \
      --images datasets/visdrone_yolo/images/val --raw-ann /path/VisDrone2019-DET-val/annotations
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
SUPER = {"pedestrian": "person", "people": "person", "bicycle": "two-wheeler", "motor": "two-wheeler",
         "tricycle": "two-wheeler", "awning-tricycle": "two-wheeler",
         "car": "vehicle", "van": "vehicle", "truck": "vehicle", "bus": "vehicle"}
OCC = {0: "none", 1: "partial (1-50%)", 2: "heavy (50-100%)"}
SIZE_EDGES = [0, 8, 16, 32, 64, 1e9]
SIZE_LABELS = ["< 8 px", "8-16 px", "16-32 px", "32-64 px", ">= 64 px"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def load_gt(ann: Path):
    out = []
    for line in ann.read_text().strip().splitlines():
        v = [int(x) for x in line.strip().strip(",").split(",")[:8]]
        x, y, w, h, score, cat, trunc, occ = v
        if score == 0 or cat in (0, 11) or w < 1 or h < 1:
            continue
        out.append({"box": (x, y, x + w, y + h), "cls": NAMES[cat - 1], "occ": occ, "trunc": trunc})
    return out


def predict_all(model_path, images, imgsz, conf, cache: Path):
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO
    m = YOLO(model_path)
    preds = {}
    for p in images:
        r = m.predict(str(p), imgsz=imgsz, conf=conf, max_det=500, verbose=False)[0]
        preds[p.name] = [[*b, s] for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(preds))
    return preds


def draw(img_path, gts, found, preds, matched_pred, out_path, title):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 700)
    for i, p in enumerate(preds):
        if i not in matched_pred:
            d.rectangle(p[:4], outline=(255, 214, 0), width=lw)          # false positive: yellow
    for g, f in zip(gts, found):
        d.rectangle(g["box"], outline=(0, 230, 118) if f else (255, 40, 40), width=lw)  # green found / red missed
    s = 1280 / im.width
    im = im.resize((1280, round(im.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1280, 34], fill=(0, 0, 0))
    d.text((10, 8), title, fill=(255, 255, 255))
    im.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--raw-ann", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-worst", type=int, default=4)
    ap.add_argument("--fig-dir", default=str(REPO / "docs" / "figures" / "failures"))
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "failure_analysis.md"))
    args = ap.parse_args()

    images = sorted(Path(args.images).glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    cache = REPO / "outputs" / f"preds_{Path(args.model).stem}_{args.imgsz}_c{args.conf}_{len(images)}.json"
    preds = predict_all(args.model, images, args.imgsz, args.conf, cache)

    rows, per_image = [], []
    for p in images:
        gts = load_gt(Path(args.raw_ann) / (p.stem + ".txt"))
        pr = sorted(preds[p.name], key=lambda x: -x[4])
        im = Image.open(p)
        W, H = im.size
        bright = float(np.asarray(im.convert("L").resize((160, 120))).mean())
        used, found = set(), [False] * len(gts)
        for pi, b in enumerate(pr):  # greedy: highest-confidence prediction takes its best free GT
            best, bj = 0.5, None
            for j, g in enumerate(gts):
                if not found[j] and (v := iou(b[:4], g["box"])) >= best:
                    best, bj = v, j
            if bj is not None:
                found[bj] = True
                used.add(pi)
        for g, f in zip(gts, found):
            x1, y1, x2, y2 = g["box"]
            size = np.sqrt((x2 - x1) * (y2 - y1)) * args.imgsz / max(W, H)
            rows.append({"image": p.name, "found": f, "class": SUPER[g["cls"]], "occlusion": OCC.get(g["occ"], "?"),
                         "size": pd.cut([size], SIZE_EDGES, labels=SIZE_LABELS, right=False)[0], "brightness": bright})
        per_image.append({"image": p.name, "n_gt": len(gts), "recall": sum(found) / max(len(gts), 1),
                          "n_pred": len(pr), "fp": len(pr) - len(used), "brightness": bright,
                          "_gts": gts, "_found": found, "_pr": pr, "_used": used, "_path": p})

    df = pd.DataFrame(rows)
    pi = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in per_image])
    df["light"] = np.where(df["brightness"] < 70, "dusk/night", "daylight")
    pi["light"] = np.where(pi["brightness"] < 70, "dusk/night", "daylight")

    def tab(col, order=None):
        g = df.groupby(col, observed=True)["found"].agg(objects="size", recall="mean").reset_index()
        if order:
            g[col] = pd.Categorical(g[col], order, ordered=True)
            g = g.sort_values(col)
        g["recall"] = g["recall"].map(lambda v: f"{v:.1%}")
        return g.to_markdown(index=False)

    light_img = pi.groupby("light").agg(images=("image", "size"), gt=("n_gt", "sum"), preds=("n_pred", "sum"),
                                        false_pos=("fp", "sum")).reset_index()
    light_img["precision"] = (1 - light_img["false_pos"] / light_img["preds"].clip(lower=1)).map(lambda v: f"{v:.1%}")

    recs = []
    for dim, order in (("size", SIZE_LABELS), ("occlusion", list(OCC.values())), ("class", None), ("light", None)):
        g = df.groupby(dim, observed=True)["found"].agg(objects="size", recall="mean").reset_index()
        for _, r in g.iterrows():
            recs.append({"dimension": dim, "bucket": str(r[dim]), "objects": int(r["objects"]),
                         "recall": round(float(r["recall"]), 4)})
    recs.append({"dimension": "overall", "bucket": "all", "objects": len(df), "recall": round(float(df["found"].mean()), 4)})
    pd.DataFrame(recs).to_csv(Path(args.out).with_suffix(".csv"), index=False)

    md = [f"# Failure analysis\n",
          f"Model `{Path(args.model).name}`, input {args.imgsz} px, confidence {args.conf}, "
          f"{len(images)} VisDrone val images, {len(df):,} labelled objects. "
          f"Found = any predicted box with IoU >= 0.5 (class-agnostic).\n",
          f"**Overall recall: {df['found'].mean():.1%}**\n",
          "## By object size (in model-input pixels)\n", tab("size", SIZE_LABELS),
          "\n## By occlusion (VisDrone annotation)\n", tab("occlusion", list(OCC.values())),
          "\n## By class\n", tab("class"),
          "\n## By lighting (mean image brightness < 70 of 255 = dusk/night; checked visually)\n", tab("light"),
          "\n", light_img.to_markdown(index=False)]

    fig = Path(args.fig_dir)
    fig.mkdir(parents=True, exist_ok=True)
    cand = sorted((r for r in per_image if r["n_gt"] >= 20), key=lambda r: r["recall"])[: args.n_worst]
    darkest = min(per_image, key=lambda r: r["brightness"])
    if darkest not in cand:
        cand.append(darkest)
    md.append("\n## Example failure images\n\nGreen = found, red = missed, yellow = false positive.\n")
    for r in cand:
        out = fig / f"{Path(r['image']).stem}.jpg"
        title = (f"{r['image']}  recall {r['recall']:.0%} ({sum(r['_found'])}/{r['n_gt']})  "
                 f"false pos {r['fp']}  brightness {r['brightness']:.0f}")
        draw(r["_path"], r["_gts"], r["_found"], r["_pr"], r["_used"], out, title)
        md.append(f"- `{out.relative_to(REPO) if out.is_relative_to(REPO) else out}` — recall {r['recall']:.0%} of {r['n_gt']} objects, "
                  f"{r['fp']} false positives, brightness {r['brightness']:.0f}")
    Path(args.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
