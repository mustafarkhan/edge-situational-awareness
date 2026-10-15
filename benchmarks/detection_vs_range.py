"""How likely is a swimmer to be detected, as a function of how many pixels it covers?

This is the measured input to search planning (src/esa/search.py): flying higher covers more sea per minute
but makes each swimmer smaller, and this curve says what that costs.

For every labelled swimmer in the 1,200 SeaDronesSee test images (7 videos never used for training) it records
its size at the model input, the drone's altitude and gimbal pitch from the image metadata, and the highest
confidence of any detection that overlaps it (IoU >= 0.3: for search, "a box on the swimmer" counts, even if
the box is not tight). From the near-straight-down DJI frames it also estimates the swimmer's visible size in
metres, which turns "pixels" into "altitude".

  python benchmarks/detection_vs_range.py --model models/sds_y11n_ep4.pt \
      --ann-train /path/to/instances_train.json --ann-val /path/to/instances_val.json

Writes benchmarks/results/sea_swimmers_test.csv (one row per swimmer) and sea_detection_curve.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
R = REPO / "benchmarks" / "results"
SWIMMER = 0  # class index in configs/datasets/seadronessee.yaml
BINS = [0, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 1e9]


def meta_of(im: dict) -> tuple[float | None, float | None]:
    """(altitude m, gimbal pitch deg) from either of the two metadata styles in SeaDronesSee ODv2."""
    m = im.get("meta") or {}
    alt = m.get("height_above_takeoff(meter)", m.get("altitude"))
    pitch = m.get("gimbal_pitch(degrees)", m.get("gimbal_pitch"))
    return (float(alt) if alt is not None else None, float(pitch) if pitch is not None else None)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def predict(model_path: str, images: list[Path], imgsz: int, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO

    model, out = YOLO(model_path), {}
    for i, p in enumerate(images):
        r = model.predict(str(p), imgsz=imgsz, conf=0.05, max_det=300, device="cpu", verbose=False)[0]
        out[p.name] = [
            [*map(float, b), float(s)]
            for b, c, s in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
            if int(c) == SWIMMER
        ]
        if i % 100 == 0:
            print(f"{i}/{len(images)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "models/sds_y11n_ep4.pt"))
    ap.add_argument("--data", default=str(REPO / "configs/datasets/seadronessee.yaml"))
    ap.add_argument("--ann-train", required=True)
    ap.add_argument("--ann-val", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25, help="operating point for the summary curve")
    ap.add_argument("--hfov", type=float, default=65.5, help="DJI Mavic 2 (28 mm equivalent), for the size estimate")
    ap.add_argument("--cache", default=str(REPO / "outputs/preds_sds_test_swimmer_1280.json"))
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.data).read_text())
    root = Path(cfg["path"])
    sel = json.loads((REPO / "data/seadronessee_selection.json").read_text())["test"]
    meta = {}
    for split, path in (("train", a.ann_train), ("val", a.ann_val)):
        for im in json.loads(Path(path).read_text())["images"]:
            meta[(split, im["file_name"])] = im
    images = [root / "images/test" / fn for fn, _ in sel]
    preds = predict(a.model, images, a.imgsz, Path(a.cache))

    rows, tp, fp = [], 0, 0
    for fn, split in sel:
        im = meta[(split, fn)]
        alt, pitch = meta_of(im)
        W, H = im["width"], im["height"]
        lbl = root / "labels/test" / (Path(fn).stem + ".txt")
        sx, sy = W / max(W, H), H / max(W, H)  # IoU on normalised coordinates must respect the aspect ratio
        gts = []
        for line in lbl.read_text().splitlines() if lbl.exists() else []:
            c, cx, cy, bw, bh = map(float, line.split())
            if int(c) == SWIMMER:
                gts.append(((cx - bw / 2) * sx, (cy - bh / 2) * sy, (cx + bw / 2) * sx, (cy + bh / 2) * sy))
        # precision at the operating point: a swimmer box that overlaps no labelled swimmer is a false alarm
        for x1, y1, x2, y2, sc in preds.get(fn, []):
            if sc >= a.conf:
                if any(iou((x1 * sx, y1 * sy, x2 * sx, y2 * sy), g) >= 0.3 for g in gts):
                    tp += 1
                else:
                    fp += 1
        for line in lbl.read_text().splitlines() if lbl.exists() else []:
            c, cx, cy, bw, bh = map(float, line.split())
            if int(c) != SWIMMER:
                continue
            g = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            gs = (g[0] * sx, g[1] * sy, g[2] * sx, g[3] * sy)
            best = 0.0
            for x1, y1, x2, y2, s in preds.get(fn, []):
                if iou(gs, (x1 * sx, y1 * sy, x2 * sx, y2 * sy)) >= 0.3:
                    best = max(best, s)
            side = math.sqrt(bw * W * bh * H)  # sqrt(area) in native pixels
            rows.append(
                {
                    "image": fn,
                    "drone": im["source"]["drone"],
                    "native_w": W,
                    "px_native": round(side, 2),
                    "px_input": round(side * a.imgsz / max(W, H), 2),
                    "alt_m": round(alt, 1) if alt is not None else "",
                    "pitch_deg": round(pitch, 1) if pitch is not None else "",
                    "best_conf": round(best, 3),
                }
            )
    out = R / "sea_swimmers_test.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    px = np.array([r["px_input"] for r in rows])
    hit = np.array([r["best_conf"] >= a.conf for r in rows])
    curve = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (px >= lo) & (px < hi)
        if m.sum():
            curve.append(
                {
                    "px_lo": lo,
                    "px_hi": hi if hi < 1e8 else "",
                    "px_median": round(float(np.median(px[m])), 2),
                    "swimmers": int(m.sum()),
                    "p_detect": round(float(hit[m].mean()), 3),
                }
            )
    with (R / "sea_detection_curve.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    summary = {
        "swimmers": len(rows),
        "conf": a.conf,
        "iou": 0.3,
        "recall": round(float(hit.mean()), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "false_alarms_per_image": round(fp / len(sel), 3),
        "images": len(sel),
        "px_input_p2_p50_p98": [round(float(x), 1) for x in np.percentile(px, [2, 50, 98])],
    }
    print(summary)
    for c in curve:
        print(c)

    # visible swimmer size in metres: near-straight-down frames from the 4K DJI camera only.
    # SeaDronesSee ODv2 gimbal pitch: degrees below the horizon, 90 = straight down (the trinity frames are all 90).
    gsd = [
        r["px_native"] * 2 * float(r["alt_m"]) * math.tan(math.radians(a.hfov / 2)) / r["native_w"]
        for r in rows
        if r["drone"] == "mavic" and r["pitch_deg"] != "" and float(r["pitch_deg"]) >= 70 and r["alt_m"] != ""
    ]
    if gsd:
        q = np.percentile(gsd, [10, 25, 50, 75, 90])
        summary["swimmer_size_m"] = {
            "n": len(gsd),
            "hfov_deg_assumed": a.hfov,
            "p10_25_50_75_90": [round(float(x), 3) for x in q],
        }
        print(f"swimmer size (sqrt area, m) from {len(gsd)} near-nadir boxes: p10/25/50/75/90 = {np.round(q, 2)}")
    (R / "sea_detection_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
