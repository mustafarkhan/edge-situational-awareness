"""Land search planning: a night thermal search for a missing person on open ground.

Same method as the sea plan (benchmarks/search_planning.py), on HIT-UAV thermal data, which records altitude and
camera angle for every image:
  1. every labelled person in the 477 held-out test images: size at the model input, altitude, angle, day/night,
     best confidence of an overlapping detection (IoU >= 0.3) from the thermal model
  2. P(detected) vs pixels; person size in metres from straight-down (angle 90) frames
  3. validation: does pixels(altitude) predict detection at each altitude?
  4. sweep rate per altitude for the HIT-UAV camera (Zenmuse XT2, 25 mm: 25 x 20 deg) and a wider 13 mm lens
     (45 x 37 deg), with ONE look per pass (no land video with identities exists here to measure look
     correlation, so this is the conservative choice)
  5. a QGroundControl mission for an example 1 km x 0.5 km area

  python benchmarks/land_search.py --data configs/datasets/hituav.yaml
Writes benchmarks/results/land_persons_test.csv, land_detection_curve.csv, land_detection_summary.json,
land_search_validation.csv, land_search_altitude.csv, land_search_example.json, docs/figures/land_search.png,
samples/land_search_example.plan.
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
sys.path.insert(0, str(REPO / "src"))
from esa.search import (  # noqa: E402
    Camera,
    DetectionCurve,
    evaluate,
    lawnmower,
    max_altitude_measured,
    path_length_m,
    qgc_plan,
    spacing_for_pod,
)

PERSON = 0
BINS = [0, 10, 12, 14, 16, 18, 20, 24, 28, 1e9]  # every bin has 30+ persons
LENSES = {"XT2 25 mm (25° lens, HIT-UAV camera)": 25.0, "XT2 13 mm (45° lens)": 45.0}
ALTS = (40, 60, 80, 100, 120, 150)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "configs/datasets/hituav.yaml"))
    ap.add_argument("--model", default=str(REPO / "models/hit_y11n_ep5.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--cache", default=str(REPO / "outputs/preds_hituav_test_person_640.json"))
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.data).read_text())
    root = Path(cfg["path"])
    images = sorted((root / "images/test").glob("*.jpg"))
    cache = Path(a.cache)
    if cache.exists():
        preds = json.loads(cache.read_text())
    else:
        from ultralytics import YOLO

        model, preds = YOLO(a.model), {}
        for p in images:
            r = model.predict(str(p), imgsz=640, conf=0.05, verbose=False, device="cpu")[0]
            preds[p.name] = [[*b, s] for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()) if int(c) == PERSON]  # fmt: skip
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(preds))

    rows, tp, fp = [], 0, 0
    for p in images:
        night, alt, angle = p.stem.split("_")[:3]
        gts = []
        for line in (root / "labels/test" / (p.stem + ".txt")).read_text().splitlines():
            c, cx, cy, bw, bh = map(float, line.split())
            if int(c) == PERSON:
                gts.append(((cx - bw / 2) * 640, (cy - bh / 2) * 512, (cx + bw / 2) * 640, (cy + bh / 2) * 512))
        for *b, s in preds.get(p.name, []):
            if s >= a.conf:
                tp, fp = (tp + 1, fp) if any(iou(b, g) >= 0.3 for g in gts) else (tp, fp + 1)
        for g in gts:
            best = max((s for *b, s in preds.get(p.name, []) if iou(b, g) >= 0.3), default=0.0)
            rows.append(
                {
                    "image": p.name,
                    "night": night == "1",
                    "alt_m": int(alt),
                    "angle_deg": int(angle),
                    "px_input": round(math.sqrt((g[2] - g[0]) * (g[3] - g[1])), 2),  # 640 px input = native width
                    "best_conf": round(best, 3),
                }
            )
    write_csv(R / "land_persons_test.csv", rows)
    px = np.array([r["px_input"] for r in rows])
    hit = np.array([r["best_conf"] >= a.conf for r in rows])
    night = np.array([r["night"] for r in rows])
    curve_rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (px >= lo) & (px < hi)
        if m.sum():
            curve_rows.append(
                {
                    "px_lo": lo,
                    "px_hi": hi if hi < 1e8 else "",
                    "px_median": round(float(np.median(px[m])), 2),
                    "persons": int(m.sum()),
                    "p_detect": round(float(hit[m].mean()), 3),
                }
            )
    write_csv(R / "land_detection_curve.csv", curve_rows)

    # person size in metres: straight-down frames only, 25 deg lens
    nadir = [r for r in rows if r["angle_deg"] == 90]
    size = [r["px_input"] * 2 * r["alt_m"] * math.tan(math.radians(25.0 / 2)) / 640 for r in nadir]
    q = [round(float(x), 3) for x in np.percentile(size, [10, 25, 50, 75, 90])]
    summ = {
        "persons": len(rows),
        "images": len(images),
        "conf": a.conf,
        "iou": 0.3,
        "recall": round(float(hit.mean()), 3),
        "recall_night": round(float(hit[night].mean()), 3),
        "recall_day": round(float(hit[~night].mean()), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "false_alarms_per_image": round(fp / len(images), 3),
        "px_input_p2_p50_p98": [round(float(x), 1) for x in np.percentile(px, [2, 50, 98])],
        "person_size_m": {"n": len(size), "hfov_deg": 25.0, "p10_25_50_75_90": q},
    }
    (R / "land_detection_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    curve = DetectionCurve.from_bins(
        [r["px_median"] for r in curve_rows], [r["p_detect"] for r in curve_rows], [r["persons"] for r in curve_rows],
        summ["px_input_p2_p50_p98"][0],
    )  # fmt: skip
    size_m = q[2]

    # validation by altitude, straight-down frames
    val = []
    cam25 = Camera(hfov_deg=25.0, aspect=640 / 512, input_px=640)
    for alt in sorted({r["alt_m"] for r in nadir}):
        sel = [r for r in nadir if r["alt_m"] == alt]
        if len(sel) < 15:
            continue
        val.append(
            {
                "alt_m": alt,
                "persons": len(sel),
                "px_median_measured": round(float(np.median([r["px_input"] for r in sel])), 1),
                "px_predicted": round(cam25.pixels_on(size_m, alt), 1),
                "p_measured": round(float(np.mean([r["best_conf"] >= a.conf for r in sel])), 3),
                "p_predicted": round(curve(cam25.pixels_on(size_m, alt)), 3),
            }
        )
    if val:
        write_csv(R / "land_search_validation.csv", val)
    print("validation:", *val, sep="\n  ")

    table = []
    for lens, hfov in LENSES.items():
        cam = Camera(hfov_deg=hfov, aspect=640 / 512, input_px=640)
        for alt in ALTS:
            p = evaluate(curve, cam, alt, a.speed, 4.0, size_m, independent_looks=1.0)
            table.append(
                {
                    "lens": lens,
                    "hfov_deg": hfov,
                    "alt_m": alt,
                    "px_on_person": round(p.px_on_target, 1),
                    "in_measured_range": p.in_measured_range,
                    "p_per_pass": round(p.p_pass, 3),
                    "footprint_m": round(p.footprint_w_m, 1),
                    "sweep_width_m": round(p.sweep_width_m, 1),
                    "km2_h_at_pod80": round(p.sweep_rate_km2_h / -math.log(1 - 0.8), 3),
                }
            )
        print(lens, "ceiling inside measured data:", round(max_altitude_measured(curve, cam, size_m)), "m")
    write_csv(R / "land_search_altitude.csv", table)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].bar(range(len(curve_rows)), [r["p_detect"] for r in curve_rows], color="#ef6c00")
    ax[0].set_xticks(
        range(len(curve_rows)), [f"{r['px_lo']}–{r['px_hi'] or '+'}" for r in curve_rows], rotation=45, fontsize=8
    )
    for i, r in enumerate(curve_rows):
        ax[0].text(i, r["p_detect"] + 0.01, r["persons"], ha="center", fontsize=7, color="#555")
    ax[0].set(ylim=(0, 1), xlabel="person size at the model input (px)", ylabel="P(person gets a box), one frame",
              title=f"Thermal, 477 held-out test images (conf {a.conf}, IoU ≥ 0.3)")  # fmt: skip
    for (lens, _), col in zip(LENSES.items(), ("#ef6c00", "#42a5f5")):
        t = [r for r in table if r["lens"] == lens]
        ok = [r for r in t if r["in_measured_range"]]
        ax[1].plot([r["alt_m"] for r in ok], [r["km2_h_at_pod80"] for r in ok], "o-", color=col, label=lens)
        bad = [r for r in t if not r["in_measured_range"]]
        if bad:
            ax[1].plot([r["alt_m"] for r in bad], [r["km2_h_at_pod80"] for r in bad], "o", mfc="none", color=col)
    ax[1].set(xlabel="altitude (m)", ylabel="km² searched per hour at 80% POD",
              title=f"Person {size_m:.2f} m, {a.speed:g} m/s, one look per pass · open = beyond data")  # fmt: skip
    for x in ax:
        x.grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPO / "docs/figures/land_search.png", dpi=120)

    # example: best lens and altitude within the measured data and the EU 120 m ceiling, 1 km x 0.5 km, 80% POD
    cand = [r for r in table if r["in_measured_range"] and r["alt_m"] <= 120]  # best lens + altitude, EU 120 m cap
    best = max(cand, key=lambda r: r["km2_h_at_pod80"])
    s = spacing_for_pod(best["sweep_width_m"], 0.8)
    pts = lawnmower(61.4600, 23.8800, 500, 1000, s)
    (REPO / "samples/land_search_example.plan").write_text(json.dumps(qgc_plan(pts, best["alt_m"], a.speed), indent=1))
    km = path_length_m(pts) / 1000
    ex = {
        "case": f"night, thermal, {best['hfov_deg']:g} deg lens, person {size_m:.2f} m, {a.speed:g} m/s",
        "alt_m": best["alt_m"],
        "track_spacing_m": round(s, 1),
        "tracks": len(pts) // 2,
        "path_km": round(km, 2),
        "flight_min": round(km * 1000 / a.speed / 60, 1),
        "pod": 0.8,
        "area_km2": 0.5,
    }
    (R / "land_search_example.json").write_text(json.dumps(ex, indent=1))
    print("example:", ex)


if __name__ == "__main__":
    main()
