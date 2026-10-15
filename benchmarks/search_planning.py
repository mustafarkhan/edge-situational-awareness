"""Search planning from measured numbers: what altitude and speed find swimmers fastest?

Inputs (all measured in this repo by benchmarks/detection_vs_range.py, or here):
  sea_detection_curve.csv     P(swimmer gets a box in one frame) vs its size in pixels at the model input
  sea_detection_summary.json  recall / precision / false alarms, size range covered, swimmer size in metres
  sea_swimmers_test.csv       per-swimmer altitude / pitch / confidence (validation by altitude)
  SeaDronesSee-MOT clips      how many consecutive looks at a swimmer are really independent

  python benchmarks/search_planning.py --clips outputs/sea_mot/DJI_0055 outputs/sea_mot/DJI_0059 ...

Writes benchmarks/results/search_looks.csv, search_validation.csv, search_altitude.csv, search_example.json,
docs/figures/search_altitude.png and an example QGroundControl mission samples/search_example.plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

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

SWIMMER_MOT = {1, 2}  # SeaDronesSee-MOT: swimmer, swimmer with life jacket
EU_MAX_ALT_M = 120  # EU open category ceiling
ALTS = (20, 30, 40, 50, 60, 80, 100, 120)


def load_curve(min_px: float) -> DetectionCurve:
    rows = list(csv.DictReader((R / "sea_detection_curve.csv").open()))
    return DetectionCurve.from_bins(
        [float(r["px_median"]) for r in rows],
        [float(r["p_detect"]) for r in rows],
        [int(r["swimmers"]) for r in rows],
        min_px,
    )


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def hit_sequences(clip: Path, model, imgsz: int, conf: float, cache: Path) -> list[list[int]]:
    """For each ground-truth swimmer track: 1/0 per frame, detected (IoU >= 0.3, no tracker) or not."""
    if cache.exists():
        dets = json.loads(cache.read_text())
    else:
        dets = {}
        for f in sorted((clip / "frames").iterdir()):
            r = model.predict(str(f), imgsz=imgsz, conf=0.05, verbose=False, device="cpu")[0]
            dets[f.stem] = [
                [*b, s] for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()) if c == 0
            ]
        cache.write_text(json.dumps(dets))
    frames = sorted(dets)
    tracks: dict[int, dict[int, tuple]] = {}
    for line in (clip / "gt.txt").read_text().splitlines():
        f, tid, x, y, w, h, _, cls, _ = line.split(",")
        if int(cls) in SWIMMER_MOT:
            tracks.setdefault(int(tid), {})[int(f) - 1] = (float(x), float(y), float(x) + float(w), float(y) + float(h))
    seqs = []
    for boxes in tracks.values():
        fs = sorted(boxes)
        seq = []
        for fi in range(fs[0], fs[-1] + 1):
            g = boxes.get(fi)
            if g is None:  # not labelled in this frame (occluded / out of view): end the run here
                if len(seq) > 1:
                    seqs.append(seq)
                seq = []
                continue
            seq.append(int(any(d[4] >= conf and iou(g, d[:4]) >= 0.3 for d in dets[frames[fi]])))
        if len(seq) > 1:
            seqs.append(seq)
    return seqs


def looks_analysis(seqs: list[list[int]], ns=(1, 2, 3, 5, 10, 15, 30, 60, 90)) -> list[dict]:
    """P(at least one detection in a window of n frames), measured vs assuming independent frames."""
    p1 = float(np.mean([v for s in seqs for v in s]))
    out = []
    for n in ns:
        wins = [max(s[i : i + n]) for s in seqs for i in range(0, len(s) - n + 1)]
        if not wins:
            continue
        pa = float(np.mean(wins))
        n_eff = math.log(1 - pa) / math.log(1 - p1) if pa < 1 and 0 < p1 < 1 else float("inf")
        out.append(
            {
                "frames": n,
                "seconds": round(n / 30, 2),
                "windows": len(wins),
                "p_any_measured": round(pa, 3),
                "p_any_if_independent": round(1 - (1 - p1) ** n, 3),
                "independent_looks_equivalent": round(n_eff, 2),
            }
        )
    return out


def validation(curve: DetectionCurve, cam: Camera, size_m: float, conf: float) -> list[dict]:
    """Does curve(pixels(altitude)) predict the detection rate actually seen at each altitude?
    Near-straight-down DJI 4K frames only (pitch >= 70 deg below the horizon), as the model assumes."""
    rows = [
        r
        for r in csv.DictReader((R / "sea_swimmers_test.csv").open())
        if r["drone"] == "mavic" and r["pitch_deg"] and float(r["pitch_deg"]) >= 70 and r["alt_m"]
    ]
    out = []
    for lo, hi in ((0, 25), (25, 40), (40, 60), (60, 80), (80, 130)):
        sel = [r for r in rows if lo <= float(r["alt_m"]) < hi]
        if len(sel) < 20:
            continue
        alt = float(np.median([float(r["alt_m"]) for r in sel]))
        out.append(
            {
                "alt_bin_m": f"{lo}-{hi}",
                "swimmers": len(sel),
                "alt_median_m": round(alt, 1),
                "px_median_measured": round(float(np.median([float(r["px_input"]) for r in sel])), 1),
                "px_predicted": round(cam.pixels_on(size_m, alt), 1),
                "p_measured": round(float(np.mean([float(r["best_conf"]) >= conf for r in sel])), 3),
                "p_predicted": round(curve(cam.pixels_on(size_m, alt)), 3),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--model", default=str(REPO / "models/sds_y11n_ep4.pt"))
    ap.add_argument("--speed", type=float, default=10.0, help="ground speed for the altitude table (m/s)")
    ap.add_argument("--cache-dir", default=str(REPO / "outputs"))
    ap.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=[4.0, 1.0],
        help="processing rates to plan for (e.g. measured on Jetson)",
    )
    a = ap.parse_args()

    summ = json.loads((R / "sea_detection_summary.json").read_text())
    conf = summ["conf"]
    min_px = summ["px_input_p2_p50_p98"][0]  # below the 2nd percentile there is too little data to trust
    curve = load_curve(min_px)
    size_q = summ["swimmer_size_m"]["p10_25_50_75_90"]
    cam = Camera(hfov_deg=summ["swimmer_size_m"]["hfov_deg_assumed"])
    sizes = {
        f"whole swimmer box ({size_q[2]:.2f} m, dataset median)": size_q[2],
        "head and shoulders (0.5 m)": 0.5,
        "head only (0.3 m)": 0.3,
    }

    # 1. how independent are consecutive looks? (real clips, detector only, no tracker)
    if a.clips:
        from ultralytics import YOLO

        model, seqs = YOLO(a.model), []
        for c in map(Path, a.clips):
            seqs += hit_sequences(c, model, 1280, conf, Path(a.cache_dir) / f"dets_{c.name}_1280.json")
        write_csv(R / "search_looks.csv", looks_analysis(seqs))
    looks = list(csv.DictReader((R / "search_looks.csv").open()))
    secs = [float(r["seconds"]) for r in looks]
    neq = [float(r["independent_looks_equivalent"]) for r in looks]

    # A pass lasting T s counts min(frames processed, independent looks measured for a T-s window). Held flat
    # beyond the longest measured window (conservative). The clips come from a slow drone, so looks during a
    # real sweep are probably less correlated than this (also conservative).
    def cap(t_pass: float) -> float:
        return float(np.interp(t_pass, secs, neq))

    print("looks:", *looks, sep="\n  ")

    # 2. validate the pixels -> probability model against altitude
    val = validation(curve, cam, size_q[2], conf)
    write_csv(R / "search_validation.csv", val)
    print("validation:", *val, sep="\n  ")

    # 3. the planning table: what is visible x altitude x processing rate
    table = []
    for size_name, size_m in sizes.items():
        for fps in a.fps:
            for alt in ALTS:
                t_pass = cam.footprint_m(alt)[1] / a.speed
                p = evaluate(curve, cam, alt, a.speed, fps, size_m, cap(t_pass))
                table.append(
                    {
                        "visible_size": size_name,
                        "size_m": size_m,
                        "fps": fps,
                        "alt_m": alt,
                        "speed_mps": a.speed,
                        "px_on_swimmer": round(p.px_on_target, 1),
                        "in_measured_range": p.in_measured_range,
                        "p_single_frame": round(p.p_single, 3),
                        "looks_per_pass": round(p.looks, 1),
                        "independent_looks": round(min(max(p.looks, 1), cap(t_pass)), 2),
                        # frames/s at which looks per pass reach the independent-look cap: more is wasted on POD
                        "min_fps_needed": round(cap(t_pass) * a.speed / cam.footprint_m(alt)[1], 2),
                        "p_per_pass": round(p.p_pass, 3),
                        "footprint_m": round(p.footprint_w_m, 1),
                        "sweep_width_m": round(p.sweep_width_m, 1),
                        "km2_h_at_pod80": round(p.sweep_rate_km2_h / -math.log(1 - 0.8), 3),
                    }
                )
    write_csv(R / "search_altitude.csv", table)
    ceilings = {k: round(max_altitude_measured(curve, cam, v), 0) for k, v in sizes.items()}
    print("highest altitude inside the measured data:", ceilings)

    # 4. figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    c = list(csv.DictReader((R / "sea_detection_curve.csv").open()))
    ax[0].bar(range(len(c)), [float(r["p_detect"]) for r in c], color="#42a5f5")
    ax[0].set_xticks(range(len(c)), [f"{r['px_lo']}–{r['px_hi'] or '+'}" for r in c], rotation=45, fontsize=8)
    for i, r in enumerate(c):
        ax[0].text(i, float(r["p_detect"]) + 0.01, r["swimmers"], ha="center", fontsize=7, color="#555")
    ax[0].set(
        ylim=(0, 1),
        xlabel="swimmer size at the model input (px)",
        ylabel="P(swimmer gets a box), one frame",
        title=f"Measured: 1,200 unseen test images (conf {conf}, IoU ≥ 0.3)",
    )
    cols = ["#f9a825", "#ef6c00", "#c62828"]
    for (name, _), col in zip(sizes.items(), cols):
        t = [r for r in table if r["visible_size"] == name and r["fps"] == a.fps[0]]
        xs = [r["alt_m"] for r in t]
        ys = [r["km2_h_at_pod80"] for r in t]
        ok = [r["in_measured_range"] for r in t]
        ax[1].plot([x for x, k in zip(xs, ok) if k], [y for y, k in zip(ys, ok) if k], "o-", color=col, label=name)
        bad = [(x, y) for x, y, k in zip(xs, ys, ok) if not k]
        if bad:
            ax[1].plot(*zip(*bad), "o", mfc="none", color=col)
    ax[1].set(
        xlabel="altitude (m)",
        ylabel="km² searched per hour at 80% POD",
        title=f"{a.speed:g} m/s, {a.fps[0]:g} frames/s · open circles = beyond measured data",
    )
    for x in ax:
        x.grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPO / "docs/figures/search_altitude.png", dpi=120)

    # 5. example mission: 500 m x 1 km strip of lake Pyhäjärvi off Tampere, head-and-shoulders case, 4 fps,
    # highest altitude inside the measured data and the EU 120 m ceiling, 80% POD
    hs = [
        r
        for r in table
        if r["size_m"] == 0.5 and r["fps"] == a.fps[0] and r["in_measured_range"] and r["alt_m"] <= EU_MAX_ALT_M
    ]
    best = max(hs, key=lambda r: r["km2_h_at_pod80"])
    s = spacing_for_pod(best["sweep_width_m"], 0.8)
    pts = lawnmower(61.4900, 23.7350, 500, 1000, s, heading_deg=0)
    out = REPO / "samples/search_example.plan"
    out.write_text(json.dumps(qgc_plan(pts, best["alt_m"], a.speed), indent=1))
    km = path_length_m(pts) / 1000
    ex = {
        "case": f"head and shoulders visible (0.5 m), {a.fps[0]:g} frames/s, {a.speed:g} m/s",
        "alt_m": best["alt_m"],
        "track_spacing_m": round(s, 1),
        "tracks": len(pts) // 2,
        "path_km": round(km, 2),
        "flight_min": round(km * 1000 / a.speed / 60, 1),
        "pod": 0.8,
        "area_km2": 0.5,
    }
    (R / "search_example.json").write_text(json.dumps({"ceilings_m": ceilings, "example": ex}, indent=1))
    print("example mission:", ex, "->", out)


if __name__ == "__main__":
    main()
