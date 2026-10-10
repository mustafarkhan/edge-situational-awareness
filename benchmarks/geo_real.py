"""Geolocation on REAL drone footage, without ground-truth object positions.

Real clips (data/prepare_sds_mot.py) come with the drone's own telemetry per frame (GPS, altitude, gimbal
heading and pitch), but nobody measured where the swimmers actually were. Physics gives a check instead:
a person in the water moves at most ~1 m/s. Placing each ground-truth box on the map with a camera model and
measuring how fast the result appears to move shows how wrong the geometry is: bad geometry makes swimmers
"move" with the drone. Two models are compared on identical boxes and telemetry:

  nadir       the original assumption: camera points straight down (pitch forced to 90)
  tilt-aware  src/esa/geo.py with the logged gimbal pitch (ray / sea-surface intersection)

Ground-truth boxes are used, not detections, so detector errors don't mix in.

  python benchmarks/geo_real.py --clips outputs/sea_mot/DJI_0055 outputs/sea_mot/DJI_0059 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from esa.geo import CameraGeo, pixel_to_latlon  # noqa: E402

EARTH = 111_320.0


def metres(a, b):
    return math.hypot((b[0] - a[0]) * EARTH, (b[1] - a[1]) * EARTH * math.cos(math.radians(a[0])))


def smooth_telemetry(tel: list[dict]) -> list[dict]:
    """The log is sample-and-hold (values change every few frames). Re-time each field by linear
    interpolation between the frames where it changes, so positions don't jump in steps."""
    out = [dict(t) for t in tel]
    n = len(tel)
    for k in ("lat", "lon", "alt_m", "heading_deg", "pitch_deg"):
        knots = [0] + [i for i in range(1, n) if tel[i][k] != tel[i - 1][k]]
        if knots[-1] != n - 1:
            knots.append(n - 1)
        vals = np.array([tel[i][k] for i in knots], float)
        if k == "heading_deg":
            vals = np.degrees(np.unwrap(np.radians(vals)))
        for i, v in enumerate(np.interp(np.arange(n), knots, vals)):
            out[i][k] = float(v % 360) if k == "heading_deg" else float(v)
    return out


def image_rotation(clip: Path) -> float | None:
    """Rotation of the scene in the image between the first and last frame (deg, + = clockwise), from a
    least-squares 2-D similarity fit to ground-truth track centres present in both frames. Rough under
    perspective and with moving objects, but enough to compare with the logged heading change."""
    pts = defaultdict(dict)
    for line in (clip / "gt.txt").read_text().splitlines():
        f, tid, x, y, w, h, *_ = line.split(",")
        pts[int(f) - 1][int(tid)] = (float(x) + float(w) / 2, float(y) + float(h) / 2)
    f0, f1 = min(pts), max(pts)
    common = [k for k in pts[f0] if k in pts[f1]]
    if len(common) < 2:
        return None
    a = np.array([pts[f0][k] for k in common])
    b = np.array([pts[f1][k] for k in common])
    a, b = a - a.mean(0), b - b.mean(0)
    return round(math.degrees(math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())), 1)


def evaluate(clip: Path, model: str, frame_w: int, frame_h: int, lag: int = 0, smooth: bool = False):
    """lag > 0: telemetry row f + lag describes video frame f (log behind the video)."""
    meta = json.loads((clip / "meta.json").read_text())
    fps = meta["fps"]
    cats = {int(k): v for k, v in meta["categories"].items()}
    tel = [json.loads(line) for line in (clip / "telemetry.jsonl").read_text().splitlines() if line.strip()]
    if smooth:
        tel = smooth_telemetry(tel)
    tracks = defaultdict(list)  # track -> [(frame, (lat, lon))]
    kind = {}
    for line in (clip / "gt.txt").read_text().splitlines():
        f, tid, x, y, w, h, _, c, _ = line.split(",")
        f, tid = int(f) - 1, int(tid)
        if not 0 <= f + lag < len(tel):
            continue
        t = tel[f + lag]
        pitch = 90.0 if model == "nadir" else t["pitch_deg"]
        cam = CameraGeo(t["lat"], t["lon"], t["alt_m"], t["hfov_deg"], t["heading_deg"], pitch)
        pos = pixel_to_latlon(float(x) + float(w) / 2, float(y) + float(h) / 2, frame_w, frame_h, cam)
        if pos:
            tracks[tid].append((f, pos))
            kind[tid] = "swimmer" if ("swimmer" in cats[int(c)] or "jacket" in cats[int(c)]) else cats[int(c)]
    win = max(1, round(fps))  # 1-second windows
    out = defaultdict(lambda: {"speed": [], "spread": []})
    for tid, pts in tracks.items():
        if len(pts) < win + 1:
            continue
        pts.sort()
        byf = dict(pts)
        sp = [metres(byf[f], byf[f + win]) * fps / win for f, _ in pts if f + win in byf]
        med = (float(np.median([p[0] for _, p in pts])), float(np.median([p[1] for _, p in pts])))
        spread = float(np.median([metres(med, p) for _, p in pts]))
        if sp:
            out[kind[tid]]["speed"].append(float(np.median(sp)))
            out[kind[tid]]["spread"].append(spread)
    return meta, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPO / "benchmarks/results/geo_real.csv"))
    ap.add_argument("--max-lag", type=int, default=45, help="frames; lag sweep range for the sync check")
    args = ap.parse_args()
    from PIL import Image

    clips = [Path(c) for c in args.clips]
    sizes = {c: Image.open(next((c / "frames").glob("*.jpg"))).size for c in clips}

    def swimmer_speed(c, model, lag, smooth):
        _, res = evaluate(c, model, *sizes[c], lag=lag, smooth=smooth)
        sp = res["swimmer"]["speed"] or res["boat"]["speed"]
        return float(np.median(sp)) if sp else float("nan")

    # Is there a fixed clock offset between video and log? Choose the lag on the OTHER clips only
    # (leave-one-out), so no clip is scored with an offset fitted to itself.
    lags = list(range(-args.max_lag, args.max_lag + 1, 3))
    curve = {c: [swimmer_speed(c, "tilt-aware", lg, True) for lg in lags] for c in clips}
    loo = {}
    for c in clips:
        others = [curve[o] for o in clips if o != c]
        loo[c] = lags[int(np.nanargmin(np.nanmean(others, axis=0)))] if others else 0
    print("lag sweep (swimmer apparent speed, tilt-aware, smoothed telemetry):")
    for c in clips:
        print(
            f"  {c.name:12s} best own lag {lags[int(np.nanargmin(curve[c]))]:+d} frames; leave-one-out lag {loo[c]:+d}"
        )

    # The sync check found no consistent offset on the published clips (best lags sit at the edges of the
    # search range and disagree between clips), so no lag correction is applied in the results.
    variants = [("nadir", 0, False), ("tilt-aware", 0, False), ("tilt-aware", 0, True)]
    rows = []
    for clip in clips:
        w, h = sizes[clip]
        for model, lag, smooth in variants:
            lag_used = lag
            meta, res = evaluate(clip, model, w, h, lag=lag_used, smooth=smooth)
            label = model + (" + smoothed log" if smooth else "")
            for kind in ("swimmer", "boat"):
                if not res[kind]["speed"]:
                    continue
                rows.append(
                    {
                        "clip": meta["clip"],
                        "model": label,
                        "object": kind,
                        "tracks": len(res[kind]["speed"]),
                        "apparent_speed_median_mps": round(float(np.median(res[kind]["speed"])), 2),
                        "apparent_speed_max_mps": round(float(np.max(res[kind]["speed"])), 2),
                        "position_spread_median_m": round(float(np.median(res[kind]["spread"])), 2),
                        "drone_speed_max_mps": meta["drone_speed_mps_max"],
                        "altitude_m": "-".join(str(a) for a in meta["altitude_m"]),
                        "gimbal_pitch_deg": "-".join(str(p) for p in meta["gimbal_pitch_deg"]),
                        "hfov_deg_assumed": meta["hfov_deg_assumed"],
                    }
                )
    diag = []
    for c in clips:
        tel = [json.loads(x) for x in (c / "telemetry.jsonl").read_text().splitlines() if x.strip()]
        diag.append(
            {
                "clip": c.name,
                "logged_heading_change_deg": round(
                    (tel[-1]["heading_deg"] - tel[0]["heading_deg"] + 540) % 360 - 180, 1
                ),
                "image_rotation_deg": image_rotation(c),
                "gimbal_pitch_deg": tel[0]["pitch_deg"],
                "best_lag_frames_own_clip": lags[int(np.nanargmin(curve[c]))],
            }
        )
    with open(Path(args.out).with_name("geo_real_diagnostics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(diag[0]))
        wr.writeheader()
        wr.writerows(diag)
    for d_ in diag:
        print(d_)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
