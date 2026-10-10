"""How wrong do reported positions get when the drone's telemetry is wrong?

src/esa/geo.py assumes a straight-down camera and perfect telemetry. This script takes every
ground-truth box centre from a simulated flight, computes its position with the pipeline's own
pixel -> lat/lon code using PERTURBED telemetry, and measures the error in metres against the
true position. One error source at a time, so each row isolates one cause:

  gps          drone position off by N metres (random direction)
  altitude     altitude reading off by +N %       (scales every offset from the image centre)
  heading      compass off by N degrees            (rotates offsets around the image centre)
  gimbal tilt  camera pitched N degrees forward but treated as straight down; evaluated with a
               proper pinhole ray / flat-ground intersection, so this is the error of the
               nadir ASSUMPTION itself

  python benchmarks/geo_sensitivity.py --flight outputs/flight_slow
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from esa.geo import CameraGeo, pixel_to_latlon  # noqa: E402

EARTH = 111_320.0


def metres(lat1, lon1, lat2, lon2):
    return math.hypot((lat2 - lat1) * EARTH, (lon2 - lon1) * EARTH * math.cos(math.radians(lat1)))


def tilted_ground_point(px, py, w, h, cam: CameraGeo, pitch_deg: float):
    """True ground position of pixel (px, py) for a camera pitched forward by pitch_deg
    (towards image-top direction), heading 0, pinhole model, flat ground."""
    f = (w / 2) / math.tan(math.radians(cam.hfov_deg) / 2)
    # camera frame: x right, y up (image top), z along the optical axis (down when pitch = 0)
    ray = np.array([px - w / 2, h / 2 - py, f], float)
    t = math.radians(pitch_deg)
    # pitch forward: rotate optical axis from straight down towards north (image top)
    east = ray[0]
    north = ray[1] * math.cos(t) + ray[2] * math.sin(t)
    down = -ray[1] * math.sin(t) + ray[2] * math.cos(t)
    if down <= 0:
        return None  # ray above the horizon
    s = cam.alt_m / down
    e_m, n_m = east * s, north * s
    return (cam.lat + n_m / EARTH, cam.lon + e_m / (EARTH * math.cos(math.radians(cam.lat))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", required=True)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "geo_sensitivity.csv"))
    args = ap.parse_args()
    fl = Path(args.flight)
    telem = [json.loads(l) for l in (fl / "telemetry.jsonl").read_text().splitlines() if l.strip()]
    pts = []
    for line in (fl / "gt.txt").read_text().splitlines():
        f, _, x, y, w, h, *_ = line.split(",")
        pts.append((int(f) - 1, float(x) + float(w) / 2, float(y) + float(h) / 2))
    rng = np.random.default_rng(0)

    def cam_for(t, **kw):
        return CameraGeo(t["lat"], t["lon"], t["alt_m"], t["hfov_deg"], t["heading_deg"], **kw)

    def run(perturb):
        errs = []
        for f, px, py in pts:
            t = telem[f]
            true_cam = cam_for(t)
            truth = pixel_to_latlon(px, py, args.w, args.h, true_cam)
            est = perturb(t, px, py, truth)
            if est is not None:
                errs.append(metres(*truth, *est))
        return np.array(errs)

    rows = []

    def add(source, level, errs, note=""):
        rows.append(
            {
                "error_source": source,
                "level": level,
                "median_m": round(float(np.median(errs)), 2),
                "p95_m": round(float(np.percentile(errs, 95)), 2),
                "max_m": round(float(errs.max()), 2),
                "n_points": len(errs),
                "note": note,
            }
        )

    for d in (2, 5):

        def p(t, px, py, truth, d=d):
            ang = rng.uniform(0, 2 * math.pi)
            t2 = dict(
                t,
                lat=t["lat"] + d * math.cos(ang) / EARTH,
                lon=t["lon"] + d * math.sin(ang) / (EARTH * math.cos(math.radians(t["lat"]))),
            )
            return pixel_to_latlon(px, py, args.w, args.h, cam_for(t2))

        add("gps", f"{d} m", run(p), "shifts every position by the same amount")
    for pct in (5, 10):
        add(
            "altitude",
            f"+{pct} %",
            run(
                lambda t, px, py, tr, pct=pct: pixel_to_latlon(
                    px, py, args.w, args.h, cam_for(dict(t, alt_m=t["alt_m"] * (1 + pct / 100)))
                )
            ),
            "grows towards the image edges",
        )
    for deg in (2, 5):
        add(
            "heading",
            f"{deg} deg",
            run(
                lambda t, px, py, tr, deg=deg: pixel_to_latlon(
                    px, py, args.w, args.h, cam_for(dict(t, heading_deg=deg))
                )
            ),
            "grows towards the image edges",
        )
    for deg in (5, 10, 20):
        # truth = tilted camera; estimate = pipeline assuming nadir
        def p(t, px, py, truth, deg=deg):
            return pixel_to_latlon(px, py, args.w, args.h, cam_for(t))

        errs = []
        for f, px, py in pts:
            t = telem[f]
            tp = tilted_ground_point(px, py, args.w, args.h, cam_for(t), deg)
            if tp is not None:
                errs.append(metres(*tp, *pixel_to_latlon(px, py, args.w, args.h, cam_for(t))))
        add("gimbal tilt (ignored)", f"{deg} deg", np.array(errs), "nadir assumption vs true ray intersection")

    alt = [t["alt_m"] for t in telem]
    import csv

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"altitude {min(alt):.0f}-{max(alt):.0f} m, hfov {telem[0]['hfov_deg']} deg, {len(pts)} points")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
