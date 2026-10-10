"""Controlled tracking test: fly a simulated camera over one real, labelled drone image.

Why: continuous drone video with per-object identities was not available in the build
environment. A camera window that moves (and zooms) over a still, fully-labelled frame gives a
test where the truth is known exactly:
  * every labelled object has one fixed identity  -> ID switches can be counted exactly
  * every object has one fixed ground position    -> geolocation error can be measured in metres
  * camera motion speed is a dial                 -> we can test trackers under slow and fast motion

What it does NOT test: objects moving on their own, real camera geometry (the source frame is
treated as a straight-down orthophoto), lighting change, motion blur, real telemetry noise.

Outputs (in --out):
  frames/000000.jpg ...   camera frames (--out-w x --out-h)
  gt.txt                  MOTChallenge-style: frame,id,x,y,w,h,1,class,visibility (frames 1-based)
  telemetry.jsonl         per frame: drone lat/lon, altitude, horizontal FOV, heading
  objects.json            per object id: class and true lat/lon of its box centre

  python tools/sim_flight.py --image datasets/visdrone_yolo/images/val/0000129_02411_d_0000138.jpg \
      --speed 6 --frames 120 --out outputs/flight_slow
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import yaml
from PIL import Image

EARTH = 111_320.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", default=None, help="YOLO label file (default: matching labels/ path)")
    ap.add_argument("--data", default="configs/datasets/visdrone.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--speed", type=float, default=6.0, help="camera ground-track speed, source px per frame")
    ap.add_argument("--out-w", type=int, default=960)
    ap.add_argument("--out-h", type=int, default=540)
    ap.add_argument("--crop-w", type=float, default=880, help="mean crop width in source px")
    ap.add_argument("--zoom-amp", type=float, default=80, help="crop width varies +/- this (altitude change)")
    ap.add_argument("--gsd", type=float, default=0.10, help="metres per source pixel (assumed)")
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--lat0", type=float, default=60.1500)
    ap.add_argument("--lon0", type=float, default=24.9500)
    ap.add_argument("--min-vis", type=float, default=0.5, help="keep a GT box if this fraction is in view")
    args = ap.parse_args()

    img_path = Path(args.image)
    lbl_path = Path(args.labels) if args.labels else Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    names = yaml.safe_load(Path(args.data).read_text())["names"]
    src = Image.open(img_path).convert("RGB")
    W, H = src.size
    aspect = args.out_h / args.out_w

    def px_to_latlon(x, y):
        north = (H / 2 - y) * args.gsd
        east = (x - W / 2) * args.gsd
        return (args.lat0 + north / EARTH,
                args.lon0 + east / (EARTH * math.cos(math.radians(args.lat0))))

    objs = []
    for i, line in enumerate(l for l in lbl_path.read_text().splitlines() if l.strip()):
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
        lat, lon = px_to_latlon(cx, cy)
        objs.append({"id": i + 1, "cls": int(c), "label": names[int(c)],
                     "box": (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), "lat": lat, "lon": lon})

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "frames").mkdir(parents=True)

    # elliptical ground track that keeps the largest crop inside the image
    max_cw = args.crop_w + args.zoom_amp
    ax, ay = (W - max_cw) / 2 * 0.95, (H - max_cw * aspect) / 2 * 0.95
    r_mean = math.sqrt((ax ** 2 + ay ** 2) / 2)
    theta = 0.0
    gt_lines, telem = [], []
    for f in range(args.frames):
        cx, cy = W / 2 + ax * math.cos(theta), H / 2 + ay * math.sin(theta)
        theta += args.speed / r_mean
        cw = args.crop_w + args.zoom_amp * math.sin(2 * math.pi * f / max(args.frames, 1))
        ch = cw * aspect
        x0, y0 = cx - cw / 2, cy - ch / 2
        frame = src.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize(
            (args.out_w, args.out_h), Image.BILINEAR)
        frame.save(out / "frames" / f"{f:06d}.jpg", quality=92)
        s = args.out_w / cw
        for o in objs:
            bx1, by1, bx2, by2 = o["box"]
            area = (bx2 - bx1) * (by2 - by1)
            ix1, iy1 = max(bx1, x0), max(by1, y0)
            ix2, iy2 = min(bx2, x0 + cw), min(by2, y0 + ch)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            vis = (ix2 - ix1) * (iy2 - iy1) / area
            if vis < args.min_vis:
                continue
            fx, fy = (ix1 - x0) * s, (iy1 - y0) * s
            fw, fh = (ix2 - ix1) * s, (iy2 - iy1) * s
            gt_lines.append(f"{f + 1},{o['id']},{fx:.1f},{fy:.1f},{fw:.1f},{fh:.1f},1,{o['cls']},{vis:.2f}")
        lat, lon = px_to_latlon(cx, cy)
        alt = args.gsd * cw / (2 * math.tan(math.radians(args.hfov) / 2))
        telem.append({"frame": f, "lat": lat, "lon": lon, "alt_m": round(alt, 2),
                      "hfov_deg": args.hfov, "heading_deg": 0.0})

    (out / "gt.txt").write_text("\n".join(gt_lines) + "\n")
    (out / "telemetry.jsonl").write_text("\n".join(json.dumps(t) for t in telem) + "\n")
    (out / "objects.json").write_text(json.dumps(
        {o["id"]: {"label": o["label"], "lat": o["lat"], "lon": o["lon"]} for o in objs}))
    meta = {"source_image": str(img_path), "frames": args.frames, "speed_px_per_frame": args.speed,
            "speed_m_per_frame": args.speed * args.gsd, "gsd_m_per_px": args.gsd,
            "altitude_m_range": [round(min(t["alt_m"] for t in telem), 1), round(max(t["alt_m"] for t in telem), 1)],
            "n_objects": len(objs), "gt_boxes": len(gt_lines)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
