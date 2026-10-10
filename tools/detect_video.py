"""Run detection + tracking on a video or camera WITHOUT ROS, and write one JSON line per
frame. Works on a laptop too. The web backend can replay the file onto the live map.

  python tools/detect_video.py --model best.pt --source clip.mp4 --out outputs/run1.jsonl \
      --lat 60.15 --lon 24.95 --alt 60 --hfov 70 --alert swimmer --save-video outputs/run1.mp4

--source 0 uses the first USB camera. On the Jetson, pass the .engine file as --model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esa.alerts import AlertManager  # noqa: E402
from esa.geo import CameraGeo, pixel_to_latlon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="bytetrack.yaml", help="or botsort.yaml")
    ap.add_argument("--out", default="outputs/detections.jsonl")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--lat", type=float, default=60.15)
    ap.add_argument("--lon", type=float, default=24.95)
    ap.add_argument("--alt", type=float, default=60.0, help="metres above sea/ground")
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--heading", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=90.0, help="camera tilt below the horizon, 90 = straight down")
    ap.add_argument(
        "--telemetry",
        default=None,
        help="JSONL, one line per frame with lat, lon, alt_m, hfov_deg, heading_deg "
        "(e.g. from a flight log or tools/sim_flight.py). Overrides the fixed values above.",
    )
    ap.add_argument("--alert", nargs="*", default=["swimmer"], help="labels that raise alerts")
    ap.add_argument("--alert-min-frames", type=int, default=5)
    ap.add_argument(
        "--no-track",
        action="store_true",
        help="detection only. Use for frames that are not continuous video: ByteTrack only "
        "reports confirmed tracks, so on sparse frames it silently drops most detections",
    )
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model, task="detect")
    cam = CameraGeo(args.lat, args.lon, args.alt, args.hfov, args.heading, args.pitch)
    alerts = AlertManager(set(args.alert), args.alert_min_frames)
    telem = (
        [json.loads(l) for l in Path(args.telemetry).read_text().splitlines() if l.strip()] if args.telemetry else None
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = None

    with open(args.out, "w") as fh:
        t_prev = time.perf_counter()
        common = dict(stream=True, imgsz=args.imgsz, conf=args.conf, max_det=500, verbose=False)
        results = (
            model.predict(source, **common)
            if args.no_track
            else model.track(source, persist=True, tracker=args.tracker, **common)
        )
        for i, r in enumerate(results):
            now = time.perf_counter()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            h, w = r.orig_shape
            if telem:
                t = telem[min(i, len(telem) - 1)]
                cam = CameraGeo(
                    t["lat"],
                    t["lon"],
                    t["alt_m"],
                    t.get("hfov_deg", args.hfov),
                    t.get("heading_deg", 0.0),
                    t.get("pitch_deg", 90.0),
                )
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            dets = []
            for (x1, y1, x2, y2), c, s, tid in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), ids
            ):
                pos = pixel_to_latlon((x1 + x2) / 2, (y1 + y2) / 2, w, h, cam)  # None above the horizon
                dets.append(
                    {
                        "track_id": tid,
                        "cls": c,
                        "label": r.names[c],
                        "conf": round(s, 3),
                        "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                        "lat": round(pos[0], 7) if pos else None,
                        "lon": round(pos[1], 7) if pos else None,
                    }
                )
            new_alerts = alerts.update(dets)
            fh.write(
                json.dumps(
                    {
                        "frame": i,
                        "t": time.time(),
                        "fps": round(fps, 1),
                        "drone": {"lat": cam.lat, "lon": cam.lon},
                        "detections": dets,
                        "alerts": new_alerts,
                    }
                )
                + "\n"
            )
            for a in new_alerts:
                print("ALERT:", a["message"])
            if args.save_video:
                frame = r.plot()
                if writer is None:
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                writer.write(frame)
            if i % 50 == 0:
                print(f"frame {i}: {len(dets)} objects, {fps:.1f} FPS")
    if writer:
        writer.release()
    print("done ->", args.out)


if __name__ == "__main__":
    main()
