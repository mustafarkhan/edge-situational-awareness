"""Prepare real SeaDronesSee-MOT clips for tracking and geolocation evaluation.

Produces, per clip, the same layout as tools/sim_flight.py, so benchmarks/tracking_eval.py works unchanged:
  frames/<id>.jpg        frames (as downloaded; may be downscaled)
  gt.txt                 MOTChallenge-style frame,id,x,y,w,h,1,class,1 (boxes scaled to the frame size)
  telemetry.jsonl        per frame: drone lat/lon, altitude, camera heading and pitch from the drone's own log
  meta.json              clip facts (frames, fps, drone speed, altitude, pitch)
  objects.json           {} (no ground-truth object positions exist for real footage)

Camera field of view is not in the metadata; --hfov is an assumption (DJI Mavic 2 Pro, 28 mm-equivalent
lens: about 65.5 deg horizontally) and is stated wherever results are reported.

  python data/prepare_sds_mot.py --ann instances_val_objects_in_water.json --frames /path/mot \
      --clips DJI_0055 DJI_0059 DJI_0057 DJI_0001_d3 --out outputs/sea_mot
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--frames", required=True, help="folder with one sub-folder of frames per clip")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/sea_mot")
    ap.add_argument("--hfov", type=float, default=65.5)
    args = ap.parse_args()

    d = json.loads(Path(args.ann).read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_video, anns = defaultdict(list), defaultdict(list)
    for im in d["images"]:
        by_video[im["source"]["folder_name"], im["video_id"]].append(im)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)

    for clip in args.clips:
        cand = [k for k in by_video if k[0] == clip and (Path(args.frames) / clip).exists()]
        frames_dir = Path(args.frames) / clip
        have = {int(p.stem) for p in frames_dir.glob("*.jpg")}
        key = max(cand, key=lambda k: len({int(i["file_name"][:-4]) for i in by_video[k]} & have))
        ims = sorted((i for i in by_video[key] if int(i["file_name"][:-4]) in have), key=lambda i: i["frame_index"])
        out = Path(args.out) / clip
        (out / "frames").mkdir(parents=True, exist_ok=True)
        gt, tel, times, speeds, cls_count = [], [], [], [], Counter()
        for f, im in enumerate(ims):
            src = frames_dir / f"{int(im['file_name'][:-4]):06d}.jpg"
            dst = out / "frames" / f"{f:06d}.jpg"
            if not dst.exists():
                dst.symlink_to(src.resolve())
            w1, _ = Image.open(src).size
            s = w1 / im["width"]
            for a in anns[im["id"]]:
                x, y, w, h = (v * s for v in a["bbox"])
                gt.append(f"{f + 1},{a['track_id']},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,{a['category_id']},1")
                cls_count[cats[a["category_id"]]] += 1
            m = im["meta"]
            tel.append(
                {
                    "frame": f,
                    "lat": m["gps_latitude"],
                    "lon": m["gps_longitude"],
                    "alt_m": m["altitude"],
                    "hfov_deg": args.hfov,
                    "heading_deg": m.get("gimbal_heading", m.get("compass_heading", 0.0)),
                    "pitch_deg": m["gimbal_pitch"],
                    "speed_mps": m.get("speed"),
                    "time": im["date_time"],
                }
            )
            times.append(datetime.fromisoformat(im["date_time"]))
            speeds.append(m.get("speed") or 0.0)
        dt = (times[-1] - times[0]).total_seconds() / max(len(times) - 1, 1)
        fps = 1 / dt if dt > 0 else 30.0
        (out / "gt.txt").write_text("\n".join(gt) + "\n")
        (out / "telemetry.jsonl").write_text("\n".join(json.dumps(t) for t in tel) + "\n")
        (out / "objects.json").write_text("{}")
        meta = {
            "clip": clip,
            "video_id": key[1],
            "frames": len(ims),
            "fps": round(fps, 2),
            "speed_m_per_frame": round(sum(speeds) / len(speeds) / fps, 3),
            "drone_speed_mps_max": round(max(speeds), 1),
            "altitude_m": [round(min(t["alt_m"] for t in tel), 1), round(max(t["alt_m"] for t in tel), 1)],
            "gimbal_pitch_deg": [min(t["pitch_deg"] for t in tel), max(t["pitch_deg"] for t in tel)],
            "hfov_deg_assumed": args.hfov,
            "tracks": len({g.split(",")[1] for g in gt}),
            "boxes": dict(cls_count),
            "categories": cats,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta))


if __name__ == "__main__":
    main()
