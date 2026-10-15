"""Search drone footage by description: "person in orange life jacket", "red kayak", "overturned boat".

Uses an open-vocabulary detector (YOLOE, text prompts; no training for the new words) and writes the same JSONL as
tools/detect_video.py, so the results go straight to the map app and to TAK. Optional geolocation from a
telemetry log, exactly as in detect_video.py.

  python tools/search_by_text.py --source samples/sea_DJI_0055/frames --query "red boat" "person in water" \
      --telemetry samples/sea_DJI_0055/telemetry.jsonl --out outputs/text_search.jsonl

Accuracy is measured in README section "Text-prompt search": on swimmers it is far below the fine-tuned model,
so use this to look for things the trained model has no class for, not to replace it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from esa.geo import CameraGeo, pixel_to_latlon  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder of frames or a video file")
    ap.add_argument("--query", nargs="+", required=True, help="one or more descriptions")
    ap.add_argument("--model", default="yoloe-11s-seg.pt", help="downloaded by Ultralytics on first use")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--telemetry", default=None, help="JSONL with lat, lon, alt_m, hfov_deg, heading_deg, pitch_deg")
    ap.add_argument("--out", default=str(REPO / "outputs/text_search.jsonl"))
    a = ap.parse_args()

    from ultralytics import YOLOE

    model = YOLOE(a.model)
    model.set_classes(a.query, model.get_text_pe(a.query))
    tel = [json.loads(x) for x in Path(a.telemetry).read_text().splitlines()] if a.telemetry else None
    src = Path(a.source)
    frames = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    hits = 0
    with open(a.out, "w") as fh:
        for i, r in enumerate(model.predict([str(f) for f in frames] if src.is_dir() else str(src), imgsz=a.imgsz,
                                            conf=a.conf, stream=True, verbose=False)):  # fmt: skip
            t0 = time.perf_counter()
            h, w = r.orig_shape
            cam = None
            if tel and i < len(tel):
                t = tel[i]
                cam = CameraGeo(t["lat"], t["lon"], t["alt_m"], t["hfov_deg"], t.get("heading_deg", 0.0),
                                t.get("pitch_deg", 90.0))  # fmt: skip
            dets = []
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                pos = pixel_to_latlon((x1 + x2) / 2, (y1 + y2) / 2, w, h, cam) if cam else None
                dets.append(
                    {
                        "track_id": None,
                        "label": a.query[int(c)],
                        "conf": round(s, 3),
                        "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                        "lat": round(pos[0], 7) if pos else None,
                        "lon": round(pos[1], 7) if pos else None,
                    }
                )
            hits += len(dets)
            ms = sum(r.speed.values()) + (time.perf_counter() - t0) * 1000
            rec = {"frame": i, "t": time.time(), "fps": round(1000 / ms, 1), "query": a.query, "detections": dets}
            if cam:
                rec["drone"] = {"lat": cam.lat, "lon": cam.lon, "alt_m": cam.alt_m}
            fh.write(json.dumps(rec) + "\n")
    print(f"{len(frames)} frames, {hits} matches for {a.query} -> {a.out}")


if __name__ == "__main__":
    main()
