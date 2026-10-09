"""The most important script in the repo: what does speed cost in accuracy, and at what power?

  sudo nvpmodel -q            # note the power mode; keep it fixed for ALL runs
  sudo jetson_clocks          # lock clocks so results are repeatable
  python benchmarks/benchmark.py --model weights/best_640_fp16.engine \
      --data configs/datasets/seadronessee.yaml --imgsz 640 --tag y11s_640_fp16

Reports: mAP (Ultralytics val), latency mean/p50/p95/p99/max, FPS, average power,
energy per frame, max temperature. Appends a row to benchmarks/results/jetson.csv.
"""
import argparse
import csv
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from esa.power import TegraStats  # noqa: E402


def power_mode() -> str:
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
        return " ".join(out.split())
    except Exception:
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt, .onnx or .engine")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=500, help="images for the timing loop")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "jetson.csv"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    img_dir = Path(cfg["path"]) / cfg[args.split]
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.n]
    # load into RAM first so we time the model, not the SD card / SSD
    frames = [cv2.imread(str(p)) for p in paths]
    model = YOLO(args.model, task="detect")

    for f in frames[: args.warmup]:  # first runs include CUDA/TensorRT init; never time them
        model.predict(f, imgsz=args.imgsz, verbose=False)

    ts = TegraStats(interval_ms=100).start()
    time.sleep(0.5)
    lat, stages = [], []
    t_start = time.time()
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=args.imgsz, verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        stages.append(r.speed)  # {'preprocess','inference','postprocess'} in ms
    t_end = time.time()
    ts.stop()
    pw = ts.summary(t_start, t_end)

    lat = np.array(lat)
    row = {
        "tag": args.tag, "model": Path(args.model).name, "imgsz": args.imgsz,
        "device": platform.node(), "power_mode": power_mode(), "n_images": len(lat),
        "lat_mean_ms": round(lat.mean(), 2), "lat_p50_ms": round(np.percentile(lat, 50), 2),
        "lat_p95_ms": round(np.percentile(lat, 95), 2), "lat_p99_ms": round(np.percentile(lat, 99), 2),
        "lat_max_ms": round(lat.max(), 2), "fps": round(1000 / lat.mean(), 1),
        "pre_ms": round(np.mean([s["preprocess"] for s in stages]), 2),
        "infer_ms": round(np.mean([s["inference"] for s in stages]), 2),
        "post_ms": round(np.mean([s["postprocess"] for s in stages]), 2),
        "power_w": round(pw["power_w_mean"], 2) if pw else "",
        "energy_j_per_frame": round(pw["power_w_mean"] * lat.mean() / 1000, 3) if pw else "",
        "temp_c_max": round(pw["temp_c_max"], 1) if pw else "",
    }

    if not args.skip_accuracy:
        m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=1, plots=False, verbose=False)
        row["mAP50"] = round(float(m.box.map50), 4)
        row["mAP50_95"] = round(float(m.box.map), 4)

    out = Path(args.out)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("\n" + "\n".join(f"{k:>20}: {v}" for k, v in row.items()))
    if not pw:
        print("\n(no tegrastats found: power/temperature columns are empty -- run on the Jetson)")


if __name__ == "__main__":
    main()
