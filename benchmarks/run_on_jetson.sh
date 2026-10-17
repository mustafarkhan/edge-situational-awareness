#!/usr/bin/env bash
# Fill the Jetson table with ONE command. Run ON THE JETSON (see docs/JETSON.md for setup), from the repo root:
#
#   bash benchmarks/run_on_jetson.sh
#
# Needs no dataset: it uses the sea model in models/, 96 SeaDronesSee training images in samples/calib_sds for
# INT8 calibration (never the evaluation images), and the real clip in samples/sea_DJI_0055 for timing.
# Optional:
#   SDS=/path/to/seadronessee_yolo   also measure accuracy on the 1,200 test images (same metric as the README)
#   ROS=1                            also run the ROS 2 node end to end with the fastest engine
# Results: benchmarks/results/jetson*.csv and jetson_report.md. Send those back (or commit them).
set -euo pipefail
cd "$(dirname "$0")/.."
WEIGHTS=${WEIGHTS:-models/sds_y11n_1280_gpu.pt}
STEM=$(basename "$WEIGHTS" .pt)
OUT=benchmarks/results
REPORT=$OUT/jetson_report.md
mkdir -p outputs

{
  echo "# Jetson run $(date -u +%Y-%m-%dT%H:%MZ)"
  echo
  echo "- board: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
  echo "- L4T: $(head -n 1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
  echo "- power mode: $(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ' || echo unknown)"
  echo "- TensorRT: $(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || echo missing)"
  echo "- ultralytics: $(python3 -c 'import ultralytics; print(ultralytics.__version__)')"
} > "$REPORT"
cat "$REPORT"
sudo jetson_clocks || echo "jetson_clocks failed: results may vary run to run"

# calibration YAML with absolute paths (Ultralytics resolves relative paths against its own datasets folder)
CAL=outputs/calib_sds_abs.yaml
python3 - "$CAL" <<'EOF'
import sys, yaml, pathlib
c = yaml.safe_load(open("samples/calib_sds/calib.yaml"))
c["path"] = str(pathlib.Path("samples/calib_sds").resolve())
yaml.safe_dump(c, open(sys.argv[1], "w"), sort_keys=False)
EOF

# baseline: the same model in PyTorch on the Jetson GPU
for SZ in 640 1280; do
  python3 benchmarks/benchmark.py --model "$WEIGHTS" --images samples/sea_DJI_0055/frames --imgsz $SZ \
    --n 300 --tag jetson_pt_${SZ} --out $OUT/jetson.csv
done
for SZ in 640 1280; do
  python3 export/export_trt.py --weights "$WEIGHTS" --data "$CAL" --imgsz $SZ --precisions fp16 int8 --fraction 1.0
  for P in fp16 int8; do
    ENG=$(dirname "$WEIGHTS")/${STEM}_${SZ}_${P}.engine
    # speed, power, energy per frame, temperature (tegrastats)
    python3 benchmarks/benchmark.py --model "$ENG" --images samples/sea_DJI_0055/frames --imgsz $SZ \
      --n 300 --tag jetson_${SZ}_${P} --out $OUT/jetson.csv
    # whole pipeline (detect + track + geolocate + alerts) on the real clip
    python3 tools/detect_video.py --model "$ENG" --source samples/sea_DJI_0055/frames --imgsz $SZ --conf 0.1 \
      --telemetry samples/sea_DJI_0055/telemetry.jsonl --alert swimmer --alert-min-frames 15 \
      --out outputs/jetson_pipeline_${SZ}_${P}.jsonl
    # accuracy with the same metric as every other sea table, if the test set is on the Jetson
    if [ -n "${SDS:-}" ]; then
      python3 - "$SDS" <<'EOF'
import sys, yaml
yaml.safe_dump({"path": sys.argv[1], "train": "images/train", "val": "images/val", "test": "images/test",
                "names": {0: "swimmer", 1: "boat", 2: "jetski", 3: "life_saving_appliances", 4: "buoy"}},
               open("outputs/sds_jetson.yaml", "w"))
EOF
      python3 training/evaluate_sea.py --model "$ENG" --data outputs/sds_jetson.yaml --split test --imgsz $SZ \
        --view native --device 0 --tag jetson_${SZ}_${P} --out $OUT/jetson_sea_accuracy.csv
    fi
  done
done

# pipeline frame rate (median over the clip, after the first 5 warm-up frames)
python3 - >> "$REPORT" <<'EOF'
import glob, json, statistics
print("\n## Whole pipeline on the real clip (detect + track + geolocate + alerts)\n")
print("| engine | median FPS |\n|---|---|")
for f in sorted(glob.glob("outputs/jetson_pipeline_*.jsonl")):
    fps = [json.loads(l)["fps"] for l in open(f)][5:]
    print(f"| {f.split('jetson_pipeline_')[1][:-6]} | {statistics.median(fps):.1f} |")
EOF

if [ "${ROS:-0}" = 1 ]; then
  ENG=$(dirname "$WEIGHTS")/${STEM}_640_fp16.engine
  (cd ros2_ws && colcon build --symlink-install) && MODEL="$ENG" bash ros2_ws/test_node.sh | tee -a "$REPORT"
fi

python3 benchmarks/make_tables.py jetson >> "$REPORT" || true
echo
cat "$REPORT"
echo
echo "Done. Results: $OUT/jetson.csv, $REPORT$( [ -n "${SDS:-}" ] && echo ", $OUT/jetson_sea_accuracy.csv")"
