#!/usr/bin/env bash
# Fill the Jetson table with one command. Run ON THE JETSON, inside the jetson-containers
# PyTorch image (see docker/README.md), from the repo root:
#
#   WEIGHTS=weights/ft_y11n_ep3.pt bash benchmarks/run_on_jetson.sh
#
# Needs VisDrone val converted on the Jetson (data/convert_visdrone.py) and a few hundred VisDrone
# train images for INT8 calibration (Ultralytics calibrates on the dataset's val split by default;
# here the YAML is pointed at train for calibration so we never calibrate on what we evaluate).
set -euo pipefail
cd "$(dirname "$0")/.."
WEIGHTS=${WEIGHTS:-weights/ft_y11n_ep3.pt}
DATA=configs/datasets/visdrone.yaml
STEM=$(basename "$WEIGHTS" .pt)

echo "power mode: $(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ')"
sudo jetson_clocks || echo "jetson_clocks failed: results may vary run to run"

# calibration YAML: same dataset, but 'val' points at train images (calibration only)
python - <<'EOF'
import yaml
c = yaml.safe_load(open("configs/datasets/visdrone.yaml"))
c["val"] = "images/train"
yaml.safe_dump(c, open("configs/datasets/visdrone_calib.yaml", "w"), sort_keys=False)
EOF

for SZ in 640 1280; do
  python export/export_trt.py --weights "$WEIGHTS" --data configs/datasets/visdrone_calib.yaml \
    --imgsz $SZ --precisions fp16 int8 --fraction 0.05
  for P in fp16 int8; do
    ENG=$(dirname "$WEIGHTS")/${STEM}_${SZ}_${P}.engine
    # speed, power, energy, temperature
    python benchmarks/benchmark.py --model "$ENG" --data $DATA --split val --imgsz $SZ \
      --tag jetson_${SZ}_${P} --skip-accuracy
    # accuracy with the SAME metric as every other table (AP_small, three classes)
    python training/evaluate_zeroshot.py --model "$ENG" --class-map visdrone --data $DATA --imgsz $SZ \
      --device 0 --tag jetson_${SZ}_${P} --out benchmarks/results/jetson_accuracy.csv
  done
done
python benchmarks/make_tables.py jetson
echo "Commit benchmarks/results/jetson*.csv and paste the table into the README."
