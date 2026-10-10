#!/usr/bin/env bash
# Phase 2 experiments, in the order they were run for the published results.
# Needs: the fine-tuned weights (WEIGHTS), VisDrone val converted (configs/datasets/visdrone.yaml),
# raw VisDrone val annotations (RAW_ANN), the built frontend (for the map recording).
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS=${WEIGHTS:-weights/ft_y11n_ep3.pt}
RAW_ANN=${RAW_ANN:-/path/to/VisDrone2019-DET-val/annotations}
DATA=configs/datasets/visdrone.yaml
SRC_IMG=datasets/visdrone_yolo/images/val/0000129_02411_d_0000138.jpg
TAG=$(basename "$WEIGHTS" .pt)

echo "== 1. accuracy of the fine-tuned model (640 and 1280 input)"
python training/evaluate_zeroshot.py --model "$WEIGHTS" --class-map visdrone --data $DATA --imgsz 640  --tag ft_y11n_640
python training/evaluate_zeroshot.py --model "$WEIGHTS" --class-map visdrone --data $DATA --imgsz 1280 --tag ft_y11n_1280

echo "== 2. controlled tracking test: 2 camera speeds x 3 trackers"
python tools/sim_flight.py --image $SRC_IMG --speed 6  --frames 120 --out outputs/flight_slow
python tools/sim_flight.py --image $SRC_IMG --speed 24 --frames 120 --out outputs/flight_fast
for speed in slow fast; do
  for trk in bytetrack.yaml configs/trackers/botsort_nogmc.yaml botsort.yaml; do
    name=$(basename "$trk" .yaml)
    python tools/detect_video.py --model "$WEIGHTS" --source outputs/flight_$speed/frames --imgsz 1280 --conf 0.1 \
      --tracker "$trk" --telemetry outputs/flight_$speed/telemetry.jsonl \
      --alert pedestrian people --alert-min-frames 10 --out outputs/track_${speed}_${name}.jsonl
    python benchmarks/tracking_eval.py --flight outputs/flight_$speed --run outputs/track_${speed}_${name}.jsonl \
      --tag ${speed}_${name} --tracker "$name"
  done
done
python benchmarks/geo_sensitivity.py --flight outputs/flight_slow

echo "== 3. INT8 quantisation (ONNX Runtime, CPU)"
python export/quantize_onnx.py --weights "$WEIGHTS" --imgsz 1280 \
  --calib-dir datasets/visdrone_yolo/images/train --n-calib 128
for v in fp32 int8_full int8_headfp32; do
  python training/evaluate_zeroshot.py --model "weights/${TAG}_1280_${v}.onnx" --class-map visdrone \
    --data $DATA --imgsz 1280 --tag onnx_${v}_1280 --out benchmarks/results/quantization.csv
done

echo "== 4. failure analysis"
python data/failure_analysis.py --model "$WEIGHTS" --imgsz 1280 \
  --images datasets/visdrone_yolo/images/val --raw-ann "$RAW_ANN"

echo "== 5. map recording + demo video"
python tools/record_map.py --replay outputs/track_slow_botsort.jsonl --seconds 14 --center 60.15,24.95 --zoom 18 --out outputs/map_recording.mp4
echo "done"
