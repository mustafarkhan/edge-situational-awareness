#!/usr/bin/env bash
# Score the GPU-trained models (docs/KAGGLE.md) on the same held-out test sets, with the same scripts, as the CPU
# models, so every row is comparable. The test images were never on Kaggle: the notebook downloads only train/val.
#
#   GPU=/path/to/esa_gpu bash scripts/gpu_scores.sh      # the unzipped esa_gpu_results.zip
#   MOT=outputs/sea_mot GPU=... bash scripts/gpu_scores.sh  # also tracking and alerts on the real sea clips
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:?set GPU to the unzipped esa_gpu folder}
E="python training/evaluate_sea.py"
M=$(mktemp -d)  # name each best.pt after its run, so the results tables say which model a row is
for r in sds_y11n_640 sds_y11n_1280 hit_y11n_640; do ln -s "$(realpath "$GPU/$r/best.pt")" "$M/${r}_gpu.pt"; done

echo "== sea (SeaDronesSee, 1,200 test images from 7 videos never trained on)"
S640=$M/sds_y11n_640_gpu.pt
S1280=$M/sds_y11n_1280_gpu.pt
for sz in 640 1280; do
  $E --model "$S640" --view native --split test --imgsz $sz --tag gpu_y11n_640_at$sz
  $E --model "$S640" --view two_class --split test --imgsz $sz --tag gpu_y11n_640_at$sz
done
$E --model "$S1280" --view native --split test --imgsz 1280 --tag gpu_y11n_1280
$E --model "$S1280" --view two_class --split test --imgsz 1280 --tag gpu_y11n_1280
echo "-- leakage check: frames from training videos vs videos never seen"
$E --model "$S1280" --view native --split seen --imgsz 1280 --tag gpu_y11n_1280_seen

echo "== thermal (HIT-UAV, 477 test images from flight groups never trained on)"
H=$M/hit_y11n_640_gpu.pt
D=configs/datasets/hituav.yaml
T="$E --data $D --out benchmarks/results/hituav.csv --imgsz 640"
SHARED=(--view shared --keep "person,car,bicycle")
$T --model "$H" --split test --name-filter "." --subset all --tag gpu_y11n_640_test_native --view native
$T --model "$H" --split test --name-filter "." --subset all --tag gpu_y11n_640_test "${SHARED[@]}"
$T --model "$H" --split test --name-filter "^0_" --subset day --tag gpu_y11n_640_test_day "${SHARED[@]}"
$T --model "$H" --split test --name-filter "^1_" --subset night --tag gpu_y11n_640_test_night "${SHARED[@]}"
$T --model "$H" --split seen --name-filter "." --subset all --tag gpu_y11n_640_seen "${SHARED[@]}"

if [ -n "${MOT:-}" ]; then  # prepared SeaDronesSee-MOT clips (scripts/sea_experiments.sh, step 3)
  echo "== tracking and alerts on 4 real sea clips, shipped GPU model"
  RUNS=${RUNS:-outputs/gpu_track}
  mkdir -p "$RUNS"
  for c in DJI_0055 DJI_0059 DJI_0057 DJI_0001_d3; do
    for n in bytetrack botsort; do
      python tools/detect_video.py --model models/sds_y11n_1280_gpu.pt --source "$MOT/$c/frames" --imgsz 1280 \
        --conf 0.1 --tracker $n.yaml --telemetry "$MOT/$c/telemetry.jsonl" --alert swimmer --alert-min-frames 15 \
        --out "$RUNS/sea_track_${c}_${n}.jsonl"
      python benchmarks/tracking_eval.py --flight "$MOT/$c" --run "$RUNS/sea_track_${c}_${n}.jsonl" \
        --tag "sea_${c}_${n}_gpu" --tracker $n --out benchmarks/results/tracking_sea.csv
    done
  done
  python benchmarks/alert_load.py --clips "$MOT"/DJI_0055 "$MOT"/DJI_0059 "$MOT"/DJI_0057 "$MOT"/DJI_0001_d3 \
    --runs "$RUNS" --out benchmarks/results/alert_load_gpu.csv
fi
echo GPU_SCORES_DONE
