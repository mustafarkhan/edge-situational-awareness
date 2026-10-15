#!/usr/bin/env bash
# v2: night / thermal detection on HIT-UAV (CC0). Every experiment in README section "At night (thermal)".
set -euo pipefail
cd "$(dirname "$0")/.."
HIT=${HIT:-/path/to/HIT-UAV}   # extracted archive: normal_xml/JPEGImages, normal_json/annotations
D=configs/datasets/hituav.yaml
OUT=benchmarks/results/hituav.csv
E="python training/evaluate_sea.py --data $D --out $OUT"
SHARED="--view shared --keep person,car,bicycle"

if [ "${SKIP_PREP:-0}" != 1 ]; then
  echo "== 0. convert with a flight-disjoint split; data check"
  (cd data && python convert_hituav.py --src "$HIT")
  python data/analyse_dataset.py --data $D --split test --imgsz 640
fi

score() {  # model tag extra-args...: all / day / night on test, and the leakage set
  local m=$1 tag=$2; shift 2
  $E --model "$m" --split test --name-filter "." --subset all --tag ${tag}_test "$@"
  $E --model "$m" --split test --name-filter "^0_" --subset day --tag ${tag}_test_day "$@"
  $E --model "$m" --split test --name-filter "^1_" --subset night --tag ${tag}_test_night "$@"
  $E --model "$m" --split seen --name-filter "." --subset all --tag ${tag}_seen "$@"
}

echo "== 1. COCO-pretrained (RGB) model on thermal images: person, car, bicycle"
if [ "${SKIP_COCO:-0}" != 1 ]; then
  score weights/yolo11n.pt coco_y11n_640 --imgsz 640 $SHARED --coco-model --coco-map 0=person,2=car,1=bicycle
fi

echo "== 2. fine-tune on thermal (CPU) and score the same way"
if [ "${SKIP_TRAIN:-0}" != 1 ]; then python training/train.py configs/train/hit_y11n_640_cpu.yaml; fi
W=runs/detect/runs/hit/y11n_640_cpu/weights/last.pt
score $W ft_y11n_640 --imgsz 640 $SHARED
$E --model $W --split test --name-filter "." --subset all --tag ft_y11n_640_test_native --imgsz 640 --view native

echo "== 3. detection vs altitude (60-130 m), test set, fine-tuned"
for alt in 60 70 80 90 100 110 120 130; do
  $E --model $W --split test --name-filter "^[01]_${alt}_" --subset alt_${alt} --tag ft_y11n_640_test_alt${alt} \
    --imgsz 640 $SHARED || true
done
echo THERMAL_DONE
