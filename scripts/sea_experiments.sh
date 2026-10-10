#!/usr/bin/env bash
# Sea module (SeaDronesSee, CC0): every experiment in README section "At sea", in the order it was run.
#
# Data: SeaDronesSee ODv2 "Compressed Version" + SeaDronesSee-MOT, from the public Nextcloud shares linked
# at https://seadronessee.cs.uni-tuebingen.de/dataset. Only the subset in data/seadronessee_selection.json
# is used (2,500 train / 400 val / 1,200 test / 400 "seen" images), downscaled to 1920 px.
set -euo pipefail
cd "$(dirname "$0")/.."

SDS=${SDS:-/path/to/sds}            # instances_train.json, instances_val.json, img/{train,val,test,seen}
MOT_ANN=${MOT_ANN:-/path/to/instances_val_objects_in_water.json}
MOT_FRAMES=${MOT_FRAMES:-/path/to/mot}   # one folder per clip
D=configs/datasets/seadronessee.yaml
W=runs/detect/runs/sds/y11n_640_cpu/weights/last.pt

echo "== 0. subset selection (video-disjoint) and conversion"
python data/sds_select_subset.py --train-json $SDS/instances_train.json --val-json $SDS/instances_val.json \
  --out $SDS/selection.json
python data/convert_seadronessee_subset.py --images $SDS/img --train-json $SDS/instances_train.json \
  --val-json $SDS/instances_val.json --out datasets/seadronessee_yolo
python data/analyse_dataset.py --data $D --split test --imgsz 640 1280

echo "== 1. COCO-pretrained baseline (swimmer = COCO person, boat = COCO boat)"
for sz in 640 1280; do
  python training/evaluate_sea.py --model weights/yolo11n.pt --coco-model --view two_class --split test \
    --imgsz $sz --tag coco_y11n_$sz
done

echo "== 2. fine-tune and evaluate"
python training/train.py configs/train/sds_y11n_640_cpu.yaml
for sz in 640 1280; do
  python training/evaluate_sea.py --model $W --view native --split test --imgsz $sz --tag ft_y11n_$sz
  python training/evaluate_sea.py --model $W --view two_class --split test --imgsz $sz --tag ft_y11n_$sz
done
echo "-- leakage check: same model, official-val frames from TRAINING videos vs videos never seen"
python training/evaluate_sea.py --model $W --view native --split seen --imgsz 1280 --tag ft_y11n_1280_seen
python training/evaluate_sea.py --model $W --view native --split val --imgsz 1280 --tag ft_y11n_1280_val

echo "== 3. real footage: tracking + geolocation (SeaDronesSee-MOT, clips not used for training)"
python data/prepare_sds_mot.py --ann $MOT_ANN --frames $MOT_FRAMES \
  --clips DJI_0055 DJI_0059 DJI_0057 DJI_0001_d3 --out outputs/sea_mot
for c in DJI_0055 DJI_0059 DJI_0057 DJI_0001_d3; do
  for trk in bytetrack.yaml botsort.yaml; do
    n=$(basename $trk .yaml)
    python tools/detect_video.py --model $W --source outputs/sea_mot/$c/frames --imgsz 1280 --conf 0.1 \
      --tracker $trk --telemetry outputs/sea_mot/$c/telemetry.jsonl --alert swimmer --alert-min-frames 15 \
      --out outputs/sea_track_${c}_${n}.jsonl
    python benchmarks/tracking_eval.py --flight outputs/sea_mot/$c --run outputs/sea_track_${c}_${n}.jsonl \
      --tag sea_${c}_${n} --tracker $n --out benchmarks/results/tracking_sea.csv
  done
done
python benchmarks/geo_real.py --clips outputs/sea_mot/DJI_0055 outputs/sea_mot/DJI_0059 \
  outputs/sea_mot/DJI_0057 outputs/sea_mot/DJI_0001_d3
echo done
