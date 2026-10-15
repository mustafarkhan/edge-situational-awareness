#!/usr/bin/env bash
# v2: text-prompt (open-vocabulary) detection on SeaDronesSee, with NO sea training at all.
# Prompts are chosen on the val split (8 videos) and the chosen set is scored once on the test split
# (7 other videos), the same discipline as for the fine-tuned model. Results: README section "Text-prompt search".
set -euo pipefail
cd "$(dirname "$0")/.."
W=${WEIGHTS_DIR:-weights}
E="python training/evaluate_sea.py --imgsz 1280"

declare -A P=(
  [p1_literal]="swimmer=swimmer;boat=boat"
  [p2_descriptive]="swimmer=person in water;boat=boat"
  [p3_synonyms]="swimmer=swimmer|person in water|person swimming|head above water;boat=boat|motorboat|sailboat"
  [p4_coco_words]="swimmer=person;boat=boat"
)
echo "== 1. prompt choice on val (YOLOE-11s)"
for k in p1_literal p2_descriptive p3_synonyms p4_coco_words; do
  $E --model $W/yoloe-11s-seg.pt --view two_class --split val --prompts "${P[$k]}" --tag yoloe11s_${k}_val
done

echo "== 2. best val prompt set (by swimmer AP) -> test, once, small and medium model"
BEST=$(python - <<'PY'
import pandas as pd
d = pd.read_csv("benchmarks/results/seadronessee_prompts_two_class.csv")
d = d[d.split == "val"].drop_duplicates("tag", keep="last")
print(d.sort_values("AP_swimmer").iloc[-1].tag.replace("yoloe11s_", "").replace("_val", ""))
PY
)
echo "best on val: $BEST"
for m in 11s 11m; do
  $E --model $W/yoloe-$m-seg.pt --view two_class --split test --prompts "${P[$BEST]}" --tag yoloe${m}_${BEST}_test
done

echo "== 3. all five sea classes by text (classes COCO has no word for), test"
NATIVE="swimmer=swimmer|person in water;boat=boat;jetski=jet ski;life_saving_appliances=life buoy|life ring|life raft;buoy=buoy"
$E --model $W/yoloe-11s-seg.pt --view native --split test --prompts "$NATIVE" --tag yoloe11s_native_test
echo "== 4. land (VisDrone val, 548 images): pedestrian + car; one fixed wording, not tuned"
L="python training/evaluate_sea.py --data configs/datasets/visdrone.yaml --split val --imgsz 1280 --view shared"
L="$L --keep pedestrian,car --out benchmarks/results/visdrone_prompts.csv --name-filter . --subset all"
LAND="pedestrian=person|pedestrian|person walking;car=car"
$L --model $W/yolo11n.pt --coco-model --coco-map 0=pedestrian,2=car --tag coco_y11n_1280
$L --model $W/ft_y11n_ep3.pt --tag ft_y11n_1280
for m in 11s 11m; do $L --model $W/yoloe-$m-seg.pt --prompts "$LAND" --tag yoloe${m}_land; done
echo PROMPTS_DONE
