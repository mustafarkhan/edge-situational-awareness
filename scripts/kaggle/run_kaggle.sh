#!/usr/bin/env bash
# One command for a GPU machine with internet (Kaggle notebook, rented GPU): download the three datasets, train the
# sea, thermal and land models properly, and pack the weights + training logs into one zip. Evaluation is NOT done
# here: the weights are scored on the held-out test sets with the same evaluation scripts as every other number in the README.
#
#   bash scripts/kaggle/run_kaggle.sh                 # everything (~6-8 h on a Kaggle T4)
#   STAGES="sea thermal" bash scripts/kaggle/run_kaggle.sh
#   QUICK=1 bash scripts/kaggle/run_kaggle.sh         # 1 epoch each, to test the setup (~30 min incl. downloads)
#
# Stages (order = priority) and the config each one trains, configs/train/gpu/<run>.yaml:
#   sea           sds_y11n_640      sea1280        sds_y11n_1280
#   sea1280_y11s  sds_y11s_1280     sea1280_y26n   sds_y26n_1280      (bigger model / YOLO26, same data and recipe)
#   thermal       hit_y11n_640      thermal1280    hit_y11n_1280      (thermal frames upscaled to 1280 px input)
#   thermal_y11s  hit_y11s_640      thermal_y26n   hit_y26n_640
#   land          vis_y11n_640
#
# Output: $OUT/esa_gpu_results.zip  (default OUT=/kaggle/working, which Kaggle keeps after the run)
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
OUT=${OUT:-/kaggle/working}
DATA=${DATA:-/tmp/esa_data}   # scratch disk, not kept (only $OUT is)
STAGES=${STAGES:-"sea thermal land sea1280"}
BUDGET_H=${BUDGET_H:-10.5}   # Kaggle stops a session at 12 h; leave time to pack the results
QUICK=${QUICK:-0}
mkdir -p "$OUT/esa_gpu" "$DATA" weights
LOG="$OUT/esa_gpu/run.log"
T0=$(date +%s)
log() { echo "[$(date -u +%H:%M:%S) +$(( ($(date +%s) - T0) / 60 ))m] $*" | tee -a "$LOG"; }
left_h() { python3 -c "print($BUDGET_H - ($(date +%s) - $T0) / 3600)"; }

NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
DEVICE=${DEVICE:-0}; [ "$NGPU" -ge 2 ] && DEVICE=${DEVICE_MULTI:-0,1}; [ "$NGPU" = 0 ] && DEVICE=cpu
log "GPUs: $NGPU ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd,)); device=$DEVICE; stages: $STAGES"
python3 -m pip install -q "ultralytics==8.4.174" lap pycocotools 2>&1 | tail -1
python3 -m pip install -q -e . 2>&1 | tail -1

weights() {  # the pretrained COCO weights a config starts from (its 'model:' line), from Ultralytics' releases
  local f; f=$(sed -n 's/^model: *//p' "$1")
  [ -s "$f" ] && return 0
  for rel in v8.4.0 v8.3.0; do
    curl -sSL --retry 5 -o "$f" "https://github.com/ultralytics/assets/releases/download/$rel/$(basename "$f")" \
      && [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -gt 1000000 ] && { log "weights $(basename "$f") ($rel)"; return 0; }
  done
  log "!! could not download $(basename "$f")"; rm -f "$f"; return 1
}

fetch() {  # url file: resumable, retried, and checked; a cut-off download must never look like data
  local url=$1 f=$2
  for attempt in 1 2 3 4 5; do
    curl -sSL --retry 8 --retry-all-errors --retry-delay 5 -C - -o "$f" "$url" || true
    if python3 -c "import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); sys.exit(0 if z.testzip() is None else 1)" "$f" 2>/dev/null; then
      log "downloaded $(basename "$f") ($(du -h "$f" | cut -f1))"; return 0
    fi
    log "download of $(basename "$f") incomplete (attempt $attempt), resuming"
    sleep 10
  done
  log "!! could not download $url"; return 1
}

has_images() { [ "$(ls "$1" 2>/dev/null | wc -l)" -gt 0 ]; }

EXTRA=("device=$DEVICE")
[ "$QUICK" = 1 ] && EXTRA+=(epochs=1 fraction=0.1 batch=8)
# shellcheck disable=SC2206
EXTRA+=(${EXTRA_ARGS:-})   # e.g. EXTRA_ARGS="batch=4 workers=0 cache=false" on a small machine

pack() {  # copy what exists so far: a session that dies still leaves results
  local name=$1 d=runs/detect/runs/gpu/$1
  [ -d "$d" ] || d=runs/gpu/$1
  [ -d "$d" ] || return 0
  mkdir -p "$OUT/esa_gpu/$name"
  cp -f "$d"/weights/best.pt "$OUT/esa_gpu/$name/" 2>/dev/null
  cp -f "$d"/results.csv "$d"/args.yaml "$d"/experiment_config.yaml "$OUT/esa_gpu/$name/" 2>/dev/null
  cp -f "$d"/*.png "$OUT/esa_gpu/$name/" 2>/dev/null
  (cd "$OUT" && rm -f esa_gpu_results.zip && python3 -m zipfile -c esa_gpu_results.zip esa_gpu)
  log "packed $name -> $OUT/esa_gpu_results.zip"
}

train() {  # name config
  local name=$1 cfg=$2
  if python3 -c "import sys; sys.exit(0 if $(left_h) > 0.75 else 1)"; then
    weights "$cfg" || return 0
    log "train $name ($cfg)"
    python3 training/train.py "$cfg" "${EXTRA[@]}" >> "$OUT/esa_gpu/train_$name.log" 2>&1 \
      || log "!! $name failed, see train_$name.log"
    pack "$name"
  else
    log "skip $name: less than 45 min of the time budget left"
  fi
}

prep_sea() {
  if [ ! -f configs/datasets/seadronessee.yaml ]; then
    log "download SeaDronesSee subset (2,900 images, ODv2 share, CC0)"
    python3 scripts/kaggle/fetch_sds.py --out "$DATA/sds" --roles train val 2>&1 | tee -a "$LOG" | tail -3
  fi
  [ -f configs/datasets/seadronessee.yaml ]
}

prep_thermal() {
  if [ ! -f configs/datasets/hituav.yaml ]; then
    log "download HIT-UAV (Zenodo 10.5281/zenodo.7633134, 815 MB, CC0)"
    if fetch "https://zenodo.org/api/records/7633134/files/suojiashun/HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip/content" \
      "$DATA/hit.zip"; then
      (cd "$DATA" && unzip -q -o hit.zip "*/normal_xml/JPEGImages/*" "*/normal_json/annotations/*" \
        && mv -- */normal_xml . && mv -- */normal_json . ) 2>&1 | tail -2
      (cd data && python3 convert_hituav.py --src "$DATA" --out "$DATA/hituav_yolo") 2>&1 | tee -a "$LOG" | tail -5
    fi
  fi
  has_images "$DATA/hituav_yolo/images/train"
}

prep_land() {
  if [ ! -f configs/datasets/visdrone.yaml ]; then
    log "download VisDrone2019-DET train + val (Ultralytics mirror, 1.6 GB)"
    for s in train val; do
      fetch "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-$s.zip" "$DATA/vd_$s.zip" \
        && (cd "$DATA" && unzip -q -o "vd_$s.zip")
    done
    (cd data && python3 convert_visdrone.py --src "$DATA" --out "$DATA/visdrone_yolo") 2>&1 | tee -a "$LOG" | tail -4
    # the two held-out demo sequences never go into training (same as the CPU run, docs/EXPERIMENT_LOG.md)
    rm -f "$DATA"/visdrone_yolo/images/train/9999951_* "$DATA"/visdrone_yolo/images/train/9999955_* \
          "$DATA"/visdrone_yolo/labels/train/9999951_* "$DATA"/visdrone_yolo/labels/train/9999955_*
    log "VisDrone train images after hold-out: $(ls "$DATA"/visdrone_yolo/images/train | wc -l)"
  fi
  if has_images "$DATA/visdrone_yolo/images/train" && has_images "$DATA/visdrone_yolo/images/val"; then return 0; fi
  rm -f configs/datasets/visdrone.yaml
  return 1
}

for stage in $STAGES; do
  case $stage in
    sea) data=sea; run=sds_y11n_640 ;;
    sea1280) data=sea; run=sds_y11n_1280 ;;
    sea1280_y11s) data=sea; run=sds_y11s_1280 ;;
    sea1280_y26n) data=sea; run=sds_y26n_1280 ;;
    thermal) data=thermal; run=hit_y11n_640 ;;
    thermal1280) data=thermal; run=hit_y11n_1280 ;;
    thermal_y11s) data=thermal; run=hit_y11s_640 ;;
    thermal_y26n) data=thermal; run=hit_y26n_640 ;;
    land) data=land; run=vis_y11n_640 ;;
    *) log "unknown stage $stage"; continue ;;
  esac
  if "prep_$data"; then train "$run" "configs/train/gpu/$run.yaml"
  else log "!! no $data data: skipping $stage"; fi
done
cp -f "$LOG" "$OUT/esa_gpu/" 2>/dev/null
(cd "$OUT" && rm -f esa_gpu_results.zip && python3 -m zipfile -c esa_gpu_results.zip esa_gpu)
log "DONE. Results: $OUT/esa_gpu_results.zip (score them with the repo's evaluation scripts, docs/KAGGLE.md)."
ls -la "$OUT/esa_gpu_results.zip" | tee -a "$LOG"
