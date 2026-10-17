# Model Card: YOLO11n trained on SeaDronesSee at 1280 px on a GPU (sds_y11n_1280_gpu)

**File:** [`models/sds_y11n_1280_gpu.pt`](../models/sds_y11n_1280_gpu.pt) (5.5 MB, PyTorch, Ultralytics 8.4.174;
SHA-256 in [`models/SHA256SUMS`](../models/SHA256SUMS)).
**Model:** YOLO11n (2.6 M parameters), COCO-pretrained, fine-tuned at 1280 px on 2× Tesla T4 (Kaggle), config
[`configs/train/gpu/sds_y11n_1280.yaml`](../configs/train/gpu/sds_y11n_1280.yaml): 60 epochs, cosine learning rate,
seed 0. The checkpoint is the epoch with the best validation mAP50-95 (epoch 39); the test set was never on the
training machine. Inference size: 1280 px.
**Classes:** swimmer, boat, jetski, life_saving_appliances, buoy.
**Version / date:** 2026-10-16. **Licence:** AGPL-3.0 (Ultralytics). Training data: SeaDronesSee, CC0 1.0.
Replaces [`sds_y11n_ep4`](MODEL_CARD_sds_y11n_ep4.md) (4 CPU epochs at 640 px) as the system's sea model.

## Intended use
Research and field trials of small-object detection in maritime drone imagery, with a human operator reviewing every
alert. **Not for operational search-and-rescue decisions without that review.**

## Training data
The same video-disjoint subset of SeaDronesSee Object Detection v2 as the CPU model
(`data/seadronessee_selection.json`): 2,500 training images from 24 videos, 400 validation images from 8 other videos.
Test: 1,200 images from 7 further videos.

## Evaluation (1,200 test images from 7 unseen videos, 6,825 objects)
COCO AP at IoU 0.50:0.95 (`training/evaluate_sea.py`, [`scripts/gpu_scores.sh`](../scripts/gpu_scores.sh)); source
[`benchmarks/results/seadronessee.csv`](../benchmarks/results/seadronessee.csv). CPU time is the median per image on
a 2-vCPU cloud VM.

| model | input | AP50 | AP | AP small | swimmer | boat | jetski | life-saving app. | buoy | CPU ms |
|---|---|---|---|---|---|---|---|---|---|---|
| sds_y11n_ep4 (4 CPU epochs, 640 px) | 1280 | 0.488 | 0.266 | 0.207 | 0.185 | 0.503 | 0.143 | 0.000 | 0.500 | 259 |
| **sds_y11n_1280_gpu** | **1280** | **0.583** | **0.325** | **0.277** | **0.222** | 0.562 | **0.242** | 0.019 | **0.579** | 260 |

Training at 640 px for 100 epochs instead (`sds_y11n_640_gpu`, not shipped) gives AP 0.296 at 1280 px input: the
longer training is worth about +0.03 AP, training at the size the model will run at another +0.03.

Leakage check: on 400 official-val frames from *training* videos this model scores swimmer AP 0.371 against 0.222 on
unseen videos (+0.149; the 4-epoch model: +0.110; a COCO model, which saw neither: +0.018). Longer training memorises
more of the training videos' scenery, so the official split would overstate this model by even more.

## Performance on device
Jetson Orin Nano: not yet measured (`benchmarks/run_on_jetson.sh`).

## Limitations and risks
Life-saving appliances (life rings, vests) are still almost never detected (AP 0.019): it is the smallest class in the
images. Most swimmers are under 16 px at 1280 px input; many are missed or loosely boxed (swimmer AP 0.222). Trained
on summer footage from a few lakes and the North Sea, mostly DJI cameras; rough sea, glare, night and thermal imagery
are untested. The search-planning curves in README §10a were measured with the earlier 4-epoch model and have not yet
been re-run with this one (that needs the SeaDronesSee per-image flight metadata again). See `docs/LIMITATIONS.md`.
