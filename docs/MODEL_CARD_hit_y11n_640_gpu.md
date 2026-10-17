# Model Card: YOLO11n trained on HIT-UAV thermal on a GPU (hit_y11n_640_gpu)

**File:** [`models/hit_y11n_640_gpu.pt`](../models/hit_y11n_640_gpu.pt) (5.5 MB, PyTorch, Ultralytics 8.4.174;
SHA-256 in [`models/SHA256SUMS`](../models/SHA256SUMS)).
**Model:** YOLO11n, COCO-pretrained, fine-tuned at 640 px on 2× Tesla T4 (Kaggle), config
[`configs/train/gpu/hit_y11n_640.yaml`](../configs/train/gpu/hit_y11n_640.yaml): 120 epochs, cosine learning rate,
seed 0. The checkpoint is the epoch with the best validation mAP50-95 (epoch 112); the test set was never used for
training or selection.
**Input:** 8-bit thermal (long-wave infrared) frames, 640x512. **Classes:** person, car, bicycle, other_vehicle.
**Version / date:** 2026-10-16. **Licence:** AGPL-3.0 (Ultralytics). Training data: HIT-UAV, CC0 1.0.
Replaces [`hit_y11n_ep5`](MODEL_CARD_hit_y11n_ep5.md) (5 CPU epochs) as the system's thermal model.

## Intended use
Research and field trials of night-time detection of people and vehicles from drones at 60-130 m, with a human operator
reviewing every alert. **Not for operational search-and-rescue decisions without that review.**

## Training data
The same flight-group split as the CPU model (`data/convert_hituav.py`): 1,902 training images, 183 validation.

## Evaluation (477 test images from held-out groups; COCO AP, `training/evaluate_sea.py`)
Source [`benchmarks/results/hituav_shared.csv`](../benchmarks/results/hituav_shared.csv) (person / car / bicycle) and
[`hituav.csv`](../benchmarks/results/hituav.csv) (all four classes); [`scripts/gpu_scores.sh`](../scripts/gpu_scores.sh).

| model | subset | images | AP50 | AP | person | car | bicycle |
|---|---|---|---|---|---|---|---|
| hit_y11n_ep5 (5 CPU epochs) | all | 477 | 0.812 | 0.445 | 0.369 | 0.602 | 0.362 |
| **hit_y11n_640_gpu** | **all** | 477 | **0.846** | **0.512** | **0.397** | **0.659** | **0.479** |
| hit_y11n_640_gpu | day | 124 | 0.841 | 0.453 | 0.345 | 0.644 | 0.370 |
| hit_y11n_640_gpu | night | 353 | 0.842 | 0.525 | 0.413 | 0.670 | 0.494 |

With all four classes AP is 0.447 (the CPU model: 0.333); "other vehicle" (16 test objects) now scores 0.253 instead
of 0. CPU time: 90 ms per image (2 vCPU).

Leakage check: 0.584 on 336 "seen" frames from training groups against 0.512 on held-out groups (+0.072; the COCO
control gains +0.017 between the same sets, so about +0.055 is memorised scenery). Longer training memorised more than
the 5-epoch model did (+0.003).

## Limitations
One thermal camera model and region; flight groups are inferred from file names, so held-out groups may share
locations with training groups (README §13). Not tested on water: HIT-UAV is land scenes only. The land search plans
in README §10b were measured with the earlier 5-epoch model and have not yet been re-run with this one.
