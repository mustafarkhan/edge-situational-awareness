# Model Card: YOLO11n fine-tuned on HIT-UAV thermal (hit_y11n_ep5)

**File:** [`models/hit_y11n_ep5.pt`](../models/hit_y11n_ep5.pt) (5.4 MB, PyTorch, Ultralytics 8.4.174).
**Model:** YOLO11n, COCO-pretrained, fine-tuned 5 epochs at 640 px on CPU (`configs/train/hit_y11n_640_cpu.yaml`).
**Input:** 8-bit thermal (long-wave infrared) frames, 640x512. **Classes:** person, car, bicycle, other_vehicle.
**Version / date:** 2026-10-15. **Licence:** AGPL-3.0 (Ultralytics). Training data: HIT-UAV, CC0 1.0.

## Intended use
Research and demonstration of night-time detection of people and vehicles from drones at 60-130 m.
**Not for operational search-and-rescue or any safety-relevant decision.**

## Training data
HIT-UAV regrouped by flight condition (day/night x altitude x camera angle) so whole groups are held out
(`data/convert_hituav.py`, `data/hituav_selection.json`): 1,902 training images. The official random split was not used:
it puts frames 0.5 s apart into train and test.

## Evaluation (477 test images from held-out groups, 3,334 objects; COCO AP, `training/evaluate_sea.py`)

| subset | images | AP50 | AP | person | car | bicycle |
|---|---|---|---|---|---|---|
| all | 477 | 0.812 | 0.445 | 0.369 | 0.602 | 0.362 |
| day | 124 | 0.695 | 0.338 | 0.277 | 0.551 | 0.186 |
| night | 353 | 0.831 | 0.471 | 0.407 | 0.622 | 0.385 |

(person / car / bicycle view; with all four classes AP is 0.333 because "other vehicle" has only 16 test objects and scores 0.)
CPU time: 88 ms per image (2 vCPU). A COCO-pretrained RGB model scores AP 0.023 on the same images.

## Limitations
Five CPU epochs; one thermal camera model and region; flight groups are inferred from file names, so held-out groups may
share locations with training groups (see README section 13). Not tested on water: HIT-UAV is land scenes only.
