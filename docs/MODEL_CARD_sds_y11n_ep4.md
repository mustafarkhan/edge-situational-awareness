# Model Card: YOLO11n fine-tuned on SeaDronesSee (sds_y11n_ep4)

**File:** [`models/sds_y11n_ep4.pt`](../models/sds_y11n_ep4.pt) (5.4 MB, PyTorch, Ultralytics 8.4.174).
**Model:** YOLO11n (2.6 M parameters), COCO-pretrained, fine-tuned 4 epochs at 640 px on CPU
(config `configs/train/sds_y11n_640_cpu.yaml`). Recommended inference size: 1280 px.
**Classes:** swimmer, boat, jetski, life_saving_appliances, buoy.
**Version / date:** 2026-10-11. **Licence:** AGPL-3.0 (Ultralytics). Training data: SeaDronesSee, CC0 1.0.

## Intended use
Research and demonstration of small-object detection in maritime drone imagery on embedded hardware.
**Not for operational search-and-rescue or any safety-relevant decision.**

## Training data
Video-disjoint subset of SeaDronesSee Object Detection v2 (`data/seadronessee_selection.json`): 2,500 images from
24 videos, downscaled to 1920 px. Validation: 400 images from 8 other videos. Test: 1,200 images from 7 further videos.
The official ODv2 val split was **not** used for model selection: it shares videos with train (see below).

## Evaluation (1,200 test images from 7 unseen videos, 6,825 objects)
COCO AP at IoU 0.50:0.95 (`training/evaluate_sea.py`); source `benchmarks/results/seadronessee.csv`.

| input | AP50 | AP | AP small | swimmer | boat | jetski | life-saving app. | buoy | CPU ms (2 vCPU) |
|---|---|---|---|---|---|---|---|---|---|
| 640 | 0.438 | 0.221 | 0.145 | 0.126 | 0.455 | 0.110 | 0.000 | 0.413 | 75 |
| 1280 | 0.488 | 0.266 | 0.207 | 0.185 | 0.503 | 0.143 | 0.000 | 0.500 | 259 |

Leakage check: on 400 official-val frames from *training* videos the same model scores swimmer AP 0.295 vs 0.185
on unseen videos. About +0.09 of that gap is leakage (a COCO model shows only +0.018 between the same two sets).

Tracking on 4 real SeaDronesSee-MOT clips (1280 px, conf 0.1): IDF1 0.50–0.79 (`benchmarks/results/tracking_sea.csv`).

## Performance on device
Jetson Orin Nano: not yet measured (`benchmarks/run_on_jetson.sh`).

## Limitations and risks
Does not detect life-saving appliances at all yet. Most swimmers are under 16 px at 1280 px input and many are
missed. Trained on summer footage from a few lakes and the North Sea, mostly DJI cameras; rough sea, glare, night and
thermal imagery are untested. Only 4 CPU epochs: treat the numbers as a lower bound. See `docs/LIMITATIONS.md`.
