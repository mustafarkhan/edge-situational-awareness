# Model Card: YOLO11n fine-tuned on VisDrone (ft_y11n_ep3)

**Model:** YOLO11n (2.6 M parameters), COCO-pretrained, fine-tuned 3 epochs on VisDrone2019-DET train at 640 px,
on CPU (config `configs/train/vis_y11n_640_cpu.yaml`). Recommended inference size: 1280 px.
**Formats measured:** PyTorch FP32; ONNX FP32; ONNX INT8 with head post-processing in FP32.
**Version / date:** 2026-10-10. **Licence:** AGPL-3.0 (Ultralytics). Weights are not distributed (dataset terms unclear).

## Intended use
Research and demonstration of small-object detection in drone imagery on embedded hardware.
**Not for operational search-and-rescue or any safety-relevant decision.**

## Training data
VisDrone2019-DET train minus sequences 9999951 and 9999955 (held out for demos): 5,703 images, 10 classes.

## Evaluation (VisDrone2019-DET val, 548 images, 38,759 objects)
Three classes (person / two-wheeler / vehicle), COCO AP at IoU 0.50:0.95, maxDets 500 (`training/evaluate_zeroshot.py`).

| format | input | AP50 | AP | AP small | AP large | CPU ms (2 vCPU) |
|---|---|---|---|---|---|---|
| PyTorch FP32 | 640 | 0.334 | 0.173 | 0.087 | 0.527 | 76 |
| PyTorch FP32 | 1280 | 0.469 | 0.253 | 0.182 | 0.378 | 295 |
| ONNX FP32 | 1280 | 0.466 | 0.251 | 0.181 | 0.385 | 251 |
| ONNX INT8 (head FP32) | 1280 | 0.445 | 0.239 | 0.172 | 0.368 | 201 |

Per-class AP at 1280 (PyTorch): person 0.150, two-wheeler 0.100, vehicle 0.510.
Recall at confidence 0.25: 49.5% overall; 10% for objects under 8 px; 23% for heavily occluded objects.

## Performance on device
Jetson Orin Nano: not yet measured (`benchmarks/run_on_jetson.sh`).

## Limitations and risks
Misses most objects under 16 px; close-range, near-ground views are worse than before fine-tuning (large-object AP
0.58 → 0.38); precision drops to 44% at night. Trained on one dataset from one region. See `docs/LIMITATIONS.md`.
