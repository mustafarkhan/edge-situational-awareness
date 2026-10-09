# Model Card: <model tag>

**Model:** YOLO11<n/s> fine-tuned from COCO-pretrained weights · input <640/1280> · <FP16/INT8> TensorRT engine
**Version / date:**
**Licence:** AGPL-3.0 (inherits from Ultralytics); dataset terms below also apply.

## Intended use
Research and demonstration of real-time detection of <classes> in overhead drone imagery on embedded hardware. **Not for operational search-and-rescue decisions.**

## Training data
Dataset, version, licence/terms, citation, number of images and boxes per split, how splits were made.

## Evaluation
| split | AP50 | AP50:95 | AP_small | per-class AP |
|---|---|---|---|---|
| test | | | | |

Measured with `training/evaluate_coco.py`, confidence 0.001, maxDets <N>.

## Performance on device
Jetson <model>, JetPack <version>, power mode <mode>, clocks locked: FPS, p99 latency, W, J/frame. From `benchmarks/results/jetson.csv`.

## Limitations and risks
Small objects below ~<N> px, glare, night, unseen sea states, false alarms on <...>. See `docs/LIMITATIONS.md`.
