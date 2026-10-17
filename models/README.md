# Models

| file | what | card |
|---|---|---|
| **`sds_y11n_1280_gpu.pt`** | **Sea model (default).** YOLO11n trained on SeaDronesSee at 1280 px on a GPU (swimmer, boat, jetski, life-saving appliances, buoy) | [model card](../docs/MODEL_CARD_sds_y11n_1280_gpu.md) |
| **`hit_y11n_640_gpu.pt`** | **Thermal model (default).** YOLO11n trained on HIT-UAV thermal images on a GPU (person, car, bicycle, other vehicle), 640 px | [model card](../docs/MODEL_CARD_hit_y11n_640_gpu.md) |
| `sds_y11n_ep4.pt` | Earlier sea model, 4 CPU epochs at 640 px. Kept because README §9-§10a numbers were measured with it | [model card](../docs/MODEL_CARD_sds_y11n_ep4.md) |
| `hit_y11n_ep5.pt` | Earlier thermal model, 5 CPU epochs. Kept because README §10b numbers were measured with it | [model card](../docs/MODEL_CARD_hit_y11n_ep5.md) |

Check the files before use: `cd models && sha256sum -c SHA256SUMS` (also checked in CI).

Licence: AGPL-3.0 (inherits from Ultralytics YOLO). The training data (SeaDronesSee, HIT-UAV) is CC0 1.0.
The VisDrone models are not distributed because VisDrone states no licence; retrain them with
`training/train.py configs/train/vis_y11n_640_cpu.yaml` (CPU) or the Kaggle notebook (GPU, [docs/KAGGLE.md](../docs/KAGGLE.md)).
