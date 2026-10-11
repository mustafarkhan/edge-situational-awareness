# Models

| file | what | card |
|---|---|---|
| `sds_y11n_ep4.pt` | YOLO11n fine-tuned on SeaDronesSee (swimmer, boat, jetski, life-saving appliances, buoy) | [model card](../docs/MODEL_CARD_sds_y11n_ep4.md) |

Licence: AGPL-3.0 (inherits from Ultralytics YOLO). The training data (SeaDronesSee) is CC0 1.0.
The VisDrone model (`ft_y11n_ep3`) is not distributed because VisDrone states no licence; retrain it with
`training/train.py configs/train/vis_y11n_640_cpu.yaml`.
