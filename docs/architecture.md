# Architecture

```mermaid
flowchart LR
    CAM[Camera RGB] -->|OpenCV| TRT[YOLO11<br/>PyTorch / ONNX INT8 / TensorRT*]
    TRT --> TRK[Tracker<br/>ByteTrack / BoT-SORT + GMC]
    TEL[MAVROS telemetry<br/>esa.telemetry: stale → no position] --> GEO
    TRK --> GEO[Pixel → lat/lon<br/>esa.geo, tilt-aware]
    GEO --> ALR[Alert debouncing<br/>esa.alerts]
    ALR --> ROS[(ROS 2 topics<br/>/detections /tracks_geo /alerts)]
    ROS --> WS[FastAPI WebSocket bridge]
    JSONL[JSONL replay file] -.laptop mode.-> WS
    WS --> MAP[React + MapLibre map<br/>tracks, trails, alerts]
```

Offline side: public datasets → `data/convert_*.py` → YOLO format → `training/train.py` → `training/evaluate_coco.py` (validation) → chosen model → `export/export_trt.py` on the Jetson → `benchmarks/benchmark.py`.

\* TensorRT engines are built by `export/export_trt.py` on the Jetson; not yet measured. Thermal cameras and rosbag
recording are not part of this release.
