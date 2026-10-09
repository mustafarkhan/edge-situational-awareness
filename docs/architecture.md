# Architecture

```mermaid
flowchart LR
    CAM[Camera RGB / thermal] -->|GStreamer / OpenCV| TRT[TensorRT YOLO11<br/>FP16 / INT8]
    TRT --> TRK[Tracker<br/>ByteTrack / BoT-SORT]
    TRK --> GEO[Pixel → lat/lon<br/>esa.geo]
    GEO --> ALR[Alert debouncing<br/>esa.alerts]
    ALR --> ROS[(ROS 2 topics<br/>/detections /tracks_geo /alerts)]
    ROS --> BAG[rosbag record / replay]
    ROS --> WS[FastAPI WebSocket bridge]
    JSONL[JSONL replay file] -.laptop mode.-> WS
    WS --> MAP[React + MapLibre map<br/>tracks, trails, alerts]
```

Offline side: public datasets → `data/convert_*.py` → YOLO format → `training/train.py` → `training/evaluate_coco.py` (validation) → chosen model → `export/export_trt.py` on the Jetson → `benchmarks/benchmark.py`.
