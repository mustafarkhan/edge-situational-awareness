# Changelog

## v1.0.0 — 2026-10-10

First complete release: detection, tracking, geolocation, INT8 and failure analysis, all measured on public data (VisDrone). TensorRT/Jetson measurements are the next release.

**Highlights** (details and sources in the README):
- Small-object AP **0.041 → 0.182** with 1280 px input and a short fine-tune of YOLO11n, beating a 3× larger pretrained model in half the time
- Camera-motion compensation: IDF1 **0.33 → 0.57**, ID switches **244 → 77** with a fast-moving camera (controlled flight test with exact ground truth)
- Geolocation error budget: ignoring a **10° gimbal tilt costs 11.8 m**, more than a 5 m GPS error
- INT8 (ONNX Runtime): quantising everything gives **no detections**; head kept in FP32: −5% small-object AP, −20% time, −66% size
- Failure analysis by object size, occlusion, class and lighting (night: precision 70% → 44%)
- ROS 2 Humble node tested end to end in CI; React + MapLibre live map

**Demo video:** `docs/video/demo_video.mp4` (2 min), rendered only from real outputs.

All speeds are from a 2-vCPU machine without a GPU; tracking was tested on a simulated flight over a real labelled frame, not on real continuous video. See `docs/LIMITATIONS.md`.

## Unreleased

- TensorRT FP16/INT8 on Jetson Orin Nano (`benchmarks/run_on_jetson.sh`)
- Real continuous footage (VisDrone-MOT, own flights); maritime data (SeaDronesSee)
