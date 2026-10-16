# Changelog

## v3.0.0-dev — towards field tests on real drones

- **Flight-ready missions:** `.plan` files now take off, set speed, search, return to launch, with an inclusion
  geofence 50 m around the search box
- **`tools/fly_mission.py`:** MAVLink mission + geofence upload to PX4/ArduPilot; arms and flies only in a simulator
- **ArduPilot SITL in CI:** both example missions flown end to end (14/14 and 36/36 waypoints, within 1.1 m of the
  planned track, never outside the fence). Found and fixed: ArduPilot's default 300 m circle fence aborts 1 km searches
- **ROS 2 node:** camera frames from `sensor_msgs/Image` (newest frame only, dropped frames counted; mono16
  thermal stretched to 8 bit), `/status` health topic (OK / DEGRADED / NO_CAMERA), `esa_dual.launch.py` for RGB +
  thermal gimbals with per-camera model and FOV, rosbag recording of every sortie
- **docs/DEPLOYMENT.md:** what the software may and may not do in the air, hardware, EU/Finnish rules checklist,
  humanitarian-use boundary, staged field tests T0-T6 with pass conditions, field data protocol

## v2.0.0 — 2026-10-15 — search planning, operator loop, TAK, text prompts, thermal

- **Land and sea throughout:** search planning for a person on land at night (thermal, 91% get a box; a 25° lens at
  120 m ≈ a 45° lens at 60 m, ~1 km²/h), text prompts measured on land too, TAK ground (a-u-G) vs sea (a-u-S) markers
- **Search planning** from measured data: detection vs pixels on 4,221 unseen swimmers (84% get a box at IoU 0.3),
  measured look correlation on real clips, altitude-validated size model, sweep rate per altitude, QGroundControl
  mission export. 100 m searches ~5× more sea per hour than 20 m; ~1 frame/s is enough for coverage at 10 m/s
- **Operator loop:** Confirm/Reject in the map app, audit log, rejection memory in the detector; spatial alert
  de-duplication (3 m: 45% fewer alerts, no swimmer lost on real clips)
- **TAK / Cursor on Target:** alerts and drone position to ATAK/WinTAK/TAK Server (UDP/TCP); end-to-end CI test against
  a real TAK server
- **Text-prompt search** (YOLOE): measured on sea data (fine-tuned model still 1.6× better on swimmers);
  `tools/search_by_text.py`
- **Night / thermal** on HIT-UAV with a flight-disjoint split (official split leaks frames 0.5 s apart)
- **Jetson:** dataset-free one-command run with shipped INT8 calibration images; `docs/JETSON.md`
- ROS 2 CI hardened (fresh daemon, explicit QoS); README checker now covers 110+ numbers

## v1.1.0 — 2026-10-11 — at sea

Maritime module on SeaDronesSee (CC0 1.0), plus real-footage tracking and tilt-aware geolocation.

- Video-disjoint SeaDronesSee subset (2,500 / 400 / 1,200 images); median object 7 px at 640 px input
- Fine-tuned YOLO11n (4 CPU epochs): swimmer AP **0.027 → 0.185**, boat **0.155 → 0.503** vs the COCO model
- Found and measured leakage in the official val split: swimmer AP overstated by **~50%**
- Tracking on 4 real sea clips: IDF1 0.50–0.79; GMC no help at slow camera speeds, as the simulation predicted
- Tilt-aware geolocation (`pitch_deg`, ray / ground intersection) in `geo.py`, the CLI and the ROS 2 node; on real flight
  logs swimmer apparent drift **3.2 → 1.0 m/s**, worse on one clip with an inconsistent attitude log
- Map app replays a real sea flight; demo video extended to 2.5 min with the sea results
- ROS 2 node takes live telemetry (MAVROS `fix`, `rel_alt`, `heading`, plus `gimbal_pitch`) and publishes no
  coordinates when telemetry is missing or stale; CI runs the node three times (nadir, 45° tilt, live telemetry)
- Runs without any dataset: sea model (`models/`), 1 s of real footage with its flight log (`samples/`), a recorded
  flight for the map (`web/sample/`), and a CI job that runs the README quick start on them
- `benchmarks/check_readme.py` (in CI): 74 README numbers checked against the results files

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
- Frame-rate, timestamped gimbal attitude (MAVLink) for real-world geolocation
- Longer sea fine-tune (life-saving appliances)
