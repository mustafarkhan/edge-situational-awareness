# Changelog

## v3.0.0-dev — towards field tests on real drones

- **RGB + thermal through to the operator (audit F-3, F-4):** the ground station subscribes to every camera
  (`--camera-topics`), gives tracks camera-scoped IDs (`rgb-5`, `thermal-5`; reviews and TAK markers use them),
  and fuses alerts from different cameras within 5 m / 30 s into one alert marked "seen by rgb + thermal"
  (`esa.fusion`). TAK is sent from a background thread with a bounded queue; operators are sent to in parallel
  with a timeout, so a slow tablet or TAK server cannot stall the map. New end-to-end ROS test of the dual setup
  (`ros2_ws/test_dual.sh`), browser test extended to two cameras
- **Flight safety (audit F-1, F-2, F-5):** `esa.mission` checks every plan before upload and refuses it if a
  waypoint is less than 5 m under the 120 m ceiling, outside the geofence, or take-off/return-to-launch is missing.
  The autopilot is set to enforce the same limits: fence ceiling at 120 m (was search altitude + 30 m, i.e. 150 m),
  return-to-launch on breach, and a return altitude for the site instead of ArduCopter's 15 m default (`RTL_ALT_M`
  on ArduCopter 4.7+, `RTL_ALT` before; PX4 `GF_*`/`RTL_RETURN_ALT`). Settings are read back. SITL now flies whole
  sorties to landing and checks the highest altitude. Found in SITL: a plan at exactly 120 m breached the fence on
  normal overshoot, hence the 5 m margin; the land example now flies the 45° lens at 60 m (0.99 km²/h, vs 0.98 for
  the 25° lens at 110 m). Mission tool split into tested functions.
- **Ground-station security (audit S-1, S-2, S-9):** every WebSocket client authenticates with an operator token
  before receiving anything; browser origins are checked; the server listens locally unless told otherwise and
  serves TLS when given a certificate; decisions are attributed server-side and kept in a hash-chained audit log;
  review messages validated, rate-limited and size-capped; alert memory bounded; `/health` no longer shows the
  TAK endpoint. Map app: popups built from text, MapLibre GL 6.14 (fixes GHSA-jrc7-96c5-q579), a
  Content-Security-Policy, and a browser test in CI that runs the built app against a script-carrying label
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
