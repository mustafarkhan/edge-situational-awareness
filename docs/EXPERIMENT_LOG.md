# Experiment Log

One entry per run or decision, written the same day. All numbers below were measured;
nothing is estimated. Raw rows live in `benchmarks/results/*.csv`.

**Environment (2026-10-09 and 10-10 runs):** cloud VM, 2 vCPU, 7 GB RAM, **no GPU**.
Ultralytics 8.4.174, SAHI 0.12.8, PyTorch 2.14.1 (CPU), ONNX Runtime 1.29.0, motmetrics 1.4.0, Python 3.13.
Jetson measurements (Lab 3) are still to come.

---

## Dataset statistics — VisDrone2019-DET val

`python data/analyse_dataset.py --data configs/datasets/visdrone.yaml --split val --imgsz 640 1280`

548 images, 38,759 boxes, 70.7 objects per image on average (max 317).

| class | boxes | share |
|---|---|---|
| car | 14064 | 36.3% |
| pedestrian | 8844 | 22.8% |
| people | 5125 | 13.2% |
| motor | 4886 | 12.6% |
| van | 1975 | 5.1% |
| bicycle | 1287 | 3.3% |
| tricycle | 1045 | 2.7% |
| truck | 750 | 1.9% |
| awning-tricycle | 532 | 1.4% |
| bus | 251 | 0.6% |

| model input | median object size | < 8 px | < 16 px | < 32 px (COCO "small") |
|---|---|---|---|---|
| 640 | 11.3 px | 31.0% | 69.9% | 92.4% |
| 1280 | 22.5 px | 5.8% | 31.0% | 69.9% |

**Takeaway:** at the default input size almost every object is "small" and a third are under 8 px.
Small-object AP is the metric that matters; overall mAP hides the problem.

---

### 2026-10-09 — evaluation bug: overall AP = -1.0
- **Symptom:** first smoke test printed AP = -1.0 while AP50 and AP_small looked normal.
- **Cause:** VisDrone scenes have up to 317 objects, so evaluation uses maxDets = 500. pycocotools'
  `summarize()` hard-codes maxDets = 100 for the overall AP only, finds no such setting, and returns -1.
- **Fix:** compute overall AP directly from the precision array at the largest maxDets
  (`src/esa/coco_utils.py`); regression test `test_overall_ap_works_with_large_max_dets`.
- **Lesson:** check the measuring instrument before trusting any number.

### 2026-10-09 — zero-shot baselines (COCO-pretrained, no fine-tuning)
- **Question:** how big is the domain gap between everyday photos (COCO) and drone imagery?
- **Method:** `training/evaluate_zeroshot.py`, all 548 val images, classes mapped to
  person / two-wheeler / vehicle (tricycles left out, see script docstring), conf 0.01, maxDets 500.
  Times are median end-to-end ms per image on the 2-vCPU VM.

| model | weights | input | slicing | AP50 | AP | **AP_small** | AP_large | CPU ms/img |
|---|---|---|---|---|---|---|---|---|
| YOLO11n | COCO | 640 | — | 0.200 | 0.108 | 0.041 | 0.576 | 86 |
| YOLO11s | COCO | 640 | — | 0.280 | 0.153 | 0.075 | 0.685 | 200 |
| YOLO11n | COCO | 1280 | — | 0.356 | 0.198 | 0.124 | 0.684 | 317 |
| YOLO11s | COCO | 1280 | — | 0.433 | 0.244 | 0.164 | 0.750 | 634 |
| YOLO11n | COCO | 640 | SAHI, 640 px tiles | 0.344 | 0.180 | 0.110 | 0.594 | 882 |
| YOLO11n | fine-tuned, 1 epoch | 640 | — | 0.266 | 0.132 | 0.057 | 0.435 | 79 |
| YOLO11n | fine-tuned, 2 epochs | 640 | — | 0.316 | 0.161 | 0.079 | 0.483 | 80 |
| YOLO11n | fine-tuned, 2 epochs | 1280 | — | 0.448 | 0.238 | **0.169** | 0.354 | 319 |

- **Resolution beats model size for small objects:** YOLO11n at 1280 (AP_small 0.124, 317 ms)
  beats YOLO11s at 640 (0.075, 200 ms).
- **SAHI helps but is not worth it here:** slicing raised YOLO11n's AP_small from 0.041 to 0.110, but
  at 882 ms per image; plain 1280 inference gets 0.124 in 317 ms. VisDrone val frames are only 960–1920 px
  wide, so 640 px slices add little detail over a 1280 input. SAHI should matter more on 4K footage;
  that is a hypothesis to test, not a result.
- The YOLO11s + SAHI run was stopped to free the CPU for fine-tuning.

### 2026-10-09 — held-out demo sequences
- Moved VisDrone train sequences **9999951** (334 frames) and **9999955** (434 frames) to
  `datasets/demo_holdout/` **before** fine-tuning. They are consecutive video frames, so the tracking
  demo runs on footage the model has never seen. Training set: 5,703 images.

### 2026-10-09 — first fine-tune (CPU)
- **Change vs parent:** fine-tune COCO-pretrained YOLO11n on VisDrone train (5,703 images, 10 classes), 640 px.
- **Config:** `configs/train/vis_y11n_640_cpu.yaml`. Planned 3 epochs; results use the epoch-2 checkpoint
  (~50 min per epoch on 2 vCPU). Defaults left as-is: warm-up spans 3 epochs and mosaic is off (close_mosaic=10 > epochs),
  so the model is under-trained. Evaluated with the same three-class metric as the baselines (`--class-map visdrone`).
- **Result (val):** see table above. AP_small 0.041 → 0.057 (1 epoch) → 0.079 (2 epochs) at 640; 0.169 at 1280.
- **Conclusion:** in-domain data is the biggest lever. Large-object AP dropped; expected to recover with longer training.
- **Next:** full schedule (50–100 epochs) on a GPU; train at 1280, since inference at 1280 helped most.

### 2026-10-09 — tracking demo failed, and why (second bug story)
- **Symptom:** ByteTrack on held-out sequence 9999951 produced 189 tracks, IDs in the thousands, a median of 1 detection per frame
  against 20–100 labelled objects, and zero alerts.
- **Cause 1:** the VisDrone-DET "sequences" are frames seconds apart; the drone moves to a new stretch of road between frames.
  No tracker can link objects across that gap.
- **Cause 2:** ByteTrack only outputs *confirmed* tracks. On sparse frames most objects never get confirmed, so the tracker
  silently hid most detections. Detection-only mode (`--no-track`) restored 8–76 detections per frame.
- **Fix:** `tools/detect_video.py --no-track`; demo shows detection on held-out frames; tracking deferred to continuous video.
- **Lesson:** check the data's time structure before choosing an algorithm, and never let a downstream stage hide upstream output unnoticed.

### 2026-10-09 — held-out frames (demo)
- 21 frames sampled from sequence 9999951 (every 16th), fine-tuned YOLO11n, 1280 px, conf 0.25, detection only.
- Greedy class-agnostic matching at IoU ≥ 0.5: **422 of 836 labelled objects found (50%)**.

### 2026-10-10 — epoch 3 (training interrupted twice)
- The VM restarted overnight (training 37% into epoch 3) and again at 13:00. Resumed both times from the
  epoch-2 checkpoint with `YOLO(last.pt).train(resume=True)`, the second time fully detached (`setsid nohup`).
  Fixed seed + deterministic mode reproduced the exact same loss values after resuming, a nice check.
- **Result (val):** AP_small 0.087 at 640 and **0.182 at 1280** (epoch 2: 0.079 / 0.169). Overall AP 0.253.
  Large-object AP 0.378, still well below the pretrained 0.58 (see failure analysis).
- Held-out frames (21 frames of sequence 9999951): 448 of 836 objects found (epoch 2: 422).

### 2026-10-10 — controlled tracking test
- **Problem:** no continuous drone video with identity labels was downloadable (only GitHub/PyPI/npm reachable).
- **Method:** `tools/sim_flight.py` flies a 960×540 camera in an ellipse over val image 0000129_02411_d_0000138
  (near-vertical, 186 labelled objects), zooming to vary altitude 57–69 m. Static objects with exact identities
  and positions → exact ID-switch and position-error scoring. Two speeds × three trackers, same detections settings
  (fine-tuned model, 1280 px, conf 0.1). The two BoT-SORT variants differ only in `gmc_method`.
- **Result:** see README §3. Fast camera: IDF1 0.33 (ByteTrack), 0.34 (BoT-SORT no GMC), **0.57** (BoT-SORT + GMC);
  ID switches 244 / 301 / 77. Slow camera: 0.66 / 0.68 / 0.70, switches 15 / 7 / 3.
- **Conclusion:** use BoT-SORT with motion compensation on drones. Without it, recall at speed also halves (31% vs 59%)
  because unconfirmed tracks hide their detections.
- **Bug found on the way:** `motmetrics` 1.4 calls `np.asfarray`, removed in NumPy 2. Wrote the IoU distance by hand
  (`benchmarks/tracking_eval.py:iou_distance`, unit-tested).

### 2026-10-10 — geolocation error budget
- With perfect telemetry the pipeline's error is 0.14 m median: that validates the code, nothing more.
- `benchmarks/geo_sensitivity.py` perturbs one input at a time over 7,634 boxes. Gimbal tilt is evaluated with a
  pinhole ray / flat-ground intersection (unit test: zero tilt reproduces the nadir model; centre shift = alt·tan θ).
- **Result:** GPS 5 m → 5.0 m; altitude +10% → 2.9 m; heading 5° → 2.5 m; **tilt 10° ignored → 11.8 m**.
- **Conclusion:** the straight-down assumption is the largest error source; fix it first.

### 2026-10-10 — INT8 on CPU (ONNX Runtime)
- **Hypothesis:** quantising everything will break the scores, because box coordinates (0–1280) and class scores (0–1)
  share one output tensor and one 8-bit range.
- **Method:** `export/quantize_onnx.py`, static QDQ, per-channel weights, MinMax calibration on 128 train images.
- **Result (548 val, 1280 px):** FP32 ONNX AP_small 0.181, 251 ms; INT8 everything: **no detections at all**, 162 ms;
  INT8 with 48 head post-processing nodes in FP32: **0.172, 201 ms, 3.7 MB** (vs 11.1 MB).
- **Conclusion:** hypothesis confirmed. −5% small-object AP for −20% time and −66% size. Next: does TensorRT INT8 behave the same?
- The first version of the exclusion list (84 nodes) also kept the head's conv-block activations in FP32; narrowed to
  post-processing only.

### 2026-10-10 — failure analysis
- `data/failure_analysis.py`, raw VisDrone annotations (for occlusion levels), class-agnostic matching at IoU 0.5.
- Recall by size at model input: < 8 px 10%, 8–16 px 26%, 16–32 px 49%, 32–64 px 73%, ≥ 64 px 87%.
- Recall by occlusion: none 63%, partial 42%, heavy 23%. By class: vehicle 74%, person 34%, two-wheeler 23%.
- Night (17 images, brightness < 70; threshold picked from the brightness histogram's gap and checked by eye):
  recall unchanged (48.7% vs 49.5%) but **precision 44% vs 70%**: duplicate and phantom boxes.
- Worst images (recall 13–16%): low oblique view down a long road; a near-ground view where *large* close-up people and
  scooters were missed (explains the large-object AP drop after fine-tuning); a toll plaza with booths occluding vehicles.

### 2026-10-10 — map app recording
- Real React app, real FastAPI backend, headless Chromium via Playwright (`tools/record_map.py`). Map tiles unreachable,
  so added an offline style (`?offline=1`, 20 m grid).
- Two small bugs fixed: alerts appeared twice because the replay looped (app now resets on frame 0; backend gained
  `--once`); the view centred on the drone's first position, so the flight drifted off-screen (`?center=&zoom=`).

### 2026-10-10 — ROS 2 node, tested end to end in CI
- ROS could not be installed on the build machine, so a GitHub Actions workflow installs ROS 2 Humble, builds the
  package, runs the node on Ultralytics' sample images and checks every output topic (`ros2_ws/test_node.sh`).
  Results are emitted as Actions annotations, because raw CI logs were not downloadable from the build machine.
- Bugs found, in order:
  1. `ros2 topic echo` exits immediately if the topic doesn't exist yet and no type is given (test bug: pass types).
  2. Sample images aren't bundled with this Ultralytics version (test bug: download them explicitly).
  3. **Node aborted at exit** ("terminate called without an active exception"): the spin thread was still running
     during teardown. Fix: explicit executor, shut it down and join the thread before destroying the node.
  4. **Alerts lost by late subscribers:** all five alerts fired in the first frames, before the subscriber connected
     (it first saw frame 16). Fix: `/alerts` is latched (transient local, depth 10).
- One run failed inside the ROS installer (download error) and passed on re-run: infrastructure, not code.
- **Result:** clean exit (code 0); `/detections`, `/tracks_geo` and `/alerts` all received.

### 2026-10-10 — sea data: getting it, and a leakage finding
- **Access:** the SeaDronesSee site and its Nextcloud shares were not reachable from the build machine, so the subset was
  pulled through a browser on a second computer: per-image WebDAV requests (ODv2), and HTTP range requests straight
  into the 27.6 GB MOT zip (its central directory gives every frame's byte offset), resized to 1920 px in the browser,
  packed into ~300 MB tar files. 4,500 + 714 images, zero failures. Licence: CC0 1.0 (dataset), per the project README.
- **Finding:** all 34 videos in the official ODv2 *val* split also appear in *train* — frames are split, videos are
  not. Built a video-disjoint split instead (`data/sds_select_subset.py`, committed as `data/seadronessee_selection.json`):
  train 2,500 images from 24 videos, val 400 from 8, test 1,200 from 7, plus "seen" = 400 official-val frames from the
  training videos, to measure the leakage effect.
- **Data check:** at 640 px input the median sea object is **7 px** across and 59% are under 8 px (VisDrone: 11 px, 31%).

### 2026-10-10 — tilt-aware geolocation, and testing it on real footage
- `src/esa/geo.py` now intersects the camera ray with the ground using gimbal pitch and heading (pitch 90 reproduces
  the old nadir formula exactly; unit-tested, and cross-checked against the independent ray code in
  `benchmarks/geo_sensitivity.py`). Boxes above the horizon get no position.
- **Real-footage test** (`benchmarks/geo_real.py`): 4 SeaDronesSee-MOT clips with the drone's own log (GPS, altitude,
  gimbal pitch/heading). No ground-truth object positions exist, so the check is physical: a swimmer moves ≲ 1 m/s, and
  wrong geometry makes swimmers appear to move with the drone. Ground-truth boxes, not detections. HFOV 65.5° assumed.
- **Result:** very oblique clip (DJI_0055, pitch 16°, drone 2.8 m/s): swimmer apparent speed **3.17 → 1.04 m/s**,
  position spread **3.9 → 0.9 m**. Steady clips: small gains or a draw. **One clip is worse** (DJI_0059: 2.13 → 3.63 m/s):
  the drone accelerates sideways to 7.6 m/s while turning 15°.
- **Diagnosis attempts:** interpolating the step-wise log (values change every 3–5 frames) changed nothing; a video-log
  clock-offset sweep (±1.5 s, leave-one-out) found no consistent offset. In DJI_0059 the scene rotates −30° in the image
  while the log reports a +15° heading change (expected image rotation at 31° pitch ≈ 8°), so the logged attitude does
  not describe the camera during that manoeuvre. Objects 50 m ahead turn small attitude errors into large position errors.
- **Conclusion:** the model is right and fixes the dominant error; its accuracy is now limited by attitude telemetry.
  Next: gimbal attitude at frame rate with timestamps (MAVLink), or estimate yaw from the image.

### 2026-10-11 — sea fine-tune (YOLO11n, 4 CPU epochs) and the checkpoint mix-up
- **Change vs parent:** same land recipe (train 640, run 1280), with `warmup_epochs 1` and `close_mosaic 1`
  (lessons from the land run). Config `configs/train/sds_y11n_640_cpu.yaml`.
- **Result (1,200 unseen-video test images, native 5 classes, 1280 px):** AP 0.266, AP50 0.488, AP_small 0.207,
  swimmer 0.185, boat 0.503, jetski 0.143, buoy 0.501, life-saving appliances 0.000. COCO model, two-class view:
  swimmer 0.094 at 1280, 0.027 at 640.
- **Bug:** the evaluation queue waited on `pgrep -f training/train.py`, but the resumed process (after a VM restart)
  was `YOLO(last.pt).train(resume=True)`, so the queue scored the epoch-1 checkpoint as "final". Caught because the
  640 rows matched the epoch-1 row exactly. Fix: wait on the resume command; print epoch / optimizer state / size of
  the checkpoint before scoring (a finished checkpoint shows epoch -1, no optimizer, 5.4 MB). Wrong rows deleted,
  everything after it re-run.
- **Conclusion:** keep. Life-saving appliances need more epochs or more examples.

### 2026-10-11 — measuring the official-split leakage
- Same fine-tuned model, 1280 px, two-class view: unseen test videos AP 0.344, "seen" (official-val frames from
  training videos) 0.463, gap +0.120. COCO model as a control for scene difficulty: 0.203 vs 0.282, gap +0.079.
  Leakage ≈ **+0.04 AP overall**; for swimmers +0.110 vs +0.018, i.e. ≈ +0.09 on 0.185, **about 50% inflation**.
- **Conclusion:** never report on the official val split for this dataset; the video-disjoint split is committed.

### 2026-10-11 — tracking on real sea footage
- 4 SeaDronesSee-MOT clips, fine-tuned model at 1280, conf 0.1, ByteTrack vs BoT-SORT+GMC.
  IDF1 0.50 / 0.79 / 0.66 / 0.77 (ByteTrack) vs 0.48 / 0.78 / 0.65 / 0.71. Camera motion 0.01–0.14 m/frame.
- **Conclusion:** consistent with the simulation's slow regime (GMC matters only when the camera moves fast).
  DJI_0055 (15 bunched swimmers, 20 m) has 40–46 ID switches; that is a detection/crowding problem, not motion.

### 2026-10-15 — v2: search planning from measured detection rates
- **Question:** how high and how fast should the drone fly so a swimmer is likely to be found?
- **Measured** (`benchmarks/detection_vs_range.py`, sea model, 1280 px, conf 0.25): 4,221 swimmers in the 1,200 unseen
  test images. With "a box overlapping the swimmer" (IoU >= 0.3) as the criterion, recall is **84%**, precision 76%,
  1.07 false alarms per image, and it stays **66-90% across the whole measured size range (4-50 px)**. The low AP
  (0.185, IoU 0.5:0.95) is mostly box tightness on tiny objects, not missed swimmers.
- **Bug caught:** first pass selected "near-nadir" frames with pitch <= -70 and found none: in SeaDronesSee ODv2
  metadata gimbal pitch is degrees *below* the horizon (90 = straight down; the Trinity frames are all 90). Fixed to
  >= 70. Swimmer size from 639 near-nadir boxes: median 1.08 m (sqrt of box area; 65.5 deg lens assumed).
- **Consecutive frames are not independent:** on the 4 real MOT clips, P(detected in >= 1 frame) for 1 s of video
  (30 frames) is 0.86, vs 1.00 if frames were independent: 1 s is worth ~1.6 independent looks, 3 s ~2.2.
- **Validation:** the pixels-on-target model predicts the swimmer size at each altitude within 3 px and detection
  within 0.10 (conservative) for 40-130 m.
- **First curve fit was optimistic:** a running maximum lifted dips to the best bin; replaced by isotonic regression
  (pool adjacent violators, weighted by swimmers per bin).
- **Result:** within the measured data, area searched per hour grows ~linearly with altitude (0.55 km^2/h at 20 m to
  2.7 km^2/h at 100 m, 80% POD, 10 m/s). The ceiling is where the data ends: 116 m if head and shoulders (0.5 m) are
  visible, 69 m for head only. **Frame rate does not limit coverage**: at 10 m/s about 1 frame/s already exceeds the
  independent looks per pass. Extra speed buys tracking and alert latency, not detection.
- **Output:** QGroundControl mission for a 0.5 km^2 example (100 m, 76 m track spacing, 7.5 km, 12 min).

### 2026-10-15 — v2: operator loop, alert de-duplication, TAK
- **Symptom:** the map app showed ~20 alerts in 5 s for 15 swimmers. Tracker ID changes re-trigger the 15-frame alert.
- **Measured** (`benchmarks/alert_load.py`, real tracking runs on the 4 clips): 33 alerts for 17 swimmers alerted
  (14 duplicates, 2 false). Spatial de-duplication: 2 m -> 21 alerts; **3 m -> 18 alerts, all 17 swimmers still
  alerted**; 5 m merges neighbouring swimmers and loses 3. Default set to 3 m. Caveat: chosen on the same 4 clips.
- Operator Confirm/Reject in the map app -> audit log, TAK update/delete, and /alert_reviews to the detector, which
  suppresses new alerts within 10 m of a rejected one for 10 min.
- **TAK:** Cursor-on-Target alerts (type a-u-S, 15 m circular error) and drone position (a-f-A-M-H-Q, 1 Hz) over UDP or
  TCP. CI starts a real TAK server (taky), connects a TAK client, replays the recorded flight, rejects an alert over
  the WebSocket and checks the client received the drone, the alert and the delete.
- **ROS 2 CI flake:** the second of three end-to-end runs crashed inside `ros2 topic echo` while auto-detecting QoS from
  a publisher cached from the previous run. Fix: fresh ROS daemon per run, explicit QoS on echo.

### 2026-10-15 — v2: thermal data (HIT-UAV) and another leaky split
- HIT-UAV (Zenodo 10.5281/zenodo.7633134, CC0): images only in an 815 MB zip on Zenodo, unreachable from the build
  machine. Read the zip's central directory with HTTP range requests in a browser on a second computer and fetched
  only the two byte ranges holding the images and annotations (201 MB); every file checked against its recorded size.
- **Leak:** the official split is random over frames sampled every 15 frames from videos; 54% of test images have a
  training image within 2 serial numbers. File names encode day/night, altitude and camera angle, so images are
  grouped by (day/night, altitude, angle) — 105 groups, a proxy for one flight — and whole groups are held out
  (`data/convert_hituav.py`): train 1,902, val 183, test 477, plus "seen" 336 held-out frames from training groups.
- Thermal objects are larger than at sea: median 27 px at 640 px input (sea: 7 px).
- **First training run was killed** (out of memory, 7 GB limit) when the YOLOE-11m prompt evaluation started next to it.
  Lesson: one heavy job at a time on this machine; queued training after the prompt runs.
- **Result** (5 CPU epochs, 640 px, 477 test images from held-out groups): COCO colour model AP 0.023 -> fine-tuned
  **0.445** (AP50 0.81; person 0.369, car 0.602, bicycle 0.362). **Night 0.471 > day 0.338** (people 0.407 vs 0.277).
  AP per altitude 0.37-0.57 over 60-130 m, no trend; person median size 21 px at 60 m, 12 px at 130 m.
- **Leakage:** seen 0.465 vs held-out 0.445 (+0.020); COCO control +0.017 -> about +0.003, much smaller than at sea.
  Possible reasons: thermal scenes are less distinctive than lake shores, or the condition groups are only a proxy for
  flights (same session at different camera angles), which would also make the held-out set easier. Not resolved.

### 2026-10-15 — v2 on land as well as sea
- **Land search planning** (`benchmarks/land_search.py`, thermal model, 477 held-out images): 91% of 1,255 people get a
  box (night 93%, day 86%), precision 75%, 0.9 false alarms per image. Person size 0.94 m (130 straight-down boxes,
  XT2 25 mm lens = 25 x 20 deg, from DJI's spec; the paper gives only the focal length). Prediction matches
  measurement at 80 m and 120 m within 2 px / 0.03. One look per pass (no land video with identities to measure
  correlation). Result: 25 deg lens 1.04 km^2/h at 120 m; 45 deg lens 0.99 km^2/h at 60 m, after which persons drop
  below the measured range (77 m ceiling vs 143 m).
- A bin with a single person (0-8 px, p = 0) dragged the extrapolated curve to zero; bins merged so each has 30+ people.
- **Text prompts on land** (VisDrone val, pedestrian + car, one untuned wording): COCO 0.251, YOLOE-11s 0.254,
  YOLOE-11m 0.285, fine-tuned 0.328 AP. Same ordering as at sea.
- TAK: ground marker type a-u-G for land classes.

### 2026-10-17 — GPU training (Kaggle, 2× T4)
- **Change vs parent:** epochs only (3-5 CPU epochs -> 60-120 GPU epochs, early stop on val), plus one sea run trained
  at 1280 px. Same data, splits and seed. Configs `configs/train/gpu/*.yaml`, notebook `kaggle/esa_gpu_training.ipynb`;
  3 h on Kaggle. The notebook downloads only train/val images: the test sets never left this machine.
- **Best epochs (val mAP50-95):** sea 640 px 48 of 73 (early stop), sea 1280 px 39 of 60, thermal 112 of 120, land 70 of 80.
- **Sea (1,200 test images):** AP 0.266 -> 0.296 (640-trained, run at 1280) -> **0.325** (1280-trained); swimmer
  0.185 -> 0.222; life-saving appliances 0.000 -> 0.019. Leakage grew: swimmer seen-vs-unseen gap +0.110 -> +0.149.
- **Thermal (477 test images):** AP 0.445 -> **0.512**; day 0.338 -> 0.453, night 0.471 -> 0.525. Leakage grew too:
  seen-vs-held-out +0.020 -> +0.072 (control +0.017).
- **Land (VisDrone val, 548):** AP at 1280 0.253 -> **0.336**, AP small 0.182 -> 0.266; at 640 large-object AP back to
  0.558 (COCO 0.576, 3 epochs 0.527). Val also picked the epoch, so new `benchmarks/holdout_recall.py` scores the 768
  frames of the two sequences held out before training: 54% -> **67%** found (IoU >= 0.5, conf 0.25), precision 69% -> 71%.
  On the 21 demo frames it counts 445 of 836 for the 3-epoch model, against 448 in the hand count of 2026-10-10
  (matching order differs).
- **Conclusion:** keep; GPU sea and thermal models become the defaults. Longer training memorises training scenery
  more, so the leakage-controlled test sets matter more, not less. Re-run §10 planning with the new models next.

### 2026-10-21 — land planning with the GPU thermal model
- **Bug found first:** `land_search.py` cached predictions in one file whatever the model, so a new model would have
  silently reused the old model's predictions. The cache is now keyed by the model's SHA-256.
- **Result:** person boxes on 82% of 1,255 test people (5-epoch: 91%), precision 88% (75%), false alarms 0.32 per
  image (0.91). By altitude: at <= 115 m 91% at 0.40 FA/image vs 90% at 1.01; at 130 m (one group, night, 80 deg)
  43% vs 95%. Class-agnostic, the GPU model still boxes 93% of people; in that group 126 of 137 are labelled car.
- **Plan:** size-only curve, unchanged method: 25 deg lens at 100 m, 0.77 km^2/h (was 45 deg at 60 m, 0.99 km^2/h).
  Validation: 80 m predicted 0.90 vs measured 0.89; 120 m predicted 0.59 vs measured 0.98, i.e. conservative.
  Fitting only <= 115 m frames was tried as a check, not adopted (it would change the method after seeing the test
  data); it still predicts 0.71 at 120 m.
- **SITL:** new land plan flown to landing, 48/48 waypoints, max cross-track 1.15 m, max altitude 100 m.
- **Next:** size-plausibility check for "car" boxes, or night data above 120 m; validate either on val, not test.

---

## Template

### YYYY-MM-DD — <tag>
- **Change vs parent:** (exactly one thing)
- **Hypothesis:**
- **Config:** `configs/train/....yaml`, run folder `runs/...`
- **Result (val):** AP50 / AP / AP_small / ms per image
- **Conclusion:** keep, drop, or investigate
- **Surprises / bugs:**
