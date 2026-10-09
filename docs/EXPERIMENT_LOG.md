# Experiment Log

One entry per run or decision, written the same day. All numbers below were measured;
nothing is estimated. Raw rows live in `benchmarks/results/*.csv`.

**Environment for the 2026-10-09 runs:** cloud VM, 2 vCPU, 7 GB RAM, **no GPU**.
Ultralytics 8.4.174, SAHI 0.12.8, PyTorch 2.14.1 (CPU), Python 3.13.
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
  at 882 ms per image; plain 1280 inference gets 0.124 in 317 ms. VisDrone frames are only ~1360–2000 px
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

---

## Template

### YYYY-MM-DD — <tag>
- **Change vs parent:** (exactly one thing)
- **Hypothesis:**
- **Config:** `configs/train/....yaml`, run folder `runs/...`
- **Result (val):** AP50 / AP / AP_small / ms per image
- **Conclusion:** keep, drop, or investigate
- **Surprises / bugs:**
