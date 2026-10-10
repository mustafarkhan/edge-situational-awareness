# Lab Guide: Edge AI for Real-Time Situational Awareness

Read this the way you would read a course handout. Each lab has the same shape:

- **Why** — the idea you need to understand. Interviewers test this, not your commands.
- **Do** — the steps, in order.
- **Checkpoint** — what must exist before you move on. Don't skip ahead; a broken lab 1 poisons everything after it.
- **Viva** — questions I would ask you in an oral exam. If you can't answer them out loud in two minutes without notes, re-read the Why section.

Keep a paper or digital notebook open the whole time. Every decision you make and every number you get goes into `docs/EXPERIMENT_LOG.md` the same day. In an interview, "I tried X, it cost Y, so I chose Z" is worth more than any single metric.

**Time budget:** about four weeks of evenings and weekends. Labs 3 and 4 are where people get stuck; leave slack there.

---

## Lab 0 — Ground rules and setup (2–3 days)

### Why

Two things can sink this project before it starts, and neither is technical.

The first is your employment contract. You work on autonomy for rescue drones, and this project is about detection for search and rescue from drones. That is close to your employer's field. Read the clauses on intellectual property, side projects and competing activity. If anything is unclear, ask your manager in writing and keep the reply. Build this on your own hardware, in your own time, with public data only, and never use anything you learned from internal code, data or documents. Say all of this in the README; it signals maturity, not weakness.

The second is licensing. Ultralytics YOLO is AGPL-3.0. That means your repo must be open source under a compatible licence, which is fine for a portfolio. Datasets have their own terms; you'll record them in Lab 1.

On the technical side, the Jetson must run from an NVMe SSD, not an SD card. SD cards are slow and wear out, and they make benchmarks noisy.

### Do

1. Read your contract. Write one line in your notebook: "Checked contract on DATE; outcome: ..."
2. Flash JetPack 6 to the NVMe SSD using NVIDIA SDK Manager (from an Ubuntu x86 PC) or the official SD-card-then-migrate route described in the Jetson Orin Nano getting-started guide. Follow NVIDIA's current instructions for your exact board; they change between JetPack releases.
3. On the Jetson:
   ```bash
   sudo apt update && sudo apt install -y python3-pip nvme-cli
   sudo pip3 install -U jetson-stats   # gives you `jtop`; reboot afterwards
   sudo nvpmodel -q                    # note your power mode
   ```
4. Install jetson-containers (see `docker/README.md`) and start a PyTorch container with this repo mounted.
5. Inside the container:
   ```bash
   pip install -r requirements-jetson.txt && pip install -e .
   yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
   ```
6. Create the GitHub repo, push this skeleton, and make the first commit "Project skeleton".
7. On your training machine (desktop GPU or a cloud instance): create a venv, `pip install -r requirements-train.txt && pip install -e .`, then run `pytest -q`. All tests should pass.

### Checkpoint

The Jetson boots from SSD, `jtop` shows the GPU, and the stock YOLO11n demo produces an annotated image. Tests pass on the training machine.

### Viva

1. Why does the AGPL licence of YOLO affect your repo?
2. Why not run from an SD card?
3. What is the difference between the Jetson's power modes, and why will you fix one for all benchmarks?

---

## Lab 1 — Data and baselines (week 1)

### Why

Most of what goes wrong in computer vision is a data problem disguised as a model problem. This lab teaches four ideas.

**Train / validation / test.** You train on train. You make every decision (model size, resolution, augmentation) by looking at validation. You look at test exactly once per final model, to report numbers. If you tune on test, your published number is optimistic and an experienced interviewer will ask how you avoided that.

**Leakage.** SeaDronesSee images come from video. Two frames a fraction of a second apart are almost identical. If one lands in train and its neighbour in validation, the model is tested on something it has effectively seen, and validation scores look great until real use. The converter therefore splits in *blocks* of consecutive image IDs rather than image by image. The official SeaDronesSee test labels are withheld for their challenge server, so we use their official validation set as our test set. VisDrone has official train / val / test-dev splits, so we use those directly.

**Small objects.** A swimmer seen from 60 m may be 10–20 pixels across in a 4K frame. YOLO resizes the whole frame so that the longest side equals the input size (640 by default). A 4K frame shrinks six times, and the swimmer becomes 2–3 pixels: almost no signal. `analyse_dataset.py` shows you this table before you train. This single insight drives Lab 2.

**Transfer learning and baselines.** We start from weights pretrained on COCO (everyday photos) and fine-tune. A baseline is the simplest reasonable run; every later experiment is compared with it. Without a baseline, "70% mAP" means nothing.

**What mAP means.** For each class: sort detections by confidence, walk down the list, and trace precision against recall; the area under that curve is AP. A detection counts as correct if its box overlaps the true box by at least an IoU threshold. mAP@0.5 uses IoU 0.5 (loose). mAP@0.5:0.95 averages over thresholds from 0.5 to 0.95 (strict, rewards precise boxes). mAP is the mean over classes. COCO also reports AP_small, AP for objects under 32×32 pixels, which is the number that matters for people in the water.

### Do

1. **Download the data** (on the training machine, into `datasets/raw/`, which git ignores).
   - SeaDronesSee: from the official SeaDronesSee website (University of Tübingen), take the *Object Detection v2* data: images plus `instances_train.json` and `instances_val.json`.
   - VisDrone: from the VisDrone GitHub page, take VisDrone2019-DET train, val and test-dev.
   - Write each dataset's licence or terms of use, citation and download date into the README's Data section. As far as I know, both are released for research use; read their terms yourself and quote them accurately.
2. **Convert:**
   ```bash
   python data/convert_seadronessee.py \
     --train-json datasets/raw/sds/annotations/instances_train.json --train-images datasets/raw/sds/images/train \
     --val-json   datasets/raw/sds/annotations/instances_val.json   --val-images   datasets/raw/sds/images/val \
     --out datasets/seadronessee_yolo
   python data/convert_visdrone.py --src datasets/raw/visdrone --out datasets/visdrone_yolo
   ```
   The converters print image and box counts per split. Write them down.
3. **Look at the data.** Open 20 random images with their labels drawn (Ultralytics saves `train_batch*.jpg` mosaics at the start of training, or use any label viewer). Check boxes sit on the objects. Converters have bugs; your eyes are the test.
4. **Analyse:**
   ```bash
   python data/analyse_dataset.py --data configs/datasets/seadronessee.yaml --split train --imgsz 640 1280
   ```
   Paste the tables into `EXPERIMENT_LOG.md` and keep the figure from `docs/figures/` for the README.
5. **Build COCO ground truth for evaluation** and run the sanity check:
   ```bash
   python data/yolo_to_coco.py --data configs/datasets/seadronessee.yaml --split val
   python data/yolo_to_coco.py --data configs/datasets/seadronessee.yaml --split test
   python training/evaluate_coco.py --gt datasets/seadronessee_yolo/coco_val.json --sanity --tag sanity
   ```
   Scoring the ground truth against itself must give AP = 1.0. If it doesn't, your evaluation is broken and every later number would be wrong. Professionals check their measuring instrument before measuring.
6. **Train the baselines:**
   ```bash
   python training/train.py configs/train/sds_y11n_640.yaml
   python training/train.py configs/train/sds_y11s_640.yaml
   python training/train.py configs/train/vis_y11s_640.yaml
   ```
   While they train, read the loss curves in `runs/.../results.png`. Training loss should fall; if validation mAP plateaus early, that's normal and early stopping will end the run.
7. **Evaluate on validation:**
   ```bash
   python training/evaluate_coco.py --model runs/sds/y11n_640/weights/best.pt \
     --gt datasets/seadronessee_yolo/coco_val.json --imgsz 640 --tag y11n_640
   ```
   Repeat for each model. For VisDrone add `--max-dets 500`, because its scenes contain hundreds of objects and COCO's default of 100 detections per image would cap recall.

### Checkpoint

A baseline table in `EXPERIMENT_LOG.md` with AP50, AP, AP_small and per-class AP for each baseline, on validation, plus the dataset statistics.

### Viva

1. Explain leakage in your dataset and how your split prevents it.
2. Why is evaluation run with confidence 0.001 instead of 0.25?
3. What is the difference between mAP@0.5 and mAP@0.5:0.95, and which one rewards precise localisation?
4. From your analysis table: what share of objects is under 16 pixels at 640 input? What does that predict about recall on swimmers?
5. Which class is rarest, and how might that affect its AP?

---

## Lab 2 — Improving accuracy (week 2)

### Why

Science, not tinkering: change **one** thing per experiment, so you know what caused the change. Each config in `configs/train/` differs from its parent in exactly one way, and its comment says which.

Three levers matter for small objects:

**Higher input resolution.** Training and running at 1280 instead of 640 gives every object four times as many pixels. The cost is roughly four times the compute, which you'll measure on the Jetson in Lab 3. This trade-off is the heart of the project.

**Sliced inference (SAHI).** Instead of shrinking the whole frame, cut it into overlapping tiles (say 640×640), detect on each tile at full detail, and merge the results. Small-object recall usually improves a lot. The cost is one forward pass per tile, so a 4K frame might need 15–20 passes. Great for offline analysis or slow-moving search; often too slow for live video on an Orin Nano. You'll measure exactly how slow.

**Augmentation.** Randomly altering training images so the model sees more variety. Overhead imagery has no natural "up", so vertical flips are valid here, whereas for a street camera they would teach nonsense. Brightness jitter imitates glare and dusk. Augmentation helps only when it imitates variation the real world will actually contain.

### Do

1. Train at high resolution: `python training/train.py configs/train/sds_y11s_1280.yaml`
2. Train with augmentation: `python training/train.py configs/train/sds_y11s_1280_aug.yaml`
3. Evaluate both on **validation** with `evaluate_coco.py` (`--imgsz 1280`).
4. SAHI on your best 640 model:
   ```bash
   python training/evaluate_coco.py --model runs/sds/y11s_640/weights/best.pt \
     --gt datasets/seadronessee_yolo/coco_val.json --sahi --slice 640 --tag y11s_640_sahi
   ```
   Compare AP_small and `ms_per_image_median` against the non-sliced run.
5. Optional, if time allows: try slice sizes of 512 and 800, or overlap 0.1 vs 0.3.
6. Write a short paragraph per experiment in the log: hypothesis, result, conclusion.
7. Choose your final models now, using validation only. Then, and only then, run each chosen model once on test (`coco_test.json`) for the README.

### Checkpoint

A comparison table: baseline vs high resolution vs augmentation vs SAHI, showing AP, AP_small and time per image, plus the final test numbers for your chosen models.

### Viva

1. Why change only one thing per experiment?
2. Why does higher resolution help small objects, and what does it cost?
3. When would you choose SAHI in a real search-and-rescue system, and when would you avoid it?
4. Give an example of an augmentation that would hurt this task.
5. Why did you choose your final model on validation and not on test?

---

## Lab 3 — Optimising for the Jetson (weeks 2–3)

### Why

A PyTorch model is flexible but slow on embedded hardware. **TensorRT** compiles the network for the specific GPU: it fuses layers, picks the fastest kernels and lowers numerical precision.

**Precision.** FP32 is the full 32-bit float. FP16 halves memory and is usually nearly free in accuracy on Jetson GPUs. INT8 uses 8-bit integers: faster again and more energy-efficient, but the model must learn how to map its float ranges onto 256 levels. That mapping is **calibration**: TensorRT runs a few hundred representative images through the network and records the range of each layer's values. Calibrate with images from the validation set, never the test set, and make sure they include the hard cases (small objects, glare). A bad calibration set is the usual reason INT8 loses accuracy.

**Engines are hardware-specific.** An engine built on your desktop will not run on the Jetson. Build engines on the Jetson itself.

**Measuring properly.**
- Warm-up: the first inferences include initialisation; never time them.
- Load images into memory first, so you time the model, not the disk.
- Report percentiles, not just the average. A system that averages 30 ms but spikes to 200 ms once a second will drop frames; p99 latency exposes that.
- Fix the power mode and lock clocks (`sudo jetson_clocks`) so runs are comparable. Log the temperature: if it climbs to throttling levels, later runs get slower and your table lies.
- Energy per frame (joules) = average power (watts) × time per frame (seconds). On a battery-powered drone this is the number that decides flight time.

### Do

1. Copy your chosen `best.pt` files to the Jetson (`scp`). Don't commit weights to git; publish them as GitHub release assets later if the licences allow.
2. Build engines (inside the container, on the Jetson):
   ```bash
   python export/export_trt.py --weights weights/sds_y11s_640.pt --data configs/datasets/seadronessee.yaml --imgsz 640
   python export/export_trt.py --weights weights/sds_y11s_1280.pt --data configs/datasets/seadronessee.yaml --imgsz 1280
   ```
   The dataset must also be on the Jetson for INT8 calibration and accuracy measurement. A copy of val and test is enough; re-run the converter there or copy the folders and edit the `path:` line of the YAML.
3. Prepare the board: `sudo nvpmodel -q` (write it down), `sudo jetson_clocks`, check the fan runs.
4. Benchmark every variant, including the `.pt` file as a reference:
   ```bash
   python benchmarks/benchmark.py --model weights/sds_y11s_640.pt --data configs/datasets/seadronessee.yaml --imgsz 640 --tag y11s_640_pytorch
   python benchmarks/benchmark.py --model weights/sds_y11s_640_640_fp16.engine --data configs/datasets/seadronessee.yaml --imgsz 640 --tag y11s_640_fp16
   # ... fp32, int8, and the 1280 variants
   ```
5. Run one variant twice, an hour apart. If the numbers differ by more than a few percent, your setup isn't stable; find out why before going on.
6. `python benchmarks/make_tables.py` and paste the result into the README.

### No Jetson yet? INT8 on CPU first

TensorRT needs NVIDIA hardware, but the INT8 question (how much accuracy does quantisation cost?)
can be answered on any CPU with ONNX Runtime:

```bash
python export/quantize_onnx.py --weights weights/ft_y11n_ep3.pt --imgsz 1280 \
  --calib-dir datasets/visdrone_yolo/images/train --n-calib 128
python training/evaluate_zeroshot.py --model weights/ft_y11n_ep3_1280_int8_headfp32.onnx --class-map visdrone \
  --data configs/datasets/visdrone.yaml --imgsz 1280 --tag onnx_int8_headfp32_1280 --out benchmarks/results/quantization.csv
```

It builds two INT8 models. One quantises everything; the other keeps the detection head's
post-processing in FP32. YOLO's last layer concatenates box coordinates (0–1280) with class
scores (0–1) into one tensor, and a single 8-bit range cannot hold both. Compare the two and you
will see the failure for yourself. That is the kind of thing to know before the Jetson arrives.

### Checkpoint

The main results table: one row per model × resolution × precision, with mAP, FPS, p50 and p99 latency, watts, joules per frame and peak temperature, all measured on the Jetson in one documented power mode. This table is the most valuable thing in the repo.

### Viva

1. What does TensorRT do to a network, at a high level?
2. What is INT8 calibration and what happens if the calibration images are unrepresentative?
3. Why report p99 latency as well as the mean?
4. Your INT8 model is 1.6× faster but loses 3 points of AP_small. Which would you deploy for search and rescue, and why? (There is no single right answer; the reasoning is what's marked.)
5. Why is energy per frame a better metric than FPS for a drone?
6. Pre-processing takes 8 ms and inference 12 ms. What would you optimise first?

---

## Lab 4 — Building the system (week 3)

### Why

A model is not a product. A detector outputs boxes per frame, with no memory. A system needs **identity over time**, **a standard interface** to other software, and **a human-facing display**.

**Tracking.** ByteTrack links detections across frames using predicted motion (a Kalman filter) and box overlap. Its trick: it also uses *low-confidence* detections in a second matching round, so an object that briefly fades (a swimmer behind a wave) keeps its ID instead of being dropped and re-born. BoT-SORT adds camera-motion compensation and appearance features, useful when the drone itself moves. Try both.

**Alert debouncing.** A single-frame false positive on a wave crest must not trigger a rescue alert. `AlertManager` fires only after a track has persisted for N frames. That is a deliberate trade: a little delay for far fewer false alarms. Be ready to discuss what N should be and who should decide it (the operators, not you).

**ROS 2.** The robotics middleware: programs (nodes) exchange typed messages on named topics. Publishing the standard `vision_msgs/Detection2DArray` means any other ROS tool can consume your detections without knowing your code. `rosbag` records topics so you can replay a run exactly, which is how robotics teams debug.

**Geolocation.** `esa/geo.py` turns a pixel into latitude and longitude assuming the camera looks straight down over flat ground from a known position and height. It's a deliberately simple model; its assumptions go in Limitations. A real system would intersect the camera ray with the ground using the drone's telemetry and gimbal angles.

### Do

1. **Without ROS first** (works on your laptop too):
   ```bash
   python tools/detect_video.py --model runs/sds/y11s_640/weights/best.pt --source some_sea_clip.mp4 \
     --out outputs/run1.jsonl --save-video outputs/run1.mp4 --alt 60 --hfov 70 --alert swimmer
   ```
   Watch `run1.mp4`. Do track IDs stay stable? Where do they switch? Try `--tracker botsort.yaml` and compare.
2. **Map, in replay mode:**
   ```bash
   python web/backend/server.py --replay outputs/run1.jsonl --rate 10
   cd web/frontend && npm install && npm run dev
   ```
   Open the printed URL. You should see detections, trails and alerts on the map.
3. **Controlled tracking test (no video needed).** Real multi-object-tracking data with per-object
   identities is large and hard to get. Instead, fly a simulated camera over one fully labelled
   frame: every object's identity and ground position is then known exactly, so ID switches and
   position error can be scored precisely, and camera speed becomes a dial.
   ```bash
   python tools/sim_flight.py --image <labelled val image> --speed 24 --frames 120 --out outputs/flight_fast
   python tools/detect_video.py --model <weights> --source outputs/flight_fast/frames --imgsz 1280 --conf 0.1 \
     --tracker botsort.yaml --telemetry outputs/flight_fast/telemetry.jsonl --out outputs/track_fast_botsort.jsonl
   python benchmarks/tracking_eval.py --flight outputs/flight_fast --run outputs/track_fast_botsort.jsonl --tag fast_botsort
   python benchmarks/geo_sensitivity.py --flight outputs/flight_fast
   ```
   Run it with `bytetrack.yaml`, `configs/trackers/botsort_nogmc.yaml` and `botsort.yaml`. The last two
   differ in one setting, camera-motion compensation, so the comparison isolates it. Know what this
   test cannot tell you: objects don't move on their own, and the source frame is treated as a
   straight-down photo.
4. **ROS 2 on the Jetson** (Humble, inside a ROS container):
   ```bash
   sudo apt install -y ros-humble-vision-msgs        # if not already in the image
   cd ros2_ws && colcon build && source install/setup.bash
   ros2 launch esa_ros esa.launch.py model:=/workspace/weights/sds_y11s_640_640_fp16.engine source:=/workspace/clip.mp4
   ros2 topic hz /detections          # in another terminal: how fast is it really publishing?
   python web/backend/server.py --ros # bridge to the map
   ros2 bag record /detections /tracks_geo /alerts
   ```
   Check the node end to end before adding a camera: `bash ros2_ws/test_node.sh` runs it on sample images
   and verifies every topic (the same test runs in CI on every ROS change).
5. Replay the bag (`ros2 bag play <folder>`) with the bridge running and confirm the map shows the same run.
6. Optional: DeepStream for several cameras at once. Only do this if everything else is done; it's a separate learning curve.

### Checkpoint

The full chain runs live on the Jetson: camera or video in, TensorRT inference, tracking, ROS topics, map in the browser with alerts. A short screen recording of it.

### Viva

1. Why do you need a tracker when you already have a detector?
2. How does ByteTrack recover objects that drop below the confidence threshold?
3. Why publish a standard ROS message type instead of your own JSON?
4. What are three assumptions behind your pixel-to-GPS conversion, and how would each cause error?
5. How would you choose the alert persistence threshold, and who should own that choice?

---

## Lab 5 — Testing in realistic conditions (week 4)

### Why

Benchmarks measure performance on data that resembles the training data. Real deployments fail on what the dataset didn't contain: fog, night, rain on the lens, low sun glare, motion blur, unusual boats. This lab is about finding where your system breaks, and documenting it honestly. Engineers who can say "here is where it fails and why" are trusted; engineers who only show successes are not.

### Do

1. Collect unseen footage: public-domain or Creative Commons drone videos of sea and roads, recorded with their source and licence. Group them by condition: clear, glare, fog/haze, dusk/night, motion blur.
2. Run `tools/detect_video.py` on each and save the annotated video.
3. For each condition, note: missed people, false alarms (what triggered them), ID switches, FPS.
4. Pick 6–10 representative failure frames, save them to `docs/figures/failures/`, and write one or two sentences per frame in `docs/LIMITATIONS.md`: what happened and the likely cause.
5. **What this repo did** (`scripts/sea_experiments.sh`, README §9): SeaDronesSee-MOT clips come with the drone's own
   log, so they test tracking *and* geolocation on real footage. Two lessons worth copying: check whether the official
   split shares videos between train and val (here it did, and inflated swimmer AP by ~50%), and when ground-truth
   positions don't exist, use a physical check (a swimmer moves ≲ 1 m/s) instead of giving up on measuring.
6. If you fly your own test: register as an operator with Traficom, follow the open-category rules (including not flying over uninvolved people), check local airspace restrictions, film only willing friends, and keep the drone over water or open land. Don't fly anything near your employer's work.

### Checkpoint

`docs/LIMITATIONS.md` filled with real failure examples, each with an image and a plausible cause.

### Viva

1. Name your system's three worst failure modes and the likely cause of each.
2. Which failure would be most dangerous in a real rescue, a miss or a false alarm? How does your design handle it?
3. What data would you collect next to fix your worst failure?

---

## Lab 6 — Publishing (end of week 4)

### Why

The repo is a document for a reader who will spend five minutes on it. The README must answer, in order: what is it, does it work (one GIF), how well (the table), how do I run it, where does it fail.

### Do

1. Fill in the README: results tables, the architecture diagram, setup steps you actually used, versions (JetPack, TensorRT, Ultralytics, Python), dataset licences and citations, Limitations link.
2. Fill in one model card per published model (`docs/MODEL_CARD_TEMPLATE.md`).
3. Record a 1–2 minute demo video: the Jetson running (show `jtop` with power), the map updating, an alert firing. Put a short GIF at the top of the README.
4. Tag a release `v1.0`. If dataset licences allow it, attach the weights as release assets.
5. Write the LinkedIn post: the problem in one sentence, the video, three numbers (for example AP_small, FPS, joules per frame), one honest limitation, the link.
6. Rehearse with `docs/INTERVIEW_PREP.md` until you can do the 60-second and 5-minute versions without notes.

### Checkpoint

A stranger can clone the repo, read the README and understand what you built, how well it works and where it doesn't, in five minutes.
