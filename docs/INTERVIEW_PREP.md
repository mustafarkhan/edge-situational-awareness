# Interview Preparation

The rule that overrides everything here: **only say numbers you measured, and know how you measured them.** Interviewers who know edge AI will ask one follow-up question about any number you quote. If the answer is "I'm not sure how that was computed", the whole project loses credibility.

## The 60-second version (all numbers measured)

> I built an open-source edge AI pipeline that detects, tracks and geolocates people and vehicles in drone
> video, aimed at an NVIDIA Jetson.
>
> I started with the data. In VisDrone, the median object is 11 pixels across at YOLO's default input,
> so I measured small-object AP instead of headline mAP. A pretrained YOLO11n scores 0.04. Higher input
> resolution and a short fine-tune took it to 0.18, better than a model three times larger in half the time.
>
> Then the system around it. In a controlled flight test where every object's true identity is known,
> camera-motion compensation raised identity accuracy from 0.33 to 0.57 when the drone moves fast. An error
> budget showed that ignoring a 10-degree gimbal tilt costs 12 metres of position accuracy, more than GPS
> error. And INT8 quantisation produced zero detections until I kept the detection head in FP32; then it cost
> 5% accuracy for a 20% speed-up and a model a third the size.
>
> I documented where it fails, mainly tiny objects and false alarms at night. Next is TensorRT on the Jetson.

## Stories to rehearse (symptom → cause → fix → how you verified)

1. **AP = -1.0.** Overall AP came out as -1 while other metrics looked fine. pycocotools' summary hard-codes
   100 detections per image; dense VisDrone scenes need 500. Computed AP from the precision array; regression test.
2. **The tracker that hid detections.** Tracking on held-out VisDrone frames gave a median of 1 detection per frame
   against 20–100 objects. The frames were seconds apart, and trackers only output confirmed tracks. Detection-only
   mode restored 8–76 per frame. Lesson: check the data's time structure before choosing an algorithm.
3. **INT8 with zero detections.** Predicted from the architecture (box coordinates and scores share one tensor),
   confirmed by experiment, fixed by excluding 48 post-processing nodes from quantisation.
4. **Alerts nobody received.** The ROS 2 end-to-end test in CI got detections but no alerts: every alert fired in the
   first frames, before the subscriber had connected. A ground station reconnecting mid-mission would have the same
   problem. Made `/alerts` latched (transient-local QoS). A test that fails for a reason that also matters in the field.
5. **No tracking data available.** Built a controlled test instead, where the ground truth is exact, and was explicit
   about what it can't show.

## The 5-minute walkthrough

Walk the pipeline left to right, and at each stage say one decision and why.

1. **Problem.** Search and rescue needs to find small objects over large areas, on a device that runs on battery. So accuracy on small objects and energy matter more than headline mAP.
2. **Data.** Public datasets only. Explain the block split and why a random split would have leaked near-duplicate video frames. Show the object-size table.
3. **Model.** YOLO11 baselines, one change per experiment. Show the comparison table: what resolution and SAHI bought in AP_small, and what they cost in time.
4. **Optimisation.** TensorRT, precision levels, INT8 calibration from validation data. Show the Jetson table. Name the configuration you would deploy and why.
5. **System.** Tracking (why ByteTrack), alert debouncing (why persistence), ROS 2 with standard messages, rosbag for replay, WebSocket to a React map.
6. **Limits.** Two or three failure examples with images. What you'd do next.

## Tailoring by role

- **Robotics / deployment roles:** spend most time on Lab 3 and Lab 4: measuring on hardware, power modes, thermals, ROS 2 integration, rosbag replay, failure analysis.
- **AI / ML roles:** spend most time on Labs 1–2: data analysis, leakage, evaluation rigour, small-object AP, the experiment log, INT8 accuracy loss.
- **Frontend / full-stack roles:** the React map, the WebSocket bridge, dropping stale frames rather than queuing them, handling reconnects, and presenting uncertainty to an operator (the geolocation disclaimer on screen).
- **Drone / UAV roles:** energy per frame, latency budgets, the nadir-camera geolocation assumptions and how telemetry plus gimbal attitude would fix them.

## Questions to expect, and what a good answer contains

Prepare your own wording. These are the points your answer should hit.

**"Why did you choose YOLO11 / Ultralytics?"** Mature, well-supported export to TensorRT, built-in tracking, fast to iterate. Trade-off: AGPL licence, so commercial use needs a different licence or model. Mention you know newer versions exist and the pipeline is model-agnostic.

**"How do you know your results aren't overfitted?"** Separate validation and test; decisions on validation only; test touched once per final model; block split to stop video-frame leakage; tested on unseen footage in Lab 5.

**"Explain mAP."** Precision-recall curve per class, area under it, IoU thresholds, mean over classes. Then say why you also report AP_small.

**"What does INT8 do and what can go wrong?"** 8-bit quantisation with calibration to find value ranges; unrepresentative calibration data clips important values and costs accuracy, often on exactly the small or rare objects. You measured the drop and decided.

**"Your FPS is X. Is that real time?"** Depends on the requirement: compare with camera frame rate and the speed at which the scene changes. Mention p99 latency, and that pre- and post-processing are part of the budget.

**"What would you do with another month?"** Pick from: proper geolocation with MAVLink telemetry and gimbal angles; thermal camera and a thermal model for night; a larger and harder test set; multi-camera DeepStream pipeline; hard-negative mining for wave-crest false alarms; INT8 quantisation-aware training if calibration loses too much.

**"What was the hardest bug?"** Have a real story from your log, with the symptom, how you found it and how you verified the fix. Write it down as soon as it happens; you won't remember the details later.

**"Did you use anything from your job?"** No: own hardware, own time, public data, checked the contract. Say it calmly and briefly.

## Demo checklist for an interview

- The demo video plays offline (don't rely on venue Wi-Fi).
- The map can run in replay mode from a JSONL file on your laptop, so you can show it live without the Jetson.
- One slide or README section with the main table, open in a tab.
- One failure example ready to show. Showing a failure confidently is memorable.
