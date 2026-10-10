# Limitations

What this project does not show yet, and where the model fails. Every failure below was measured
(`data/failure_analysis.py`, `benchmarks/geo_sensitivity.py`) or seen in the annotated images.

## Scope of the measurements

- **CPU only so far.** All speeds come from a 2-vCPU cloud VM without a GPU; repeated runs vary by about ±10%.
  They compare configurations; they are not Jetson performance. The Jetson table is still empty.
- **Short fine-tune.** 3 CPU epochs at 640 px; the learning-rate warm-up spans the whole run. Treat the numbers
  as a lower bound on what fine-tuning gives.
- **One dataset.** VisDrone (Chinese cities, specific cameras and altitudes). Performance on Nordic scenes,
  sea, winter or other sensors is unknown. The planned maritime data (SeaDronesSee) is not in yet.
- **Ten classes mapped to three** (person / two-wheeler / vehicle) so pretrained and fine-tuned models can be
  compared. Tricycles are left out of that ground truth.
- **Tracking was tested on a controlled simulation**, not real continuous video: a camera flown over one still,
  labelled frame. Objects don't move on their own, there is no blur or lighting change, and the frame is treated
  as a straight-down photo. It measures the tracker's response to camera motion, and nothing else.
- **ROS 2 tested in CI, not on the Jetson.** The node builds and runs end to end on ROS 2 Humble (GitHub runner,
  CPU, sample images); it has not yet run on the Jetson with a camera.
- **Demo footage is held out, not foreign.** Sequences 9999951 and 9999955 were removed before training, but
  other VisDrone sequences may show the same city.
- **Not a safety-certified system.** Research and portfolio work, not for operational rescue use.

## Geolocation

The pipeline assumes a straight-down camera over flat ground with exact telemetry. Measured effect of
breaking each assumption (drone at 57–69 m, 70° lens, median over 7,634 objects):

| assumption broken | error |
|---|---|
| GPS off by 5 m | 5.0 m |
| altitude reading 10% high | 2.9 m |
| compass off by 5° | 2.5 m |
| **gimbal tilted 10°, treated as straight down** | **11.8 m** |
| gimbal tilted 20° | 24.2 m |

Fix: read gimbal attitude from telemetry (MAVLink) and intersect the camera ray with the ground (and terrain).

## Observed failures

Green = found, red = missed, yellow = false positive. Fine-tuned YOLO11n, 1280 px, confidence 0.25.

### Tiny objects: under 8 px, 1 in 10 is found
Recall by object size at model input: 10% (< 8 px), 26% (8–16), 49% (16–32), 73% (32–64), 87% (≥ 64).
Likely fix: train at 1280 px (training here was at 640), a P2 detection head for small strides, or tiling on 4K input.

### Low, oblique view down a long road: recall 13%
![](figures/failures/0000153_01801_d_0000001.jpg)
Most of the 55 objects sit far down the road and shrink to a few pixels; only the nearest cars are found.
Likely cause: perspective makes object size vary with distance far more than in near-vertical views.

### Near-ground view: large, close objects missed, recall 13%
![](figures/failures/0000021_00000_d_0000001.jpg)
People and scooters close to the camera are missed even though they are large. This matches the drop in
large-object AP after fine-tuning (0.58 → 0.38): three short epochs on mostly high-altitude imagery made the
model forget how close-up objects look. Likely fix: longer training, keep some close-range data in the mix.

### Dense, occluded scene (toll plaza): recall 16%
![](figures/failures/0000273_05601_d_0000466.jpg)
Vehicles packed under a canopy and partly hidden by booths. Over the whole val set, recall is 63% for
unoccluded objects, 42% partially occluded, 23% heavily occluded.

### Night: the model invents objects
![](figures/failures/0000055_00714_d_0000110.jpg)
On 17 dusk/night images recall holds (48.7% vs 49.5% in daylight) but precision falls from 70% to 44%:
duplicate boxes on the same person and boxes on dark blobs. 101 false positives in this one image. Likely fix:
more night training data, a thermal camera, or a higher confidence threshold at night.

### Two-wheelers and people
Recall: vehicles 74%, people 34%, two-wheelers 23%. People and bikes are smaller and more often occluded.
