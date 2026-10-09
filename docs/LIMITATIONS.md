# Limitations

Fill this in during Lab 5 with real examples. Each entry: an image, what happened, likely cause, what would fix it.

## Known by design

- **Geolocation is approximate.** Positions assume a camera pointing straight down over flat ground, with exactly known drone position, altitude and heading, and no lens distortion. Any camera tilt, altitude error or terrain makes positions wrong, increasingly so towards the image edges. A real system would use drone telemetry and gimbal attitude.
- **Training data is from public datasets** with their own sensors, altitudes, sea states and regions. Performance on other cameras, Nordic sea conditions, or winter scenes is not established.
- **RGB only.** No thermal model; night performance is expected to be poor.
- **Measured on CPU so far.** All published speeds come from a 2-vCPU cloud VM without a GPU. They are
  useful for comparing configurations against each other, not as Jetson performance.
- **Short fine-tune.** The fine-tuned model saw 2 CPU epochs at 640 px. The learning-rate warm-up (3
  epochs by default) spans the whole run, so the model is under-trained; numbers are a lower bound on
  what fine-tuning gives, not a tuned result.
- **Ten VisDrone classes mapped to three.** COCO and VisDrone classes don't match one-to-one, so all
  comparisons use person / two-wheeler / vehicle. Tricycles are excluded from the ground truth;
  detections on them count as false positives.
- **Demo footage is not truly unseen scenery.** Sequences 9999951 and 9999955 were removed from training,
  but other VisDrone sequences may show the same city and camera.
- **Not a safety-certified system.** Research software, not for operational rescue use.

## Observed failures

### <condition>: <short title>
![](figures/failures/example.jpg)
What happened:
Likely cause:
Possible fix:
