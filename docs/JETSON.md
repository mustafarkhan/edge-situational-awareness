# Running on an NVIDIA Jetson

Everything in this repo was measured on a CPU. This page gets the same pipeline onto a Jetson and fills the
missing table (speed, watts, joules per frame, temperature, and accuracy after TensorRT FP16/INT8) with one
command.

## What to buy

| item | why | notes |
|---|---|---|
| **Jetson Orin Nano Super Developer Kit (8 GB)** | the board this project targets; 67 TOPS | list price was raised from $249 to $399 in 2026 ([JetsonHacks, Aug 2026](https://jetsonhacks.com/2026/08/26/jetsonhacks-august-2026-newsletter/)); check EU resellers |
| NVMe M.2 2280 SSD, 256 GB+ | engines, datasets, and much faster than a microSD card | install JetPack onto it |
| microSD card, 64 GB (UHS-I) | only if you flash with the SD-card image | |
| USB-C or DC power supply that came with the kit | power readings are only comparable on the kit's own supply | |
| (optional) USB power meter | cross-check the on-board `tegrastats` power reading | |

The developer kit already has a fan; keep it on for the measurements, which run the board flat out for several
minutes.

## Set up (once, about an hour)

1. Flash **JetPack 6.2.x** (the current line for Orin; JetPack 7 targets Jetson Thor) with NVIDIA SDK Manager or the
   SD-card image, following NVIDIA's *Jetson Orin Nano Developer Kit Getting Started* guide. Choose the **MAXN SUPER**
   power mode afterwards: `sudo nvpmodel -m 2` (check the mode list with `sudo nvpmodel -q --verbose`).
2. Install the container tooling and start a PyTorch container matching your JetPack:
   ```bash
   git clone https://github.com/dusty-nv/jetson-containers && bash jetson-containers/install.sh
   git clone https://github.com/mustafarkhan/edge-situational-awareness
   jetson-containers run -v $(pwd)/edge-situational-awareness:/workspace $(autotag l4t-pytorch)
   ```
3. Inside the container:
   ```bash
   cd /workspace && pip install -r requirements-jetson.txt && pip install -e .
   ```

## Measure (one command, about 20 minutes)

```bash
bash benchmarks/run_on_jetson.sh            # no dataset needed
SDS=/data/seadronessee_yolo bash benchmarks/run_on_jetson.sh   # + accuracy on the 1,200 test images
ROS=1 bash benchmarks/run_on_jetson.sh      # + the ROS 2 node end to end (needs a ROS 2 Humble container)
```

It builds TensorRT FP16 and INT8 engines at 640 and 1280 px for the sea model. INT8 is calibrated on the 96
SeaDronesSee **training** images in `samples/calib_sds`, never on evaluation images. Each engine and the PyTorch
baseline are then measured in two ways:

- **model only:** frames per second, p50/p95/p99 latency, average power, joules per frame and peak temperature,
  recorded by `tegrastats` over 300 frames of the real clip in `samples/`;
- **whole pipeline:** detection, tracking, geolocation and alerts on the same clip, giving the median frame rate.

The results are in `benchmarks/results/jetson.csv` and `benchmarks/results/jetson_report.md` (plus
`jetson_sea_accuracy.csv` if you set `SDS`).

## What to look for

- **INT8 accuracy.** On the CPU (ONNX Runtime), quantising the whole model gave zero detections until the detection
  head was kept in FP32. Does TensorRT's INT8 show the same problem? The `SDS` run answers this.
- **How much speed is needed.** Search planning (README section 10) shows that for *coverage* about 1 processed
  frame per second is enough at 10 m/s. Extra speed buys tracking and alert latency, not detection probability.
  The interesting Jetson number is therefore **joules per frame at the frame rate the mission needs**, not peak
  frames per second.
- **Thermals.** Whether the temperature in a closed drone enclosure stays below the throttling point. Note the
  ambient temperature in the report.

Once you have the CSV, recompute the planning table with the measured frame rates:
`python benchmarks/search_planning.py --fps <measured_fps> 1`.
