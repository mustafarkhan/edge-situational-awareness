# Training on a free Kaggle GPU

The first models in this repo were trained for 3-5 epochs on a 2-vCPU machine. This page runs proper training (60-120
epochs) on Kaggle's free GPU with one notebook: [`kaggle/esa_gpu_training.ipynb`](../kaggle/esa_gpu_training.ipynb).
It downloads the three datasets itself, uses exactly the same splits as the repo, and saves `esa_gpu_results.zip`
(weights and training logs, about 30 MB). Scoring happens outside Kaggle with the repo's own evaluation scripts, so
the new numbers are comparable with every other number in the README.

**First full run (2026-10-16, 3 h on 2× T4):** sea AP 0.266 → 0.325, thermal 0.445 → 0.512, land 0.253 → 0.336, all on
the same held-out sets as before (`scripts/gpu_scores.sh`, README §2, §9, §13).

## Once: set up Kaggle (10 minutes)

1. Create an account at kaggle.com.
2. **Verify your phone number** (Settings → Phone verification). Kaggle only gives GPUs and internet access to verified
   accounts.

## Test run (about 30 minutes)

1. kaggle.com → **Code → New Notebook**, then **File → Import Notebook** and upload `esa_gpu_training.ipynb`.
2. In the right-hand panel, under **Session options**: Accelerator **GPU T4 x2** (or P100), **Internet: on**.
3. The second cell already says `QUICK = 1` (a short test). **Run All**.
4. The last cell should list the runs and an `esa_gpu_results.zip`. If the sea download fails, the log says so. Copy
   the "Object Detection v2" link from <https://seadronessee.cs.uni-tuebingen.de/dataset> into `SDS_SHARE` and
   run again.

## Full run (6-10 hours, runs without you)

1. Set `QUICK = 0` in the second cell.
2. **Save Version → Save & Run All (Commit) → Save.** Kaggle runs it in the background for up to 12 hours; you can close
   the tab.
3. When it finishes (the version page shows "Successful"), open it, go to the **Output** tab and download
   `esa_gpu_results.zip`.
4. Unzip it into the repository and score each model on its held-out test set with the same evaluation scripts as
   every other number in the README (README, "Reproduce"). Then update the README, model cards and figures.

## What it trains

| stage | data | model | epochs (early stop on val) | approx. T4 time |
|---|---|---|---|---|
| `sea` | SeaDronesSee subset (2,500 train / 400 val) | YOLO11n, 640 px | 100 | 1-1.5 h |
| `thermal` | HIT-UAV, flight-disjoint (1,902 / 183) | YOLO11n, 640 px | 120 | 0.5-1 h |
| `land` | VisDrone train minus the 2 held-out sequences (5,703 / 548) | YOLO11n, 640 px | 80 | 2-3 h |
| `sea1280` | SeaDronesSee, trained at 1280 px (tiny swimmers) | YOLO11n, 1280 px | 60 | 2-3 h |

### Round 2: bigger model, YOLO26, thermal at 1280 px

Same data, splits, seed and recipe as round 1; one thing changes per run, so each result answers one question.
The notebook's default `STAGES` is this round.

| stage | run | the one change vs round 1 | approx. T4 time |
|---|---|---|---|
| `thermal_y11s` | `hit_y11s_640` | YOLO11s (3.6x the parameters) | 1 h |
| `sea1280_y11s` | `sds_y11s_1280` | YOLO11s | 2.5-3 h |
| `sea1280_y26n` | `sds_y26n_1280` | YOLO26n instead of YOLO11n | 1 h |
| `thermal_y26n` | `hit_y26n_640` | YOLO26n instead of YOLO11n | 0.5 h |
| `thermal1280` | `hit_y11n_1280` | thermal frames upscaled to 1280 px input | 1.5-2 h |

A bigger model is slower on the Jetson, so a gain only counts if it is worth its cost in milliseconds; that is
measured when the models are scored.

Stages run in that order; if fewer than 45 minutes of the 10.5-hour budget remain, later stages are skipped. Each
finished stage is added to the zip immediately, so a session that stops early still leaves results. Kaggle's free
GPU quota is about 30 hours a week (Kaggle does not publish an exact figure), and one full run uses about a third of it.

The same script runs on any rented GPU with internet (RunPod, Verda, ...):
`OUT=~/results bash scripts/kaggle/run_kaggle.sh`.
