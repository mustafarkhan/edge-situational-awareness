# Samples

`sea_DJI_0055/`: the first 30 frames (1 s) of SeaDronesSee-MOT clip DJI_0055, resized to 1280 px, with the
drone's own log for those frames (`telemetry.jsonl`: GPS, height, gimbal pitch and heading; field of view
assumed 65.5°). SeaDronesSee is CC0 1.0 (Varga et al., WACV 2022). This clip was never used for training.
Enough to run the whole pipeline (detect, track, geolocate, alert, map) without downloading any dataset.
