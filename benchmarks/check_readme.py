"""Check that the numbers quoted in the README match the results files that produced them.

Every claim names a README line (by a unique piece of its text) and the values that line must contain,
each read from a CSV in benchmarks/results/. Runs in CI, so the README cannot silently drift away from
the data after a re-run.

  python benchmarks/check_readme.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = REPO / "benchmarks" / "results"


def v(csv: str, where: dict, col: str, fmt: str = "{:.3f}", scale: float = 1.0) -> str:
    df = pd.read_csv(R / csv)
    for k, val in where.items():
        df = df[df[k] == val]
    df = df.drop_duplicates(keep="last")
    if len(df) == 0:
        raise LookupError(f"{csv}: no row {where}")
    return fmt.format(float(df.iloc[-1][col]) * scale)


Z = "zeroshot.csv"
SEA, SEA2 = "seadronessee.csv", "seadronessee_two_class.csv"
pct = "{:.0f}%"

CLAIMS: list[tuple[str, list[str]]] = [
    # README line (unique text it contains), values it must contain
    ("| YOLO11n | COCO | 640 | 0.200", [v(Z, {"tag": "y11n_640"}, c) for c in ("AP", "AP_small", "AP_large")]),
    ("| YOLO11n | COCO | 1280 |", [v(Z, {"tag": "y11n_1280"}, c) for c in ("AP50", "AP", "AP_small")]),
    ("| YOLO11s | COCO | 640 |", [v(Z, {"tag": "y11s_640"}, c) for c in ("AP50", "AP", "AP_small")]),
    ("| YOLO11s | COCO | 1280 |", [v(Z, {"tag": "y11s_1280"}, c) for c in ("AP50", "AP", "AP_small")]),
    ("| YOLO11n + SAHI", [v(Z, {"tag": "y11n_640_sahi640"}, c) for c in ("AP50", "AP", "AP_small")]),
    (
        "| **YOLO11n** | **fine-tuned, 3 epochs** | **1280**",
        [v(Z, {"tag": "ft_y11n_1280"}, c) for c in ("AP50", "AP", "AP_small", "AP_large")],
    ),
    ("| YOLO11n | fine-tuned, 3 epochs | 640 |", [v(Z, {"tag": "ft_y11n_640"}, c) for c in ("AP50", "AP", "AP_small")]),
    # tracking (simulated flight)
    ("| fast | BoT-SORT, motion compensation **on**", [v("tracking.csv", {"tag": "fast_botsort"}, "IDF1")]),
    ("| fast (2.4 m/frame) | ByteTrack", [v("tracking.csv", {"tag": "fast_bytetrack"}, "IDF1")]),
    # quantisation
    ("| ONNX INT8, head post-processing kept FP32", [v("quantization.csv", {"tag": "onnx_int8_headfp32_1280"}, "AP")]),
    ("| ONNX FP32 |", [v("quantization.csv", {"tag": "onnx_fp32_1280"}, c) for c in ("AP", "AP_small")]),
    # geolocation budget
    (
        "| **gimbal tilt ignored, 5° / 10° / 20°**",
        [
            v("geo_sensitivity.csv", {"error_source": "gimbal tilt (ignored)", "level": lv}, "median_m", "{:.1f} m")
            for lv in ("5 deg", "10 deg", "20 deg")
        ],
    ),
    # sea detection
    (
        "| YOLO11n COCO (2 classes) | 640 |",
        [v(SEA2, {"tag": "coco_y11n_640"}, c) for c in ("AP", "AP_small", "AP_swimmer", "AP_boat")],
    ),
    (
        "| YOLO11n COCO (2 classes) | 1280 |",
        [v(SEA2, {"tag": "coco_y11n_1280"}, c) for c in ("AP", "AP_small", "AP_swimmer", "AP_boat")],
    ),
    (
        "| YOLO11n fine-tuned 4 epochs, 5 classes | 640 |",
        [v(SEA, {"tag": "ft_y11n_640"}, c) for c in ("AP", "AP_small", "AP_swimmer", "AP_boat")],
    ),
    (
        "| **YOLO11n fine-tuned 4 epochs, 5 classes** | **1280** |",
        [v(SEA, {"tag": "ft_y11n_1280"}, c) for c in ("AP", "AP_small", "AP_swimmer", "AP_boat")],
    ),
    # leakage
    ("| COCO model (control) |", [v(SEA2, {"tag": t}, "AP") for t in ("coco_y11n_1280", "coco_y11n_1280_seen")]),
    ("| fine-tuned | 0.", [v(SEA2, {"tag": t}, "AP") for t in ("ft_y11n_1280", "ft_y11n_1280_seen")]),
    # real tracking
    *[
        (
            f"| {clip} | ",
            [
                v("tracking_sea.csv", {"tag": f"sea_{clip}_{trk}"}, c, f)
                for trk in ("bytetrack", "botsort")
                for c, f in (("IDF1", "{:.3f}"), ("id_switches", "{:.0f}"))
            ],
        )
        for clip in ("DJI_0055", "DJI_0059", "DJI_0057", "DJI_0001_d3")
    ],
    # real geolocation
    *[
        (
            f"| {clip} | ",
            [
                v(
                    "geo_real.csv",
                    {"clip": clip, "object": "swimmer", "model": m},
                    "apparent_speed_median_mps",
                    "{:.2f}",
                )
                for m in ("nadir", "tilt-aware")
            ],
        )
        for clip in ("DJI_0055", "DJI_0057", "DJI_0001_d3", "DJI_0059")
    ],
]


def main() -> int:
    lines = (REPO / "README.md").read_text().splitlines()
    bad = 0
    for anchor, values in CLAIMS:
        hits = [ln for ln in lines if anchor in ln]
        if not hits:
            print(f"MISSING  README line containing {anchor!r}")
            bad += 1
            continue
        # a line passes if one README line with this anchor carries all of its values
        if not any(all(val in ln for val in values) for ln in hits):
            print(f"MISMATCH {anchor!r}: expected {values}\n         README: {hits[0].strip()}")
            for h in hits[1:]:
                print(f"         README: {h.strip()}")
            bad += 1
    n = sum(len(v_) for _, v_ in CLAIMS)
    print(f"{len(CLAIMS) - bad}/{len(CLAIMS)} README lines match the results files ({n} numbers checked)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
