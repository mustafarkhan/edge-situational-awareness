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


def json_value(name: str, key: str):
    import json

    return json.loads((R / name).read_text())[key]


def j(name: str, key: str, fmt: str = "{:.2f}") -> str:
    """A value from a JSON result file (e.g. the SITL flight reports)."""
    import json

    val = json.loads((R / name).read_text())[key]
    return fmt.format(val) if isinstance(val, (int, float)) else str(val)


Z = "zeroshot.csv"
SEA, SEA2 = "seadronessee.csv", "seadronessee_two_class.csv"
pct = "{:.0f}%"
HIT, HOLD = "hituav_shared.csv", "visdrone_holdout.csv"

CLAIMS: list[tuple[str, list[str]]] = [
    # README line (unique text it contains), values it must contain
    ("| YOLO11n | COCO | 640 | 0.200", [v(Z, {"tag": "y11n_640"}, c) for c in ("AP", "AP_small", "AP_large")]),
    ("| YOLO11n | COCO | 1280 |", [v(Z, {"tag": "y11n_1280"}, c) for c in ("AP50", "AP", "AP_small")]),
    ("| YOLO11s | COCO | 640 |", [v(Z, {"tag": "y11s_640"}, c) for c in ("AP50", "AP", "AP_small")]),
    ("| YOLO11s | COCO | 1280 |", [v(Z, {"tag": "y11s_1280"}, c) for c in ("AP50", "AP", "AP_small")]),
    ("| YOLO11n + SAHI", [v(Z, {"tag": "y11n_640_sahi640"}, c) for c in ("AP50", "AP", "AP_small")]),
    (
        "| YOLO11n | fine-tuned, 3 epochs | 1280 |",
        [v(Z, {"tag": "ft_y11n_1280"}, c) for c in ("AP50", "AP", "AP_small", "AP_large")],
    ),
    (
        "| YOLO11n | trained 80 epochs on a GPU | 640 |",
        [v(Z, {"tag": "gpu_y11n_640"}, c) for c in ("AP50", "AP", "AP_small", "AP_large")],
    ),
    (
        "| **YOLO11n** | **trained 80 epochs on a GPU** | **1280**",
        [v(Z, {"tag": "gpu_y11n_1280"}, c) for c in ("AP50", "AP", "AP_small", "AP_large")],
    ),
    (
        "the GPU model **67%**",
        [v(HOLD, {"tag": t, "sequence": "all"}, "recall", pct, 100) for t in ("ft_y11n_ep3", "gpu_y11n_640")]
        + [v(HOLD, {"tag": t, "sequence": "all"}, "precision", pct, 100) for t in ("ft_y11n_ep3", "gpu_y11n_640")],
    ),
    (
        "| What helps most?",
        [
            v(Z, {"tag": "y11n_640"}, "AP_small"),
            v(Z, {"tag": "gpu_y11n_1280"}, "AP_small"),
            v(Z, {"tag": "ft_y11n_1280"}, "AP_small"),
        ],
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
        "| YOLO11n fine-tuned 4 epochs, 5 classes | 1280 |",
        [v(SEA, {"tag": "ft_y11n_1280"}, c) for c in ("AP", "AP_small", "AP_swimmer", "AP_boat")],
    ),
    (
        "| YOLO11n trained 100 epochs at 640 px on a GPU | 1280 |",
        [v(SEA, {"tag": "gpu_y11n_640_at1280"}, c) for c in ("AP", "AP_small", "AP_swimmer", "AP_boat")],
    ),
    (
        "| **YOLO11n trained 60 epochs at 1280 px on a GPU** | **1280** |",
        [v(SEA, {"tag": "gpu_y11n_1280"}, c) for c in ("AP", "AP_small", "AP_swimmer", "AP_boat")],
    ),
    (
        "at 1280: jetski 0.143 → 0.242",
        [
            v(SEA, {"tag": t}, c)
            for t in ("ft_y11n_1280", "gpu_y11n_1280")
            for c in ("AP_buoy", "AP_life_saving_appliances")
        ],
    ),
    (
        "Swimmer AP **0.027 → 0.222**",
        [
            v(SEA2, {"tag": "coco_y11n_640"}, "AP_swimmer"),
            v(SEA, {"tag": "gpu_y11n_1280"}, "AP_swimmer"),
            v(SEA2, {"tag": "coco_y11n_640"}, "AP_boat"),
            v(SEA, {"tag": "gpu_y11n_1280"}, "AP_boat"),
        ],
    ),
    (
        "Longer training memorises more: the GPU model's swimmer AP",
        [v(SEA, {"tag": t}, "AP_swimmer") for t in ("gpu_y11n_1280_seen", "gpu_y11n_1280")],
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
    # v2: search planning (head-and-shoulders case, 4 frames/s)
    *[
        (
            f"| **{alt} m** |" if alt == 100 else f"| {alt} m | ",
            [
                v("search_altitude.csv", {"size_m": 0.5, "fps": 4.0, "alt_m": alt}, "p_per_pass", "{:.2f}"),
                v("search_altitude.csv", {"size_m": 0.5, "fps": 4.0, "alt_m": alt}, "km2_h_at_pod80", "{:.2f}"),
            ],
        )
        for alt in (20, 50, 100, 120)
    ],
    # v2: alert de-duplication (all clips)
    *[
        (
            label,
            [
                v("alert_load.csv", {"clip": "all", "dedupe_radius_m": r}, c, "{:.0f}")
                for c in ("alerts", "unique_swimmers_alerted", "duplicate_alerts", "false_alerts")
            ],
        )
        for label, r in (("| off | ", 0), ("| 2 m | ", 2), ("| **3 m (default)** |", 3), ("| 5 m | ", 5))
    ],
    # v2: text prompts (test, swimmer + boat)
    *[
        (anchor, [v(csv_, {"tag": tag}, c) for c in ("AP", "AP_swimmer", "AP_boat")])
        for anchor, csv_, tag in (
            ('| YOLO11n, COCO "person" / "boat" |', SEA2, "coco_y11n_1280"),
            ("| YOLOE-11s, text prompts |", "seadronessee_prompts_two_class.csv", "yoloe11s_p3_synonyms_test"),
            ("| YOLOE-11m, text prompts |", "seadronessee_prompts_two_class.csv", "yoloe11m_p3_synonyms_test"),
            ("| **YOLO11n fine-tuned (§9)** |", SEA2, "ft_y11n_1280"),
        )
    ],
    # v2: thermal
    (
        "| YOLO11n COCO (colour photos) |",
        [v("hituav_shared.csv", {"tag": "coco_y11n_640_test"}, c) for c in ("AP", "AP_person", "AP_car", "AP_bicycle")]
        + [v("hituav_shared.csv", {"tag": t}, "AP") for t in ("coco_y11n_640_test_day", "coco_y11n_640_test_night")],
    ),
    (
        "| YOLO11n fine-tuned, 5 CPU epochs |",
        [v("hituav_shared.csv", {"tag": "ft_y11n_640_test"}, c) for c in ("AP", "AP_person", "AP_car", "AP_bicycle")]
        + [v("hituav_shared.csv", {"tag": t}, "AP") for t in ("ft_y11n_640_test_day", "ft_y11n_640_test_night")],
    ),
    (
        "| **YOLO11n trained 120 epochs on a GPU** |",
        [v(HIT, {"tag": "gpu_y11n_640_test"}, c) for c in ("AP", "AP_person", "AP_car", "AP_bicycle")]
        + [v(HIT, {"tag": t}, "AP") for t in ("gpu_y11n_640_test_day", "gpu_y11n_640_test_night")],
    ),
    (
        "GPU-trained **0.512**",
        [v(HIT, {"tag": t}, "AP") for t in ("gpu_y11n_640_test", "gpu_y11n_640_test_night", "gpu_y11n_640_test_day")],
    ),
    (
        "The GPU model memorises more: 0.584 seen",
        [v(HIT, {"tag": t}, "AP") for t in ("gpu_y11n_640_seen", "gpu_y11n_640_test")],
    ),
    # v2: land search plan (thermal)
    (
        "| 25° (25 mm, the HIT-UAV camera) | 110 m |",
        [
            v("land_search_altitude.csv", {"hfov_deg": 25.0, "alt_m": 110}, c, f)
            for c, f in (("px_on_person", "{:.0f} px"), ("p_per_pass", "{:.2f}"), ("km2_h_at_pod80", "{:.2f}"))
        ],
    ),
    (
        "| 45° (13 mm) | 60 m |",
        [
            v("land_search_altitude.csv", {"hfov_deg": 45.0, "alt_m": 60}, c, f)
            for c, f in (("px_on_person", "{:.0f} px"), ("p_per_pass", "{:.2f}"), ("km2_h_at_pod80", "{:.2f}"))
        ],
    ),  # fmt: skip
    # v2: text prompts on land
    *[
        (anchor, [v(csv_, {"tag": tag}, c) for c in ("AP", "AP_pedestrian", "AP_car")])
        for anchor, csv_, tag in (
            ('| YOLO11n, COCO "person" / "car" |', "visdrone_prompts_shared.csv", "coco_y11n_1280"),
            ("| YOLOE-11s, text prompts | no | 0.25", "visdrone_prompts_prompts_shared.csv", "yoloe11s_land"),
            ("| YOLOE-11m, text prompts | no | 0.28", "visdrone_prompts_prompts_shared.csv", "yoloe11m_land"),
            ("| **YOLO11n fine-tuned (§2)** |", "visdrone_prompts_shared.csv", "ft_y11n_1280"),
        )
    ],
    # v3: simulated flights of the example missions (whole sorties)
    *[
        (
            anchor,
            [
                j(f, "waypoints_reached", "{:d}/") + j(f, "waypoints_planned", "{:d}"),
                j(f, "cross_track_m_median", "{:.2f} m"),
                j(f, "cross_track_m_max", "{:.2f} m"),
                j(f, "positions_outside_fence", "| {:d} |"),
                j(f, "max_alt_m", "{:.1f} m"),
                "| yes |" if json_value(f, "landed") is True else "| NOT LANDED |",
            ],
        )
        for anchor, f in (
            ("| sea, 100 m, 0.5 km²", "sitl_sea.json"),
            ("| land at night, thermal, 60 m", "sitl_land.json"),
        )
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
