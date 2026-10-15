"""Turn the result CSVs into Markdown tables for the README and experiment log.

python benchmarks/make_tables.py            # print all tables
python benchmarks/make_tables.py zeroshot   # print one table
"""

import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
TABLES = {
    "zeroshot": [
        "tag",
        "imgsz",
        "sahi",
        "AP50",
        "AP",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AP_person",
        "AP_two_wheeler",
        "AP_vehicle",
        "cpu_ms_per_image_median",
    ],
    "tracking": [
        "tag",
        "speed_m_per_frame",
        "IDF1",
        "id_switches",
        "MOTA",
        "recall",
        "precision",
        "geo_err_median_m",
        "cpu_fps_median",
    ],
    "quantization": ["tag", "AP50", "AP", "AP_small", "AP_large", "cpu_ms_per_image_median"],
    "geo_sensitivity": ["error_source", "level", "median_m", "p95_m", "max_m"],
    "failure_analysis": ["dimension", "bucket", "objects", "recall"],
    "jetson_accuracy": ["tag", "imgsz", "AP50", "AP", "AP_small", "cpu_ms_per_image_median"],
    "jetson_sea_accuracy": ["tag", "imgsz", "AP50", "AP", "AP_small", "AP_swimmer", "AP_boat"],
    "latency_cpu": ["tag", "model", "imgsz", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "fps"],
    "accuracy": ["tag", "split", "imgsz", "sahi", "AP50", "AP", "AP_small", "ms_per_image_median"],
    "jetson": [
        "tag",
        "imgsz",
        "mAP50",
        "mAP50_95",
        "fps",
        "lat_p50_ms",
        "lat_p99_ms",
        "power_w",
        "energy_j_per_frame",
        "temp_c_max",
    ],
}


def table(name: str) -> str | None:
    p = R / f"{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "tag" in df.columns:
        df = df.drop_duplicates("tag", keep="last")
    df = df[[c for c in TABLES[name] if c in df.columns]]
    return df.to_markdown(index=False, floatfmt=".3f")


if __name__ == "__main__":
    for name in sys.argv[1:] or TABLES:
        if name not in TABLES:
            sys.exit(f"unknown table {name!r}; choose from {', '.join(TABLES)}")
        t = table(name)
        if t:
            print(f"\n### {name}\n\n{t}")
