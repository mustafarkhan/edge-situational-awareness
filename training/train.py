"""Train a YOLO model from a YAML config. One config file = one experiment.

  python training/train.py configs/train/sds_y11n_640.yaml
  python training/train.py configs/train/gpu/sds_y11n_640.yaml device=0,1 epochs=5   # key=value overrides

Every key in the config except 'model' is passed straight to Ultralytics model.train().
The config is copied into the run folder so every result can be traced to its settings.
"""

import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def main(cfg_path: str, overrides: list[str] | None = None):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    for kv in overrides or []:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)  # numbers and booleans keep their type; "0,1" stays a string
    model = YOLO(cfg.pop("model"))
    cfg.setdefault("seed", 0)
    cfg.setdefault("deterministic", True)
    results = model.train(**cfg)
    run_dir = Path(results.save_dir)
    shutil.copy(cfg_path, run_dir / "experiment_config.yaml")
    print(f"\nweights: {run_dir / 'weights' / 'best.pt'}")
    print("Now log this run in docs/EXPERIMENT_LOG.md (what you changed, and why).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python training/train.py <config.yaml> [key=value ...]")
    main(sys.argv[1], sys.argv[2:])
