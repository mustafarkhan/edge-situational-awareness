"""Train a YOLO model from a YAML config. One config file = one experiment.

  python training/train.py configs/train/sds_y11n_640.yaml

Every key in the config except 'model' is passed straight to Ultralytics model.train().
The config is copied into the run folder so every result can be traced to its settings.
"""

import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    model = YOLO(cfg.pop("model"))
    cfg.setdefault("seed", 0)
    cfg.setdefault("deterministic", True)
    results = model.train(**cfg)
    run_dir = Path(results.save_dir)
    shutil.copy(cfg_path, run_dir / "experiment_config.yaml")
    print(f"\nweights: {run_dir / 'weights' / 'best.pt'}")
    print("Now log this run in docs/EXPERIMENT_LOG.md (what you changed, and why).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python training/train.py <config.yaml>")
    main(sys.argv[1])
