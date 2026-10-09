"""Helpers shared by the dataset converters."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def link_or_copy(src: Path, dst: Path):
    """Symlink images instead of copying them (saves tens of GB). Falls back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_label(path: Path, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def yolo_row(cls: int, x: float, y: float, w: float, h: float, iw: int, ih: int) -> str | None:
    """COCO-style [x, y, w, h] in pixels -> YOLO 'cls cx cy w h' normalised to 0..1."""
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(iw), x + w), min(float(ih), y + h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None  # degenerate or fully outside the image
    cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / iw:.6f} {(y2 - y1) / ih:.6f}"


def write_dataset_yaml(name: str, out_root: Path, names: list[str]) -> Path:
    """Ultralytics resolves relative dataset paths against its own settings dir, which
    confuses everyone once. Writing an absolute path avoids that."""
    cfg = {"path": str(out_root.resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": {i: n for i, n in enumerate(names)}}
    p = REPO / "configs" / "datasets" / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p
