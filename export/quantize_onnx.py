"""INT8 post-training quantisation with ONNX Runtime, as a CPU stand-in for TensorRT INT8.

TensorRT needs an NVIDIA GPU. The question INT8 raises is the same on any runtime, though: how
much accuracy does 8-bit quantisation cost, and where? This script answers it on CPU:

  1. export the trained .pt to ONNX (FP32)
  2. static INT8 quantisation (QDQ, per-channel weights) calibrated on TRAIN images
     (never the images we evaluate on)
  3. two INT8 variants:
       int8_full     everything quantised
       int8_headfp32 the detection head's post-processing (box decoding, DFL softmax, final
                     concat of box coordinates with class scores) kept in FP32. Box coordinates
                     run 0..1280 and class scores 0..1; squeezing both into one 8-bit range is
                     the classic way YOLO INT8 loses its scores.

Evaluate each file with training/evaluate_zeroshot.py (same metric as everything else).

  python export/quantize_onnx.py --weights weights/ft_y11n_ep3.pt --imgsz 1280 \
      --calib-dir datasets/visdrone_yolo/images/train --n-calib 128
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from ultralytics import YOLO


def letterbox(path: Path, size: int) -> np.ndarray:
    """Same preprocessing Ultralytics applies to a fixed-size ONNX model: resize the long side to
    `size`, pad with grey (114) to a square, RGB, 0..1, NCHW."""
    im = Image.open(path).convert("RGB")
    s = size / max(im.size)
    nw, nh = round(im.width * s), round(im.height * s)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]


class Reader(CalibrationDataReader):
    def __init__(self, files, size, input_name):
        self.it = iter(files)
        self.size, self.name = size, input_name

    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {self.name: letterbox(p, self.size)}


def head_nodes(model: onnx.ModelProto, head_prefix: str) -> list[str]:
    """The head's post-processing only: nodes directly under the Detect module (per-scale concat of
    box and class outputs, box decoding, final concat, class sigmoid) and the DFL block. The head's
    conv branches (cv2/cv3) stay quantised."""
    out = []
    for n in model.graph.node:
        if not n.name.startswith(head_prefix):
            continue
        rest = n.name[len(head_prefix) :]
        if "/" not in rest or rest.startswith("dfl/"):
            out.append(n.name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w = Path(args.weights)
    fp32 = Path(YOLO(str(w)).export(format="onnx", imgsz=args.imgsz, simplify=True, opset=17, dynamic=False))
    fp32 = fp32.rename(w.with_name(f"{w.stem}_{args.imgsz}_fp32.onnx"))
    print("FP32:", fp32)

    pre = fp32.with_name(fp32.stem.replace("_fp32", "_pre") + ".onnx")
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    base = onnx.load(str(fp32))
    input_name = base.graph.input[0].name
    # node names look like "/model.23/dfl/conv/Conv"; the highest index is the Detect head
    head_idx = max(int(n.name.split("/")[1].split(".")[1]) for n in base.graph.node if n.name.startswith("/model."))
    excluded = head_nodes(base, f"/model.{head_idx}/")
    files = sorted(Path(args.calib_dir).glob("*.jpg"))
    random.Random(args.seed).shuffle(files)
    calib = files[: args.n_calib]
    print(
        f"calibration: {len(calib)} train images; head = model.{head_idx}, {len(excluded)} head nodes kept FP32 in variant 2"
    )

    for tag, exclude in (("int8_full", []), ("int8_headfp32", excluded)):
        out = fp32.with_name(fp32.stem.replace("_fp32", f"_{tag}") + ".onnx")
        quantize_static(
            str(pre),
            str(out),
            Reader(calib, args.imgsz, input_name),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=exclude,
        )
        # keep Ultralytics' metadata (class names, stride, imgsz) so YOLO() can load the file
        q = onnx.load(str(out))
        del q.metadata_props[:]
        q.metadata_props.extend(base.metadata_props)
        onnx.save(q, str(out))
        print(f"{tag}: {out} ({out.stat().st_size / 1e6:.1f} MB vs FP32 {fp32.stat().st_size / 1e6:.1f} MB)")
    pre.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
