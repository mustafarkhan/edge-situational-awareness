"""Export trained weights to ONNX and TensorRT engines (FP32, FP16, INT8).

RUN THIS ON THE JETSON. A TensorRT engine is compiled for the exact GPU, TensorRT
version and JetPack it was built on; an engine built on your desktop will not load there.

  python export/export_trt.py --weights best.pt --data configs/datasets/seadronessee.yaml --imgsz 640

INT8 needs calibration images: Ultralytics uses the dataset's *val* split for this
(never test). --fraction controls how much of val is used; a few hundred images is enough.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"])
    ap.add_argument("--fraction", type=float, default=0.25, help="share of val used for INT8 calibration")
    args = ap.parse_args()

    w = Path(args.weights)
    onnx = YOLO(str(w)).export(format="onnx", imgsz=args.imgsz, simplify=True, opset=17)
    print("ONNX:", onnx)

    for p in args.precisions:
        kwargs = dict(format="engine", imgsz=args.imgsz, batch=1, device=0, half=(p == "fp16"), int8=(p == "int8"))
        if p == "int8":
            kwargs.update(data=args.data, fraction=args.fraction)
        built = Path(YOLO(str(w)).export(**kwargs))
        final = built.with_name(f"{w.stem}_{args.imgsz}_{p}.engine")
        shutil.move(built, final)  # Ultralytics always writes best.engine; keep each one
        print(f"{p.upper()} engine: {final}")


if __name__ == "__main__":
    main()
