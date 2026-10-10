"""Render the demo video from REAL outputs only: dataset labels, model detections, the
tracking JSONL from tools/detect_video.py, and benchmarks/results/*.csv.

  python docs/video/make_video.py --track-jsonl outputs/demo_track.jsonl \
      --track-frames datasets/demo_holdout/images --repo-url github.com/<you>/edge-situational-awareness

Needs ffmpeg. Output: outputs/demo_video.mp4 (1920x1080, 30 fps, H.264).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
W, H = 1920, 1080
BG, FG, MUTED, ACCENT, ALERT = (14, 17, 22), (236, 238, 241), (150, 158, 170), (255, 196, 0), (255, 82, 82)
CLASS_COL = {"person": (255, 82, 82), "two_wheeler": (186, 104, 200), "vehicle": (66, 165, 245)}
VIS_SUPER = {"pedestrian": "person", "people": "person", "bicycle": "two_wheeler", "motor": "two_wheeler",
             "car": "vehicle", "van": "vehicle", "truck": "vehicle", "bus": "vehicle",
             "tricycle": "two_wheeler", "awning-tricycle": "two_wheeler"}
COCO_SUPER = {"person": "person", "bicycle": "two_wheeler", "motorcycle": "two_wheeler",
              "car": "vehicle", "bus": "vehicle", "truck": "vehicle"}


def font(size, bold=False):
    for p in (["/usr/share/fonts/opentype/inter/Inter-Bold.otf"] if bold else []) + \
             ["/usr/share/fonts/opentype/inter/Inter-Regular.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_H1, F_H2, F_BODY, F_SMALL, F_TINY = font(76, True), font(48, True), font(36), font(28), font(22)


def canvas():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def footer(d, text="Measured on VisDrone (public data) · CPU results, not yet Jetson"):
    d.text((60, H - 50), text, font=F_TINY, fill=MUTED)


def fit(img: Image.Image, box_w, box_h):
    s = min(box_w / img.width, box_h / img.height)
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS), s


def boxes_from_yolo(lbl: Path, names, iw, ih):
    out = []
    for line in lbl.read_text().splitlines():
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * iw, float(cy) * ih, float(bw) * iw, float(bh) * ih
        out.append((VIS_SUPER[names[int(c)]], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def draw_boxes(d, boxes, s, ox, oy, width=2):
    for lab, x1, y1, x2, y2 in boxes:
        d.rectangle([ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s], outline=CLASS_COL.get(lab, ACCENT), width=width)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def count_found(gt_boxes, det_boxes, thr=0.5):
    """Greedy one-to-one matching, class-agnostic, IoU >= thr. Returns ground-truth objects found."""
    used, found = set(), 0
    for g in gt_boxes:
        best, bi = thr, None
        for i, d_ in enumerate(det_boxes):
            if i not in used and (v := iou(g, d_)) >= best:
                best, bi = v, i
        if bi is not None:
            used.add(bi); found += 1
    return found


# ---------------------------------------------------------------- scenes
def scene_title(repo_url):
    im, d = canvas()
    d.text((120, 330), "Edge AI for drone situational awareness", font=F_H1, fill=FG)
    d.text((120, 440), "Detecting people and vehicles from the air, and what the data says about it",
           font=F_BODY, fill=MUTED)
    d.rectangle([120, 520, 320, 526], fill=ACCENT)
    d.text((120, 570), "YOLO11 · SAHI · ByteTrack / BoT-SORT · ONNX INT8 · React map · Jetson (next)", font=F_SMALL, fill=FG)
    d.text((120, 940), repo_url, font=F_SMALL, fill=MUTED)
    return im


def scene_tiny_objects(img_path, lbl_path, names, stats):
    im, d = canvas()
    src = Image.open(img_path).convert("RGB")
    boxes = boxes_from_yolo(lbl_path, names, src.width, src.height)
    fitted, s = fit(src, 1180, 900)
    ox, oy = 60, 110
    im.paste(fitted, (ox, oy))
    draw_boxes(d, boxes, s, ox, oy, 1)
    # zoom inset on the densest small-object region
    persons = [b for b in boxes if b[0] == "person"] or boxes
    cx = np.median([(b[1] + b[3]) / 2 for b in persons]); cy = np.median([(b[2] + b[4]) / 2 for b in persons])
    zw = 160
    x0, y0 = int(np.clip(cx - zw / 2, 0, src.width - zw)), int(np.clip(cy - zw / 2, 0, src.height - zw))
    crop = src.crop((x0, y0, x0 + zw, y0 + zw)).resize((480, 480), Image.NEAREST)
    cd = ImageDraw.Draw(crop)
    k = 480 / zw
    for lab, x1, y1, x2, y2 in boxes:
        if x2 > x0 and x1 < x0 + zw and y2 > y0 and y1 < y0 + zw:
            cd.rectangle([(x1 - x0) * k, (y1 - y0) * k, (x2 - x0) * k, (y2 - y0) * k], outline=CLASS_COL[lab], width=3)
    d.rectangle([ox + x0 * s, oy + y0 * s, ox + (x0 + zw) * s, oy + (y0 + zw) * s], outline=ACCENT, width=3)
    im.paste(crop, (1300, 110)); d.rectangle([1300, 110, 1780, 590], outline=ACCENT, width=3)
    d.text((60, 40), "The real problem: objects are tiny", font=F_H2, fill=FG)
    y = 660
    for line in [f"{stats['n_boxes']:,} labelled objects",
                 f"in {stats['n_images']} validation images",
                 "",
                 f"At 640 px model input:",
                 f"median object {stats['median640']:.0f} px across",
                 f"{stats['lt16_640']:.0%} under 16 px",
                 f"{stats['lt32_640']:.0%} 'small' (COCO)"]:
        d.text((1300, y), line, font=F_SMALL, fill=FG); y += 52
    d.text((1300, 598), "Inset: zoomed 3x, pixels not smoothed", font=F_TINY, fill=MUTED)
    footer(d, "VisDrone2019-DET val, ground-truth labels")
    return im


def scene_plain_vs_sahi(img_path, plain, sahi, t_plain, t_sahi):
    im, d = canvas()
    src = Image.open(img_path).convert("RGB")
    d.text((60, 40), "Same frame, same model (YOLO11n, COCO weights)", font=F_H2, fill=FG)
    for i, (title, dets, t) in enumerate([("Whole frame at 640 px", plain, t_plain),
                                          ("Sliced into 640 px tiles (SAHI)", sahi, t_sahi)]):
        fitted, s = fit(src, 880, 700)
        ox, oy = 60 + i * 920, 200
        im.paste(fitted, (ox, oy))
        draw_boxes(d, dets, s, ox, oy, 2)
        d.text((ox, 140), title, font=F_BODY, fill=ACCENT if i else FG)
        n_p = sum(1 for b in dets if b[0] == "person")
        d.text((ox, oy + fitted.height + 30), f"{len(dets)} detections", font=F_H2, fill=FG)
        d.text((ox, oy + fitted.height + 100), f"{n_p} of them people · {t:.0f} ms on CPU", font=F_SMALL, fill=FG)
    footer(d, "Confidence 0.25 · real detections on a VisDrone validation image")
    return im


def scene_bars(df, title, metric, label, note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 20, "text.color": "#eceef1", "axes.labelcolor": "#eceef1",
                         "xtick.color": "#eceef1", "ytick.color": "#969eaa", "axes.edgecolor": "#3a414b"})
    fig, ax = plt.subplots(figsize=(17, 7.6), dpi=100, facecolor="#0e1116")
    ax.set_facecolor("#0e1116")
    colors = ["#ffc400" if "fine-tuned" in t else "#42a5f5" for t in df["label"]]
    b = ax.bar(df["label"], df[metric], color=colors)
    for rect, v, ms in zip(b, df[metric], df["cpu_ms_per_image_median"]):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}\n{ms:.0f} ms", ha="center", va="bottom", fontsize=18)
    ax.set_ylabel(label); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(df[metric]) * 1.3)
    fig.tight_layout()
    tmp = Path("/tmp/esa-bars.png"); fig.savefig(tmp, facecolor=fig.get_facecolor()); plt.close(fig)
    im, d = canvas()
    d.text((60, 40), title, font=F_H2, fill=FG)
    im.paste(Image.open(tmp).convert("RGB"), (60, 150))
    footer(d, note)
    return im


def scene_heldout(frame_path, rec, gt_boxes, totals, model_note):
    im, d = canvas()
    src = Image.open(frame_path).convert("RGB")
    fitted, sc = fit(src, 1180, 885)
    ox, oy = 40, 120
    im.paste(fitted, (ox, oy))
    dets = []
    for det in rec["detections"]:
        lab = VIS_SUPER.get(det["label"], COCO_SUPER.get(det["label"], "vehicle"))
        x1, y1, x2, y2 = det["bbox"]; dets.append((x1, y1, x2, y2))
        d.rectangle([ox + x1 * sc, oy + y1 * sc, ox + x2 * sc, oy + y2 * sc], outline=CLASS_COL[lab], width=2)
    d.text((40, 40), "Fine-tuned model on footage it never saw", font=F_H2, fill=FG)
    found = count_found([g[1:] for g in gt_boxes], dets)
    totals["found"] += found; totals["gt"] += len(gt_boxes)
    px = 1260
    d.text((px, 140), "This frame", font=F_SMALL, fill=MUTED)
    d.text((px, 180), f"{found} of {len(gt_boxes)}", font=F_H1, fill=FG)
    d.text((px, 275), "labelled objects found", font=F_SMALL, fill=FG)
    d.text((px, 360), "Running total", font=F_SMALL, fill=MUTED)
    d.text((px, 400), f"{totals['found']} of {totals['gt']}", font=F_H2, fill=ACCENT)
    d.text((px, 470), f"({totals['found'] / max(totals['gt'], 1):.0%} recall at IoU 0.5)", font=F_SMALL, fill=FG)
    counts = {}
    for det in rec["detections"]:
        lab = VIS_SUPER.get(det["label"], COCO_SUPER.get(det["label"], "vehicle"))
        counts[lab] = counts.get(lab, 0) + 1
    y = 580
    for lab in ("person", "two_wheeler", "vehicle"):
        d.rectangle([px, y + 8, px + 18, y + 26], fill=CLASS_COL[lab])
        d.text((px + 30, y), f"{lab.replace('_', ' ')} detections: {counts.get(lab, 0)}", font=F_SMALL, fill=FG); y += 44
    for k, part in enumerate(model_note.split(" | ")):
        d.text((px, 760 + k * 30), part, font=F_TINY, fill=MUTED)
    footer(d, "VisDrone sequence 9999951, removed from training before fine-tuning · frames are seconds apart, not continuous video")
    return im


def scene_text(title, lines, note=None):
    im, d = canvas()
    d.text((120, 160), title, font=F_H2, fill=FG)
    y = 290
    for ln in lines:
        d.text((120, y), "•", font=F_BODY, fill=ACCENT)
        d.text((165, y), ln, font=F_BODY, fill=FG); y += 78
    if note:
        footer(d, note)
    return im


# ---------------------------------------------------------------- phase-2 scenes
def mpl_dark():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 20, "text.color": "#eceef1", "axes.labelcolor": "#eceef1",
                         "xtick.color": "#eceef1", "ytick.color": "#969eaa", "axes.edgecolor": "#3a414b",
                         "axes.facecolor": "#0e1116", "figure.facecolor": "#0e1116", "legend.frameon": False})
    return plt


def chart_scene(title, fig, note, size=(1800, 860)):
    plt = mpl_dark()
    tmp = Path("/tmp/esa-chart.png")
    fig.savefig(tmp, facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    im, d = canvas()
    d.text((60, 40), title, font=F_H2, fill=FG)
    ch = Image.open(tmp).convert("RGB")
    ch.thumbnail(size)
    im.paste(ch, (60, 140))
    footer(d, note)
    return im


TRACK_PALETTE = [(255, 82, 82), (66, 165, 245), (255, 196, 0), (102, 187, 106), (186, 104, 200), (255, 138, 101),
                 (77, 208, 225), (240, 98, 146), (174, 213, 129), (149, 117, 205)]


def scene_tracking(frame_path, rec, telem, summary, title):
    im, d = canvas()
    src = Image.open(frame_path).convert("RGB")
    fitted, sc = fit(src, 1300, 731)
    ox, oy = 40, 130
    im.paste(fitted, (ox, oy))
    for det in rec["detections"]:
        if det["track_id"] is None:
            continue
        col = TRACK_PALETTE[det["track_id"] % len(TRACK_PALETTE)]
        x1, y1, x2, y2 = det["bbox"]
        d.rectangle([ox + x1 * sc, oy + y1 * sc, ox + x2 * sc, oy + y2 * sc], outline=col, width=2)
        d.text((ox + x1 * sc, oy + y1 * sc - 20), str(det["track_id"]), font=F_TINY, fill=col)
    d.text((40, 40), title, font=F_H2, fill=FG)
    px = 1380
    d.text((px, 140), f"Frame {rec['frame'] + 1} / {summary['frames']}", font=F_SMALL, fill=MUTED)
    d.text((px, 185), f"Altitude {telem['alt_m']:.0f} m (simulated)", font=F_SMALL, fill=MUTED)
    d.text((px, 260), "Whole run", font=F_SMALL, fill=MUTED)
    d.text((px, 300), f"IDF1 {summary['IDF1']:.2f}", font=F_H1, fill=ACCENT)
    d.text((px, 400), f"{summary['id_switches']} ID switches", font=F_BODY, fill=FG)
    d.text((px, 450), f"{summary['gt_objects']} true objects", font=F_BODY, fill=FG)
    d.text((px, 500), f"recall {summary['recall']:.0%} · precision {summary['precision']:.0%}", font=F_SMALL, fill=FG)
    d.text((px, 560), f"position error {summary['geo_err_median_m']} m (median)", font=F_SMALL, fill=FG)
    d.text((px, 640), "Same colour + number =", font=F_SMALL, fill=MUTED)
    d.text((px, 675), "same track ID", font=F_SMALL, fill=MUTED)
    footer(d, "Camera flown over a real, fully labelled VisDrone frame: every object's identity and position is known exactly")
    return im


def fig_trackers(tr):
    plt = mpl_dark()
    fig, ax = plt.subplots(figsize=(18, 8.6))
    speeds = [("slow", "slow camera\n0.6 m per frame"), ("fast", "fast camera\n2.4 m per frame")]
    trackers = [("bytetrack", "ByteTrack", "#42a5f5"), ("botsort_nogmc", "BoT-SORT, no motion comp.", "#90a4ae"),
                ("botsort", "BoT-SORT + motion comp.", "#ffc400")]
    wbar = 0.26
    for j, (key, lab, col) in enumerate(trackers):
        vals, sw = [], []
        for sp, _ in speeds:
            r = tr[tr["tag"] == f"{sp}_{key}"].iloc[0]
            vals.append(r["IDF1"]); sw.append(int(r["id_switches"]))
        xs = [i + (j - 1) * wbar for i in range(len(speeds))]
        bars = ax.bar(xs, vals, wbar * 0.92, color=col, label=lab)
        for b, v, n in zip(bars, vals, sw):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}\n{n} sw", ha="center", va="bottom", fontsize=17)
    ax.set_xticks(range(len(speeds)), [s[1] for s in speeds])
    ax.set_ylabel("IDF1 (identity accuracy)"); ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), fontsize=18)
    fig.tight_layout()
    return fig


def fig_geo(geo):
    plt = mpl_dark()
    fig, ax = plt.subplots(figsize=(18, 8.6))
    labels = [f"{r.error_source} {r.level}" for r in geo.itertuples()]
    cols = ["#ff5252" if "tilt" in r.error_source else "#42a5f5" for r in geo.itertuples()]
    y = range(len(geo))[::-1]
    ax.barh(list(y), geo["median_m"], color=cols)
    ax.scatter(geo["p95_m"], list(y), color="#eceef1", marker="|", s=400)
    for yy, m, p95 in zip(y, geo["median_m"], geo["p95_m"]):
        ax.text(max(m, p95) + 0.5, yy, f"{m:.1f} m", va="center", fontsize=17)
    ax.set_yticks(list(y), labels); ax.set_xlabel("position error (metres): bar = median, tick = 95th percentile")
    ax.set_xlim(0, geo["p95_m"].max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_quant(q, pt_row):
    plt = mpl_dark()
    rows = [("PyTorch FP32", pt_row["AP_small"], pt_row["cpu_ms_per_image_median"], "#90a4ae")]
    lab = {"onnx_fp32_1280": "ONNX FP32", "onnx_int8_full_1280": "ONNX INT8\n(everything)",
           "onnx_int8_headfp32_1280": "ONNX INT8\n(head kept FP32)"}
    for t, l in lab.items():
        r = q[q["tag"] == t].iloc[0]
        rows.append((l, r["AP_small"], r["cpu_ms_per_image_median"], "#ff5252" if "everything" in l else
                     ("#ffc400" if "head" in l else "#42a5f5")))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8.6))
    for ax, k, ttl, fmt in ((a1, 1, "AP (small objects)", "{:.3f}"), (a2, 2, "CPU ms per image", "{:.0f}")):
        b = ax.bar([r[0] for r in rows], [r[k] for r in rows], color=[r[3] for r in rows])
        for bb, r in zip(b, rows):
            txt = "no detections\nat all" if (k == 1 and r[k] == 0) else fmt.format(r[k])
            ax.text(bb.get_x() + bb.get_width() / 2, r[k], txt, ha="center", va="bottom", fontsize=18,
                    color="#ff5252" if txt.startswith("no") else "#eceef1")
        ax.set_title(ttl, fontsize=22); ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=15)
    fig.tight_layout()
    return fig


def fig_failure(fr):
    plt = mpl_dark()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8.6), gridspec_kw={"width_ratios": [5, 4]})
    for ax, dim, ttl in ((a1, "size", "by object size (model-input px)"), (a2, "light", "by lighting")):
        g = fr[fr["dimension"] == dim].copy()
        order = ["< 8 px", "8-16 px", "16-32 px", "32-64 px", ">= 64 px", "daylight", "dusk/night"]
        g["o"] = [order.index(x) if x in order else 99 for x in g["bucket"]]
        g = g.sort_values("o")
        b = ax.bar(g["bucket"], g["recall"], color="#42a5f5" if dim == "size" else ["#42a5f5", "#ff5252"])
        for bb, v, n in zip(b, g["recall"], g["objects"]):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.0%}\n{n:,} obj", ha="center", va="bottom", fontsize=16)
        ax.set_ylim(0, 1.1); ax.set_title(ttl, fontsize=22); ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=16)
    a1.set_ylabel("recall (share of labelled objects found)")
    fig.tight_layout()
    return fig


def map_clip_bg(path_png):
    im, d = canvas()
    d.text((60, 40), "The real web app, replaying that flight", font=F_H2, fill=FG)
    d.text((60, 100), "React + MapLibre over a FastAPI WebSocket · objects stay put while the drone (yellow) moves",
           font=F_TINY, fill=MUTED)
    footer(d, "Offline map style (grid every 20 m) because map tiles were not reachable from the build machine")
    im.save(path_png)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "configs/datasets/visdrone.yaml"))
    ap.add_argument("--hero-image", required=True, help="val image used for the tiny-objects and SAHI scenes")
    ap.add_argument("--track-jsonl", required=True)
    ap.add_argument("--track-frames", required=True)
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--sahi-json", required=True, help="from docs/video/sahi_example.py")
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--out", default=str(REPO / "outputs/demo_video.mp4"))
    ap.add_argument("--holdout-labels", required=True)
    ap.add_argument("--frame-secs", type=float, default=0.8)
    ap.add_argument("--model-note", default="")
    ap.add_argument("--flight-dir", required=True)
    ap.add_argument("--flight-run", required=True, help="tracking JSONL shown in the video")
    ap.add_argument("--flight-tag", required=True, help="its row tag in tracking.csv")
    ap.add_argument("--map-clip", default=None)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.data).read_text())
    names = cfg["names"]
    root = Path(cfg["path"])
    hero = Path(args.hero_image)
    stats = json.loads(Path(args.stats_json).read_text())
    sx = json.loads(Path(args.sahi_json).read_text())

    work = Path("/tmp/esa-video"); shutil.rmtree(work, ignore_errors=True); work.mkdir()
    R = REPO / "benchmarks/results"
    seq = []  # (png, seconds)
    parts = []  # list of ("pngs", list) or ("clip", path)

    counter = [0]  # global across parts: file names must never be reused

    def add(img, secs):
        p = work / f"{counter[0]:05d}.png"; counter[0] += 1
        img.save(p); seq.append((p, secs))

    def flush():
        if seq:
            parts.append(("pngs", list(seq))); seq.clear()

    add(scene_title(args.repo_url), 4)
    add(scene_tiny_objects(hero, root / "labels/val" / (hero.stem + ".txt"), names, stats), 7)
    add(scene_plain_vs_sahi(hero, [tuple(b) for b in sx["plain"]], [tuple(b) for b in sx["sahi"]],
                            sx["ms_plain"], sx["ms_sahi"]), 7)

    df = pd.read_csv(R / "zeroshot.csv").drop_duplicates("tag", keep="last")
    pretty = {"y11n_640": "11n\n640", "y11n_1280": "11n\n1280", "y11s_640": "11s\n640", "y11s_1280": "11s\n1280",
              "y11n_640_sahi640": "11n\nSAHI", "ft_y11n_640": "11n fine-tuned\n640",
              "ft_y11n_1280": "11n fine-tuned\n1280"}
    df = df[df["tag"].isin(pretty)].copy()
    df["label"] = df["tag"].map(pretty)
    df["order"] = [list(pretty).index(t) for t in df["tag"]]
    df = df.sort_values("order")
    add(scene_bars(df, "Small-object accuracy vs time per image", "AP_small", "AP (small objects)",
                   "COCO AP for objects < 32x32 px · 548 VisDrone val images · median CPU ms per image (2 vCPU)"), 8)

    recs = [json.loads(l) for l in Path(args.track_jsonl).read_text().splitlines()]
    frames = sorted(Path(args.track_frames).iterdir())
    lbl_dir = Path(args.holdout_labels)
    totals = {"found": 0, "gt": 0}
    for rec, fp in zip(recs, frames):
        g_im = Image.open(fp)
        gt_boxes = boxes_from_yolo(lbl_dir / (fp.stem + ".txt"), names, g_im.width, g_im.height)
        add(scene_heldout(fp, rec, gt_boxes, totals, args.model_note), args.frame_secs)

    # tracking on the simulated flight
    tr = pd.read_csv(R / "tracking.csv").drop_duplicates("tag", keep="last")
    fl = Path(args.flight_dir)
    run = [json.loads(l) for l in Path(args.flight_run).read_text().splitlines()]
    telem = [json.loads(l) for l in (fl / "telemetry.jsonl").read_text().splitlines()]
    summ = tr[tr["tag"] == args.flight_tag].iloc[0].to_dict()
    for rec, fp in zip(run, sorted((fl / "frames").iterdir())):
        add(scene_tracking(fp, rec, telem[rec["frame"]], summ, "Tracking: does each object keep one ID?"), 0.1)
    add(chart_scene("Camera motion vs tracker: identity accuracy", fig_trackers(tr),
                    "Same detections, same flights; only the tracker changes · sw = ID switches · 120 frames per run"), 8)
    flush()
    if args.map_clip and Path(args.map_clip).exists():
        bg = work / "map_bg.png"; map_clip_bg(bg)
        parts.append(("clip", Path(args.map_clip), bg))
    geo = pd.read_csv(R / "geo_sensitivity.csv")
    add(chart_scene("Where do position errors come from?", fig_geo(geo),
                    "Every labelled object in the flight, one error source at a time · drone at 57-69 m, 70 deg lens"), 8)
    q = pd.read_csv(R / "quantization.csv").drop_duplicates("tag", keep="last")
    add(chart_scene("INT8: smaller and faster, if you quantise the right parts", fig_quant(q, df[df.tag == "ft_y11n_1280"].iloc[0]),
                    "Fine-tuned YOLO11n at 1280 px · ONNX Runtime on CPU (TensorRT on the Jetson is next) · 548 val images"), 8)
    fr = pd.read_csv(R / "failure_analysis.csv")
    add(chart_scene("Where it still fails", fig_failure(fr),
                    "Fine-tuned YOLO11n, 1280 px, confidence 0.25 · found = a box with IoU >= 0.5 · 548 val images"), 8)
    add(scene_text("What this shows, honestly", [
        "Off-the-shelf detectors miss most small objects in drone views",
        "Resolution and a short fine-tune recover the most; slicing costs too much here",
        "Camera-motion compensation keeps IDs stable when the drone moves fast",
        "Ignoring gimbal tilt costs more position accuracy than GPS error",
        "INT8 works only with the detection head kept in FP32"],
        "All numbers measured on public data; nothing estimated"), 9)
    add(scene_text("Next", ["TensorRT FP16 / INT8 on Jetson Orin Nano: speed, watts, joules per frame",
                            "Real continuous footage: VisDrone-MOT and my own flights",
                            "Maritime search and rescue data (SeaDronesSee)"], None), 5)
    add(scene_title(args.repo_url), 3)
    flush()

    # render each part, then concatenate
    outs = []
    for k, part in enumerate(parts):
        o = work / f"part{k}.mp4"
        if part[0] == "pngs":
            lst = work / f"list{k}.txt"
            with lst.open("w") as f:
                for p, s_ in part[1]:
                    f.write(f"file '{p}'\nduration {s_:.4f}\n")
                f.write(f"file '{part[1][-1][0]}'\n")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "20", str(o)], check=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(part[2]), "-i", str(part[1]),
                            "-filter_complex", "[1:v]scale=1520:855[m];[0:v][m]overlay=200:135:shortest=1,fps=30,format=yuv420p",
                            "-c:v", "libx264", "-crf", "20", str(o)], check=True)
        outs.append(o)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    lst = work / "parts.txt"
    lst.write_text("".join(f"file '{o}'\n" for o in outs))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                         capture_output=True, text=True).stdout.strip()
    print("video ->", out, f"({float(dur):.1f} s)")


if __name__ == "__main__":
    sys.exit(main())
