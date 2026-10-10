"""Sea scenes for the demo video (SeaDronesSee). Every number comes from benchmarks/results/*.csv."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from make_video import (
    ACCENT,
    F_H1,
    F_H2,
    F_SMALL,
    F_TINY,
    FG,
    MUTED,
    TRACK_PALETTE,
    canvas,
    chart_scene,
    fit,
    footer,
    mpl_dark,
)
from PIL import Image, ImageDraw

SEA_COL = {
    "swimmer": (255, 82, 82),
    "boat": (66, 165, 245),
    "jetski": (186, 104, 200),
    "life_saving_appliances": (255, 196, 0),
    "buoy": (102, 187, 106),
}


def yolo_boxes(lbl: Path, names, iw, ih):
    out = []
    for line in lbl.read_text().splitlines():
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * iw, float(cy) * ih, float(bw) * iw, float(bh) * ih
        out.append((names[int(c)], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def scene_sea_intro(img_path: Path, lbl_path: Path, names, stats_md: str):
    """Ground truth on a sea test image, with a zoomed inset and the dataset's size statistics."""
    im, d = canvas()
    src = Image.open(img_path).convert("RGB")
    boxes = yolo_boxes(lbl_path, names, src.width, src.height)
    fitted, s = fit(src, 1180, 760)
    ox, oy = 60, 120
    im.paste(fitted, (ox, oy))
    for lab, x1, y1, x2, y2 in boxes:
        d.rectangle([ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s], outline=SEA_COL[lab], width=2)
    sw = [b for b in boxes if b[0] == "swimmer"] or boxes
    cx, cy = (sw[0][1] + sw[0][3]) / 2, (sw[0][2] + sw[0][4]) / 2
    zw = 120
    x0 = int(np.clip(cx - zw / 2, 0, src.width - zw))
    y0 = int(np.clip(cy - zw / 2, 0, src.height - zw))
    crop = src.crop((x0, y0, x0 + zw, y0 + zw)).resize((480, 480), Image.NEAREST)
    cd = ImageDraw.Draw(crop)
    k = 480 / zw
    for lab, x1, y1, x2, y2 in boxes:
        if x2 > x0 and x1 < x0 + zw and y2 > y0 and y1 < y0 + zw:
            cd.rectangle([(x1 - x0) * k, (y1 - y0) * k, (x2 - x0) * k, (y2 - y0) * k], outline=SEA_COL[lab], width=3)
    d.rectangle([ox + x0 * s, oy + y0 * s, ox + (x0 + zw) * s, oy + (y0 + zw) * s], outline=ACCENT, width=3)
    im.paste(crop, (1300, 120))
    d.rectangle([1300, 120, 1780, 600], outline=ACCENT, width=3)
    d.text((60, 40), "At sea: SeaDronesSee, swimmers and boats", font=F_H2, fill=FG)
    d.text((1300, 610), "Inset: zoomed 4x, pixels not smoothed", font=F_TINY, fill=MUTED)
    y = 670
    for line in stats_md:
        d.text((1300, y), line, font=F_SMALL, fill=FG)
        y += 46
    x = 60
    for lab, col in SEA_COL.items():
        d.rectangle([x, oy + fitted.height + 24, x + 18, oy + fitted.height + 42], fill=col)
        d.text((x + 26, oy + fitted.height + 16), lab.replace("_", " "), font=F_TINY, fill=FG)
        x += 40 + 11 * len(lab)
    footer(d, "Test video never used for training · ground-truth labels · dataset CC0 1.0")
    return im


def fig_sea_ap(two: pd.DataFrame, native: pd.DataFrame):
    plt = mpl_dark()
    rows = [
        ("COCO weights\n640", two, "coco_y11n_640", "#90a4ae"),
        ("COCO weights\n1280", two, "coco_y11n_1280", "#90a4ae"),
        ("fine-tuned\n640", two, "ft_y11n_640", "#42a5f5"),
        ("fine-tuned\n1280", two, "ft_y11n_1280", "#ffc400"),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8.6))
    for ax, metric, ttl in ((a1, "AP_small", "AP, small objects"), (a2, "AP_swimmer", "AP, swimmers")):
        vals = [float(df[(df.tag == t) & (df.split == "test")][metric].iloc[0]) for _, df, t, _ in rows]
        b = ax.bar([r[0] for r in rows], vals, color=[r[3] for r in rows])
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=18)
        ax.set_title(ttl, fontsize=22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=16)
    fig.tight_layout()
    return fig


def fig_leakage(two: pd.DataFrame):
    plt = mpl_dark()
    fig, ax = plt.subplots(figsize=(18, 8.6))
    groups = [("COCO weights (never trained on sea data)", "coco_y11n_1280"), ("fine-tuned", "ft_y11n_1280")]
    w = 0.35
    for j, (split, lab, col) in enumerate(
        (("test", "videos never seen in training", "#42a5f5"), ("seen", "other frames of training videos", "#ff5252"))
    ):
        vals = []
        for _, t in groups:
            tag = t if split == "test" else t + "_seen"
            vals.append(float(two[(two.tag == tag) & (two.split == split)]["AP"].iloc[0]))
        xs = [i + (j - 0.5) * w for i in range(len(groups))]
        b = ax.bar(xs, vals, w * 0.92, color=col, label=lab)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=18)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel("AP (swimmer + boat)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=18)
    fig.tight_layout()
    return fig


def fig_geo_real(geo: pd.DataFrame):
    plt = mpl_dark()
    g = geo[(geo.object == "swimmer") & geo.model.isin(["nadir", "tilt-aware"])]
    clips = list(dict.fromkeys(g["clip"]))
    fig, ax = plt.subplots(figsize=(18, 8.6))
    w = 0.36
    for j, (model, col) in enumerate((("nadir", "#90a4ae"), ("tilt-aware", "#ffc400"))):
        vals = [float(g[(g["clip"] == c) & (g.model == model)]["apparent_speed_median_mps"].iloc[0]) for c in clips]
        xs = [i + (j - 0.5) * w for i in range(len(clips))]
        b = ax.bar(xs, vals, w * 0.92, color=col, label="straight-down camera" if model == "nadir" else "tilt-aware")
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=18)
    pitch = {c: g[g["clip"] == c]["gimbal_pitch_deg"].iloc[0].split("-")[0] for c in clips}
    ax.set_xticks(range(len(clips)), [f"{c}\npitch {pitch[c]}°" for c in clips])
    ax.axhline(1.0, color="#eceef1", ls="--", lw=1.5)
    ax.text(len(clips) - 0.5, 1.05, "a swimmer moves ≲ 1 m/s", ha="right", va="bottom", fontsize=16)
    ax.set_ylabel("apparent swimmer speed on the map (m/s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=18)
    fig.tight_layout()
    return fig


def scene_sea_track(frame_path, rec, telem, summ, clip, idx, n):
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
        d.rectangle([ox + x1 * sc, oy + y1 * sc, ox + x2 * sc, oy + y2 * sc], outline=col, width=3)
        d.text((ox + x1 * sc, oy + y1 * sc - 22), str(det["track_id"]), font=F_TINY, fill=col)
    d.text((40, 40), "Real footage: tracking swimmers and boats", font=F_H2, fill=FG)
    px = 1380
    d.text((px, 140), f"{clip} · frame {idx + 1}/{n}", font=F_SMALL, fill=MUTED)
    d.text((px, 180), f"altitude {telem['alt_m']:.0f} m · pitch {telem['pitch_deg']:.0f}°", font=F_SMALL, fill=MUTED)
    d.text((px, 220), f"drone speed {telem.get('speed_mps') or 0:.1f} m/s", font=F_SMALL, fill=MUTED)
    d.text((px, 290), "Whole clip", font=F_SMALL, fill=MUTED)
    d.text((px, 330), f"IDF1 {summ['IDF1']:.2f}", font=F_H1, fill=ACCENT)
    d.text((px, 430), f"{int(summ['id_switches'])} ID switches", font=F_SMALL, fill=FG)
    d.text((px, 470), f"recall {summ['recall']:.0%} · precision {summ['precision']:.0%}", font=F_SMALL, fill=FG)
    d.text((px, 530), f"{int(summ['gt_objects'])} labelled objects", font=F_SMALL, fill=FG)
    footer(d, "SeaDronesSee-MOT clip never used for training · real drone log (GPS, altitude, gimbal) per frame")
    return im


def add_sea_scenes(add, flush, R: Path, sea_hero: Path, sea_names, sea_clip: Path, sea_run: Path, sea_tag: str):
    stats = json.loads((R / "seadronessee_test_stats.json").read_text())
    lines = [
        f"{stats['n_boxes']:,} objects in {stats['n_images']:,} test images",
        "",
        "At 640 px model input:",
        f"median object {stats['median640']:.0f} px across",
        f"{stats['lt8_640']:.0%} under 8 px",
    ]
    add(
        scene_sea_intro(
            sea_hero, Path(str(sea_hero).replace("/images/", "/labels/")).with_suffix(".txt"), sea_names, lines
        ),
        7,
    )
    two = pd.read_csv(R / "seadronessee_two_class.csv").drop_duplicates(["tag", "split"], keep="last")
    native = pd.read_csv(R / "seadronessee.csv").drop_duplicates(["tag", "split"], keep="last")
    add(
        chart_scene(
            "At sea: what fine-tuning buys",
            fig_sea_ap(two, native),
            "Test = 7 videos never used for training · swimmer + boat (the classes COCO weights know) · CPU",
        ),
        8,
    )
    add(
        chart_scene(
            "The official split leaks: same videos in train and val",
            fig_leakage(two),
            "COCO weights are the control: their gap is image difficulty; the fine-tuned model's extra gap is leakage",
        ),
        8,
    )
    tr = pd.read_csv(R / "tracking_sea.csv").drop_duplicates("tag", keep="last")
    summ = tr[tr.tag == sea_tag].iloc[0].to_dict()
    run = [json.loads(x) for x in sea_run.read_text().splitlines()]
    telem = [json.loads(x) for x in (sea_clip / "telemetry.jsonl").read_text().splitlines()]
    frames = sorted((sea_clip / "frames").iterdir())
    for i, (rec, fp) in enumerate(zip(run, frames)):
        if i % 2:  # 30 fps footage shown at 15 fps
            continue
        add(scene_sea_track(fp, rec, telem[rec["frame"]], summ, sea_clip.name, i, len(frames)), 1 / 15)
    geo = pd.read_csv(R / "geo_real.csv")
    add(
        chart_scene(
            "Real footage: does the camera model put swimmers in the right place?",
            fig_geo_real(geo),
            "Ground-truth boxes + the drone's own log · lower = more plausible · 65.5° lens assumed",
        ),
        9,
    )
