"""45-second LinkedIn cut (1080x1350, 4:5, captions burned in because feeds autoplay muted).

Same rule as the full demo: only real footage, real model output and numbers read from
benchmarks/results/*.csv.

  python docs/video/make_linkedin.py --sea-mot outputs/sea_mot --runs outputs --map-clip outputs/map_sea_trim.mp4 \
      --land-hero datasets/visdrone_yolo/images/val/0000295_02400_d_0000033.jpg --out outputs/linkedin.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
R = REPO / "benchmarks" / "results"
W, H, FPS = 1080, 1350, 30
BG, FG, MUTED, ACCENT, RED, BLUE = (
    (14, 17, 22),
    (236, 238, 241),
    (150, 158, 170),
    (255, 196, 0),
    (255, 82, 82),
    (66, 165, 245),
)
COL = {"swimmer": RED, "boat": BLUE, "jetski": (186, 104, 200), "buoy": (102, 187, 106)}
INTER = Path("/usr/share/fonts/opentype/inter")


def font(size, weight="Regular"):
    p = INTER / f"Inter-{weight}.otf"
    return ImageFont.truetype(str(p if p.exists() else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), size)


def text_block(d, xy, text, f, fill=FG, width_chars=30, spacing=10):
    x, y = xy
    for para in text.split("\n"):
        for line in textwrap.wrap(para, width_chars) or [""]:
            d.text((x, y), line, font=f, fill=fill)
            y += f.size + spacing
    return y


def blank():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def brand(d):
    d.text((60, H - 70), "github.com/mustafarkhan/edge-situational-awareness", font=font(26), fill=MUTED)


def footage_frame(
    frame: Path, rec: dict, crop: tuple[int, int, int, int], top: str, bottom: str, badge: str | None, ids: bool = True
):
    """A tracking frame: the camera image cropped to 4:3, boxes from the real model output, captions."""
    im, d = blank()
    src = Image.open(frame).convert("RGB")
    x0, y0, x1, y1 = crop
    view = src.crop(crop).resize((W, int(W * (y1 - y0) / (x1 - x0))), Image.LANCZOS)
    s = W / (x1 - x0)
    oy = 280
    im.paste(view, (0, oy))
    for det in rec["detections"]:
        bx1, by1, bx2, by2 = det["bbox"]
        c = COL.get(det["label"], ACCENT)
        r = [(bx1 - x0) * s, oy + (by1 - y0) * s, (bx2 - x0) * s, oy + (by2 - y0) * s]
        if r[2] < 0 or r[0] > W or r[3] < oy or r[1] > oy + view.height:
            continue
        pad = 4
        d.rectangle([r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad], outline=c, width=4)
        if ids and det.get("track_id") is not None:
            d.text((r[0] - pad, r[1] - pad - 30), str(det["track_id"]), font=font(26, "Bold"), fill=c)
    text_block(d, (60, 70), top, font(52, "Bold"), width_chars=30)
    y = oy + view.height + 40
    if badge:
        d.text((60, y), badge, font=font(40, "Bold"), fill=ACCENT)
        y += 60
    text_block(d, (60, y), bottom, font(36), fill=FG, width_chars=44)
    d.text((W - 360, oy + 16), "real drone footage", font=font(24), fill=(255, 255, 255))
    brand(d)
    return im


def stat_card(kicker: str, before: str, after: str, unit: str, body: str, note: str):
    im, d = blank()
    text_block(d, (60, 140), kicker, font(48, "Bold"), width_chars=32)
    d.text((60, 470), before, font=font(150, "Bold"), fill=MUTED)
    bw = d.textlength(before, font=font(150, "Bold"))
    d.text((60 + bw + 30, 500), "→", font=font(110), fill=FG)
    d.text((60 + bw + 160, 470), after, font=font(150, "Bold"), fill=ACCENT)
    d.text((60, 660), unit, font=font(34), fill=MUTED)
    text_block(d, (60, 780), body, font(40), width_chars=40)
    text_block(d, (60, H - 200), note, font(26), fill=MUTED, width_chars=62)
    brand(d)
    return im


def text_card(title: str, body: str, note: str = "", accent_line: str = ""):
    im, d = blank()
    y = text_block(d, (60, 180), title, font(58, "Bold"), width_chars=28)
    if accent_line:
        y = text_block(d, (60, y + 30), accent_line, font(44, "Bold"), fill=ACCENT, width_chars=36)
    text_block(d, (60, y + 40), body, font(38), width_chars=44, spacing=14)
    if note:
        text_block(d, (60, H - 200), note, font(26), fill=MUTED, width_chars=62)
    brand(d)
    return im


def clip_crop(recs, label="swimmer", w=1500, h=1000, src_w=1920, src_h=1080):
    """3:2 crop centred on the given objects in the first frames (keeps tiny swimmers visible)."""
    xs = [(d["bbox"][0] + d["bbox"][2]) / 2 for r in recs[:10] for d in r["detections"] if d["label"] == label]
    cx = sorted(xs)[len(xs) // 2] if xs else src_w / 2
    cy = sorted((d["bbox"][1] + d["bbox"][3]) / 2 for r in recs[:10] for d in r["detections"])
    cy = cy[len(cy) // 2] if cy else src_h / 2
    x0 = int(min(max(cx - w / 2, 0), src_w - w))
    y0 = int(min(max(cy - h / 2, 0), src_h - h))
    return (x0, y0, x0 + w, y0 + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sea-mot", required=True)
    ap.add_argument("--runs", required=True, help="folder with sea_track_<clip>_bytetrack.jsonl")
    ap.add_argument("--map-clip", required=True)
    ap.add_argument("--land-hero", required=True)
    ap.add_argument("--out", default=str(REPO / "outputs/linkedin.mp4"))
    a = ap.parse_args()

    two = pd.read_csv(R / "seadronessee_two_class.csv").set_index("tag")
    nat = pd.read_csv(R / "seadronessee.csv").set_index("tag")
    trk = pd.read_csv(R / "tracking_sea.csv").set_index("tag")
    geo = pd.read_csv(R / "geo_real.csv")
    z = pd.read_csv(R / "zeroshot.csv").drop_duplicates("tag", keep="last").set_index("tag")
    tr = pd.read_csv(R / "tracking.csv").set_index("tag")
    stats = json.loads((R / "seadronessee_test_stats.json").read_text())

    sw_coco, sw_ft = two.loc["coco_y11n_640", "AP_swimmer"], nat.loc["ft_y11n_1280", "AP_swimmer"]
    sw_seen = nat.loc["ft_y11n_1280_seen", "AP_swimmer"]
    g = geo[(geo["clip"] == "DJI_0055") & (geo.object == "swimmer")].set_index("model")
    g59 = geo[(geo["clip"] == "DJI_0059") & (geo.object == "swimmer")].set_index("model")

    tmp = Path(tempfile.mkdtemp())
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS)]
        + ["-i", "-", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE,
    )

    def hold(img, secs):
        b = img.tobytes()
        for _ in range(int(secs * FPS)):
            ff.stdin.write(b)

    def footage(clip, top, bottom, badge, n=None, ids=True):
        recs = [json.loads(x) for x in (Path(a.runs) / f"sea_track_{clip}_bytetrack.jsonl").read_text().splitlines()]
        frames = sorted((Path(a.sea_mot) / clip / "frames").iterdir())
        crop = clip_crop(recs)
        for rec in recs[:n]:
            ff.stdin.write(footage_frame(frames[rec["frame"]], rec, crop, top, bottom, badge, ids).tobytes())

    # 1. hook (real clip DJI_0055, 15 swimmers at 20 m)
    footage(
        "DJI_0055",
        "Can a drone find a swimmer in the sea, by itself, in real time?",
        f"At the detector's default input, half of all objects in this dataset are under "
        f"{stats['median640']:.0f} pixels wide.",
        None,
        n=120,
        ids=False,
    )
    # 2. tracking on a clip never used for training
    t59 = trk.loc["sea_DJI_0059_bytetrack"]
    footage(
        "DJI_0059",
        "Detect, track, and keep one ID per person",
        f"Clip never used for training. {int(t59.gt_objects)} labelled objects, {int(t59.id_switches)} ID switches "
        f"in {int(t59.frames)} frames.",
        f"Identity accuracy (IDF1) {t59.IDF1:.2f}",
        n=143,
    )
    # 3. fine-tuning
    hold(
        stat_card(
            "Off-the-shelf models barely see swimmers. Training on sea footage changes that.",
            f"{sw_coco:.2f}",
            f"{sw_ft:.2f}",
            "swimmer accuracy (COCO AP)",
            "A small YOLO11n model, fine-tuned for 4 epochs on a laptop-class CPU.",
            "SeaDronesSee (CC0), 1,200 test images from 7 videos the model never saw. "
            "COCO weights at 640 px vs fine-tuned at 1280 px.",
        ),
        5.5,
    )
    # 4. leakage
    hold(
        stat_card(
            "The benchmark's own validation split would have flattered my model.",
            f"{sw_ft:.2f}",
            f"{sw_seen:.2f}",
            "swimmer AP: unseen videos  →  frames from training videos",
            "The official split shares videos between training and validation. "
            "After a control for scene difficulty, it overstates swimmer accuracy by about 50%.",
            "Control: a COCO model, which saw neither set, gains only "
            f"{two.loc['coco_y11n_1280_seen', 'AP_swimmer'] - two.loc['coco_y11n_1280', 'AP_swimmer']:+.3f}.",
        ),
        6.5,
    )
    # 5. map (real app, real flight log)
    mapdir = tmp / "map"
    mapdir.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            "2",
            "-i",
            a.map_clip,
            "-t",
            "7",
            "-vf",
            f"fps={FPS}",
            str(mapdir / "%04d.png"),
        ],
        check=True,
    )
    for p in sorted(mapdir.iterdir()):
        im, d = blank()
        m = Image.open(p).convert("RGB")
        m = m.resize((W, int(m.height * W / m.width)), Image.LANCZOS)
        im.paste(m, (0, 330))
        text_block(d, (60, 70), "From pixels to positions on a map", font(52, "Bold"), width_chars=30)
        text_block(
            d,
            (60, 330 + m.height + 40),
            "Each swimmer (red) and boat (blue) placed using the drone's own GPS, height and camera tilt. "
            "Alerts fire only after a swimmer has been tracked for 15 frames.",
            font(34),
            width_chars=48,
        )
        brand(d)
        ff.stdin.write(im.tobytes())
    # 6. geolocation honesty
    hold(
        stat_card(
            "Accounting for camera tilt makes swimmers stay still on the map.",
            f"{g.loc['nadir', 'apparent_speed_median_mps']:.1f}",
            f"{g.loc['tilt-aware', 'apparent_speed_median_mps']:.1f}",
            "swimmer speed on the map, m/s (real drift < 1)",
            f"But on another flight it got worse ({g59.loc['nadir', 'apparent_speed_median_mps']:.1f} → "
            f"{g59.loc['tilt-aware', 'apparent_speed_median_mps']:.1f} m/s): the drone's attitude log didn't match "
            "the video. Better geometry can't fix bad telemetry.",
            "No ground-truth positions exist for real footage, so this is a physical plausibility test.",
        ),
        7,
    )
    # 7. land
    im, d = blank()
    hero = Image.open(a.land_hero).convert("RGB")
    hero = hero.crop((0, 0, hero.width, min(hero.height, int(hero.width * 9 / 16))))
    hero = hero.resize((W, int(W * hero.height / hero.width)), Image.LANCZOS)
    im.paste(hero, (0, 330))
    text_block(d, (60, 70), "Same pipeline over cities (VisDrone)", font(52, "Bold"), width_chars=30)
    y = 330 + hero.height + 40
    d.text(
        (60, y),
        f"Tiny-object accuracy {z.loc['y11n_640', 'AP_small']:.2f} → {z.loc['ft_y11n_1280', 'AP_small']:.2f}",
        font=font(40, "Bold"),
        fill=ACCENT,
    )
    d.text(
        (60, y + 64),
        f"Stable IDs on a fast camera {tr.loc['fast_bytetrack', 'IDF1']:.2f} → {tr.loc['fast_botsort', 'IDF1']:.2f}",
        font=font(40, "Bold"),
        fill=ACCENT,
    )
    text_block(
        d, (60, y + 140), "INT8: zero detections until the detection head stayed FP32.", font(34), width_chars=48
    )
    brand(d)
    hold(im, 5)
    # 8. close
    hold(
        text_card(
            "Open source. Every number measured on public data.",
            "ROS 2 node with live drone telemetry, tested in CI · map app · failure analysis · experiment log with "
            "the bugs.\n\nNext: NVIDIA Jetson — speed, watts, joules per frame.",
            "CPU measurements so far. Research project, not for operational rescue use.",
            "github.com/mustafarkhan/edge-situational-awareness",
        ),
        5,
    )
    ff.stdin.close()
    ff.wait()
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.out],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print("linkedin ->", a.out, f"({float(dur):.1f} s)")


if __name__ == "__main__":
    main()
