"""v2 LinkedIn cut (1080x1350, 4:5, captions burned in): search planning, operator loop + TAK, thermal, text prompts.

Same rule as every video in this repo: real footage, real model output, numbers read from benchmarks/results.

  python docs/video/make_linkedin_v2.py --sea-mot outputs/sea_mot --runs outputs --review-clip outputs/review.mp4 \
      --thermal-dir datasets/hituav_yolo --out outputs/linkedin_v2.mp4
(review.mp4 comes from tools/screenshot_review.py --video)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_linkedin import (  # noqa: E402
    ACCENT,
    BG,
    FG,
    FPS,
    MUTED,
    REPO,
    H,
    R,
    W,
    blank,
    brand,
    clip_crop,
    font,
    footage_frame,
    stat_card,
    text_block,
    text_card,
)

GREEN = (46, 160, 67)


def chart_card(rows: list[tuple], title: str, sub: str, note: str):
    """Big horizontal bars, drawn directly (no matplotlib) so it matches the other cards.
    rows: (label, value, annotation, highlighted) or a plain string for a group header."""
    im, d = blank()
    y = text_block(d, (60, 110), title, font(52, "Bold"), width_chars=30)
    y = text_block(d, (60, y + 10), sub, font(30), fill=MUTED, width_chars=54)
    vmax = max(r[1] for r in rows if not isinstance(r, str))
    yy = max(y + 40, 400)
    for r in rows:
        if isinstance(r, str):
            d.text((60, yy), r, font=font(34, "Bold"), fill=ACCENT)
            yy += 60
            continue
        label, v, ann, hi = r
        d.text((60, yy), label, font=font(34), fill=FG)
        bw = max(int((W - 340) * v / vmax), 4)
        d.rectangle([60, yy + 46, 60 + bw, yy + 96], fill=ACCENT if hi else (120, 130, 140))
        d.text((80 + bw, yy + 50), ann, font=font(34, "Bold"), fill=FG)
        yy += 125
    text_block(d, (60, H - 230), note, font(27), fill=MUTED, width_chars=62)
    brand(d)
    return im


def route_frames(plan: dict, ex: dict, n: int):
    """Animate the exported QGroundControl mission being flown: tracks drawn progressively over a 20 m grid."""
    pts = [(it["params"][4], it["params"][5]) for it in plan["mission"]["items"] if it["command"] == 16]
    lat0 = pts[0][0]
    k = 111_320.0
    xy = [((lo - pts[0][1]) * k * math.cos(math.radians(lat0)), (la - pts[0][0]) * k) for la, lo in pts]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    pad = 40
    box = (100, 330, W - 100, H - 330)  # drawing area in pixels
    sx = (box[2] - box[0] - 2 * pad) / max(max(xs) - min(xs), 1)
    sy = (box[3] - box[1] - 2 * pad) / max(max(ys) - min(ys), 1)
    s = min(sx, sy)

    def px(p):
        return (box[0] + pad + (p[0] - min(xs)) * s, box[3] - pad - (p[1] - min(ys)) * s)

    seg = [(xy[i], xy[i + 1]) for i in range(len(xy) - 1)]
    lens = [math.dist(a, b) for a, b in seg]
    total = sum(lens)
    sweep = ex["track_spacing_m"]
    for f in range(n):
        im, d = blank()
        y = text_block(d, (60, 70), "Then it plans the flight", font(52, "Bold"), width_chars=30)
        text_block(
            d,
            (60, y + 4),
            f"{ex['area_km2']} km² of lake · {ex['alt_m']} m · {ex['tracks']} tracks {ex['track_spacing_m']:.0f} m apart",
            font(32),
            fill=MUTED,
            width_chars=52,
        )
        d.rectangle(box, outline=(44, 51, 61), width=2)
        gx = box[0]
        while gx < box[2]:
            d.line([(gx, box[1]), (gx, box[3])], fill=(28, 33, 40))
            gx += 50 * s
        gy = box[1]
        while gy < box[3]:
            d.line([(box[0], gy), (box[2], gy)], fill=(28, 33, 40))
            gy += 50 * s
        done = total * min(1.0, f / (n * 0.8))
        acc = 0.0
        cur = px(xy[0])
        for (a, b), L in zip(seg, lens):
            if acc >= done:
                break
            t = min(1.0, (done - acc) / L)
            e = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            # searched swath (camera footprint width ~ sweep width) under the drone path
            w = sweep * s / 2
            pa, pe = px(a), px(e)
            if abs(pa[0] - pe[0]) < 1:  # north-south track
                d.rectangle([pa[0] - w, min(pa[1], pe[1]), pa[0] + w, max(pa[1], pe[1])], fill=(40, 52, 30))
            d.line([pa, pe], fill=ACCENT, width=5)
            acc += L
            cur = pe
        d.ellipse([cur[0] - 12, cur[1] - 12, cur[0] + 12, cur[1] + 12], outline=ACCENT, width=4, fill=BG)
        text_block(
            d,
            (60, H - 300),
            f"Exported as a QGroundControl mission (PX4 / ArduPilot): {ex['path_km']} km, {ex['flight_min']:.0f} min, "
            f"{int(ex['pod'] * 100)}% probability of detection.",
            font(34),
            width_chars=48,
        )
        brand(d)
        yield im


def thermal_compare(img: Path, coco_model, ft_model, labels: Path):
    """Same night frame: colour-trained model vs thermal-trained model (real detections)."""
    im, d = blank()
    text_block(d, (60, 60), "At night: a thermal camera", font(52, "Bold"), width_chars=30)
    src = Image.open(img).convert("RGB")
    gts = []
    for line in labels.read_text().splitlines():
        c, cx, cy, bw, bh = map(float, line.split())
        if int(c) == 0:
            gts.append(((cx - bw / 2) * src.width, (cy - bh / 2) * src.height, (cx + bw / 2) * src.width,
                        (cy + bh / 2) * src.height))  # fmt: skip
    half_h = 430
    for k, (name, model, col) in enumerate(
        (("Model trained on colour photos", coco_model, MUTED), ("Fine-tuned on thermal images", ft_model, GREEN))
    ):
        r = model.predict(str(img), imgsz=640, conf=0.3, verbose=False)[0]
        people = [b for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()) if int(c) == 0]
        view = src.resize((int(src.width * half_h / src.height), half_h), Image.LANCZOS)
        x0, y0 = (W - view.width) // 2, 165 + k * (half_h + 90)
        im.paste(view, (x0, y0))
        sc = half_h / src.height
        for x1, y1, x2, y2 in people:
            d.rectangle([x0 + x1 * sc - 3, y0 + y1 * sc - 3, x0 + x2 * sc + 3, y0 + y2 * sc + 3], outline=col, width=3)
        found = sum(any(_iou(g, b) >= 0.3 for b in people) for g in gts)  # labelled people with a box on them
        d.text((60, y0 + half_h + 12), f"{name}: {found} of {len(gts)} people", font=font(32, "Bold"), fill=col)
    text_block(
        d,
        (60, H - 145),
        "Real HIT-UAV night frame (80 m) from a flight never used for training. A person counts as found if a box "
        "overlaps them (IoU ≥ 0.3).",
        font(24),
        fill=MUTED,
        width_chars=70,
    )
    brand(d)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sea-mot", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--review-clip", required=True, help="sea operator session (tools/screenshot_review.py --video)")
    ap.add_argument("--review-clip-land", required=True, help="land operator session")
    ap.add_argument("--land-flight", required=True, help="simulated flight folder over a real VisDrone frame")
    ap.add_argument("--land-run", required=True, help="tracking JSONL for that flight")
    ap.add_argument("--thermal-dir", required=True)
    ap.add_argument("--thermal-image", default="1_80_50_0_01218.jpg")
    ap.add_argument("--out", default=str(REPO / "outputs/linkedin_v2.mp4"))
    a = ap.parse_args()

    sa = pd.read_csv(R / "search_altitude.csv")
    sa = sa[(sa.size_m == 0.5) & (sa.fps == 4.0)].set_index("alt_m")
    looks = pd.read_csv(R / "search_looks.csv").set_index("frames")
    summ = json.loads((R / "sea_detection_summary.json").read_text())
    ex = json.loads((R / "search_example.json").read_text())["example"]
    plan = json.loads((REPO / "samples/search_example.plan").read_text())
    al = pd.read_csv(R / "alert_load.csv")
    al = al[al["clip"] == "all"].set_index("dedupe_radius_m")
    th = pd.read_csv(R / "hituav_shared.csv").set_index("tag")
    pr = pd.read_csv(R / "seadronessee_prompts_two_class.csv").set_index("tag")
    two = pd.read_csv(R / "seadronessee_two_class.csv").set_index("tag")
    lsa = pd.read_csv(R / "land_search_altitude.csv").set_index(["hfov_deg", "alt_m"])
    lsum = json.loads((R / "land_detection_summary.json").read_text())
    vp = pd.concat(
        [pd.read_csv(R / "visdrone_prompts_shared.csv"), pd.read_csv(R / "visdrone_prompts_prompts_shared.csv")]
    )
    vp = vp.drop_duplicates("tag", keep="last").set_index("tag")

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS)]
        + ["-i", "-", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE,
    )

    def hold(img, secs):
        b = img.tobytes()
        for _ in range(int(secs * FPS)):
            ff.stdin.write(b)

    # 1. hook: real footage
    recs = [json.loads(x) for x in (Path(a.runs) / "sea_track_DJI_0055_bytetrack.jsonl").read_text().splitlines()]
    frames = sorted((Path(a.sea_mot) / "DJI_0055" / "frames").iterdir())
    crop = clip_crop(recs)
    for rec in recs[:110]:
        ff.stdin.write(
            footage_frame(
                frames[rec["frame"]],
                rec,
                crop,
                "Version 2: finding people at sea…",
                "Search planning · operator review · alerts on TAK maps · night-time thermal",
                None,
                ids=False,
            ).tobytes()
        )
    lrecs = [json.loads(x) for x in Path(a.land_run).read_text().splitlines()]
    lframes = sorted((Path(a.land_flight) / "frames").iterdir())
    w0, h0 = Image.open(lframes[0]).size
    for rec in lrecs[:90]:
        im = footage_frame(
            lframes[rec["frame"]],
            rec,
            (0, 0, w0, h0),
            "…and on land",
            "Same pipeline over cities: people and vehicles, tracked and placed on the map.",
            None,
            ids=False,
        )
        d = ImageDraw.Draw(im)
        d.rectangle([W - 380, 290, W, 330], fill=BG)  # replace the "real drone footage" tag: this one is simulated
        d.text((W - 370, 296), "simulated flight, real image", font=font(24), fill=(255, 255, 255))
        ff.stdin.write(im.tobytes())

    # 2. how high to fly
    def km(v):
        return f"{v:.2f} km²/h"

    rows = [
        "At sea, by day (colour camera, swimmer)",
        ("20 m", float(sa.loc[20, "km2_h_at_pod80"]), km(sa.loc[20, "km2_h_at_pod80"]), False),
        ("100 m", float(sa.loc[100, "km2_h_at_pod80"]), km(sa.loc[100, "km2_h_at_pod80"]), True),
        "On land, at night (thermal camera, person)",
        ("40 m", float(lsa.loc[(25.0, 40), "km2_h_at_pod80"]), km(lsa.loc[(25.0, 40), "km2_h_at_pod80"]), False),
        ("120 m", float(lsa.loc[(25.0, 120), "km2_h_at_pod80"]), km(lsa.loc[(25.0, 120), "km2_h_at_pod80"]), True),
    ]
    hold(
        chart_card(
            rows,
            "How high should a search drone fly?",
            "Area searched per hour at 80% probability of detection, 10 m/s. Measured, not assumed.",
            f"Sea: {summ['recall']:.0%} of {summ['swimmers']:,} unseen swimmers get a box, even at 4–6 px; 1 s of video ≈ "
            f"{looks.loc[30, 'independent_looks_equivalent']:.1f} independent looks. Land at night: "
            f"{lsum['recall']:.0%} of {lsum['persons']:,} people; lens choice matters as much as altitude.",
        ),
        8.5,
    )
    # 3. the plan
    for im in route_frames(plan, ex, int(5.5 * FPS)):
        ff.stdin.write(im.tobytes())
    # 4. operator review (real map app, recorded)
    clips = []
    for clip, start, dur, kind in ((a.review_clip_land, 5.5, 5.0, "land"), (a.review_clip, 3.5, 6.0, "sea")):
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", clip, "-vf",
                        f"fps={FPS}", str(tmp / "%04d.png")], check=True)  # fmt: skip
        clips += [(p, kind) for p in sorted(tmp.iterdir())]
    for p, kind in clips:
        im, d = blank()
        m = Image.open(p).convert("RGB")
        m = m.crop((int(m.width * 0.42), int(m.height * 0.12), m.width, m.height))  # swimmers + alert panel
        sc = min(W / m.width, 780 / m.height)
        m = m.resize((int(m.width * sc), int(m.height * sc)), Image.LANCZOS)
        im.paste(m, ((W - m.width) // 2, 230))
        text_block(d, (60, 70), "A person stays in charge", font(52, "Bold"), width_chars=30)
        d.text((60, 150), "on land" if kind == "land" else "at sea", font=font(34), fill=MUTED)
        y = 230 + m.height + 35
        head = (
            "Confirm or reject every alert"
            if kind == "land"
            else f"Duplicate alerts: {int(al.loc[0, 'alerts'])} → {int(al.loc[3, 'alerts'])}, no swimmer lost"
        )
        d.text((60, y), head, font=font(38, "Bold"), fill=ACCENT)
        text_block(
            d,
            (60, y + 70),
            "A rejected spot stops alerting; every decision is logged and shared with every operator.",
            font(34),
            width_chars=48,
        )
        brand(d)
        ff.stdin.write(im.tobytes())
    # 5. TAK
    cot = (
        '<event type="a-u-S" uid="esa-drone-1-swimmer-5">  sea\n'
        '  <point lat="47.671774" lon="9.269403" ce="15.0"/>\n\n'
        '<event type="a-u-G" uid="esa-drone-1-person-172"> land\n'
        '  <point lat="60.149994" lon="24.950095" ce="15.0"/>'
    )
    im = text_card(
        "Alerts go to the map rescue teams already use",
        "ATAK / WinTAK / TAK Server, as Cursor-on-Target markers with position uncertainty. A rejected alert deletes "
        "its marker. Tested in CI against a real TAK server.",
        "",
    )
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 760, W - 60, 1010], radius=12, fill=(24, 28, 34))
    mono = font(26) if not Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf").exists() else None
    if mono is None:
        from PIL import ImageFont

        mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26)
    d.multiline_text((90, 790), cot, font=mono, fill=(160, 220, 160), spacing=12)
    hold(im, 5)
    # 6. thermal
    from ultralytics import YOLO

    td = Path(a.thermal_dir)
    img = td / "images/test" / a.thermal_image
    hold(
        thermal_compare(
            img,
            _CocoPersonOnly(YOLO(str(REPO / "weights/yolo11n.pt"))),
            YOLO(str(REPO / "models/hit_y11n_ep5.pt")),
            td / "labels/test" / (img.stem + ".txt"),
        ),
        5.5,
    )
    hold(
        stat_card(
            "Thermal works better at night than by day, the opposite of a normal camera.",
            f"{th.loc['coco_y11n_640_test', 'AP']:.2f}",
            f"{th.loc['ft_y11n_640_test', 'AP']:.2f}",
            "accuracy (AP): colour-trained → fine-tuned on thermal",
            f"Night {th.loc['ft_y11n_640_test_night', 'AP']:.2f} vs day {th.loc['ft_y11n_640_test_day', 'AP']:.2f}. "
            "Five CPU epochs on HIT-UAV, whole flights held out of training.",
            "477 test images from flight conditions never trained on (the dataset's own split mixes frames ~1 s apart).",
        ),
        6,
    )

    # 7. text prompts
    def ap_(v):
        return f"{v:.2f}"

    sea_txt, sea_ft = pr.loc["yoloe11m_p3_synonyms_test", "AP_swimmer"], two.loc["ft_y11n_1280", "AP_swimmer"]
    land_txt, land_ft = vp.loc["yoloe11m_land", "AP_pedestrian"], vp.loc["ft_y11n_1280", "AP_pedestrian"]
    hold(
        chart_card(
            [
                "At sea: swimmer AP",
                ("text prompts (YOLOE-11m, no training)", float(sea_txt), ap_(sea_txt), False),
                ("small model, fine-tuned", float(sea_ft), ap_(sea_ft), True),
                "On land: pedestrian AP",
                ("text prompts (YOLOE-11m, no training)", float(land_txt), ap_(land_txt), False),
                ("small model, fine-tuned", float(land_ft), ap_(land_ft), True),
            ],
            "Can you just type “find a person”?",
            "Open-vocabulary detection, measured against small fine-tuned models on unseen test images.",
            "The text-prompt model is 3× larger and needs ~10× the compute. Text search stays as a tool for things the "
            "trained model has no class for.",
        ),
        6.5,
    )
    # 8. close
    hold(
        text_card(
            "Open source. Every number measured, and checked in CI.",
            "128 README numbers are tested against the results files on every push.\n\nNext: NVIDIA Jetson, "
            "measuring watts and joules at the frame rate a search really needs.",
            "CPU measurements so far. Research project on public data, not for operational rescue use.",
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
    print("linkedin v2 ->", a.out, f"({float(dur):.1f} s)")


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class _CocoPersonOnly:
    """COCO model as used in the thermal baseline: only its 'person' class (id 0) counts as a person."""

    def __init__(self, m):
        self.m = m

    def predict(self, *args, **kw):
        return self.m.predict(*args, **kw)


if __name__ == "__main__":
    main()
