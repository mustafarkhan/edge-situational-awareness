"""The v3 project video: the whole system, from the search plan to the operator, built only from real outputs.

  python docs/video/make_demo.py --sitl-track out/sitl_sea_track.csv --sitl-report out/sitl_sea.json \
      --land-track out/sitl_land_track.csv --land-report out/sitl_land.json \
      --refusal out/refusal.txt --review-clip out/review_v3.mp4 --dual-clip out/dual_v3.mp4 \
      --sea-clip outputs/sea_mot/DJI_0055 --sea-run outputs/gpu_track/sea_track_DJI_0055_bytetrack.jsonl \
      --thermal-dir datasets/hituav_yolo --land-dir datasets/demo_holdout --land-model weights/vis_y11n_640_gpu.pt \
      --out outputs/project_v3.mp4            # add --short for the 75-second cut

Inputs come from the repository's own tools: scripts/sitl_fly.sh (simulated flight + track), tools/fly_mission.py
(the refusal), tools/screenshot_review.py (operator sessions), tools/detect_video.py (tracking), and
benchmarks/results (every number on screen). Nothing is typed in by hand except the captions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
from make_video import (  # noqa: E402
    ACCENT,
    ALERT,
    BG,
    F_BODY,
    F_H1,
    F_H2,
    F_SMALL,
    F_TINY,
    FG,
    MUTED,
    canvas,
    fit,
    footer,
    mpl_dark,
)
from PIL import Image  # noqa: E402
from sea_scenes import scene_sea_track  # noqa: E402

from esa.cot import alert_event  # noqa: E402
from esa.mission import MAX_PLAN_ALT_M  # noqa: E402

FPS = 30
GREEN, BLUE = (102, 187, 106), (66, 165, 245)
PHASE_COL = {"transit": MUTED, "search": ACCENT, "return": BLUE}
HONEST = "Datasets and simulation · not yet flown on real hardware · field tests are the next stage"


def wrap(d, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if d.textlength(trial, font=fnt) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def text_block(d, xy, lines, fnt=F_BODY, fill=FG, gap=62, bullet=True, width=1680):
    """Bulleted lines, wrapped to `width` px; an empty string is a half-height gap without a bullet."""
    x, y = xy
    indent = 45 if bullet else 0
    for ln in lines:
        if not ln:
            y += gap // 2
            continue
        for k, part in enumerate(wrap(d, ln, fnt, width - indent)):
            if bullet and k == 0:
                d.text((x, y), "•", font=fnt, fill=ACCENT)
            d.text((x + indent, y), part, font=fnt, fill=fill)
            y += gap
    return y


# ---------------------------------------------------------------- scenes
def scene_title(repo_url):
    im, d = canvas()
    d.text((120, 300), "Drone search and rescue", font=F_H1, fill=FG)
    d.text((120, 395), "that rescuers can plan with", font=F_H1, fill=ACCENT)
    d.rectangle([120, 520, 320, 526], fill=ACCENT)
    d.text((120, 570), "Land, sea and night · RGB + thermal · plans flown in simulation · built for NVIDIA Jetson",
           font=F_SMALL, fill=FG)  # fmt: skip
    d.text((120, 620), "Open source · every number checked against its results file in CI", font=F_SMALL, fill=MUTED)
    d.text((120, 940), repo_url, font=F_SMALL, fill=MUTED)
    return im


def scene_question():
    im, d = canvas()
    d.text((120, 150), "The question a search manager asks", font=F_H2, fill=FG)
    for k, ln in enumerate(("Where should the drone fly, how high,", "and how sure are we that it",
                            "would have seen someone?")):  # fmt: skip
        d.text((120, 240 + 88 * k), ln, font=F_H1, fill=ACCENT)
    text_block(d, (120, 560), [
        "Searches are planned in sweep width and probability of detection.",
        "For a person searching forest: 64 m wide by day, 22 m at night (Koester 2023).",
        "AI detectors are reported in AP on datasets, which a search plan cannot use.",
        "This system measures detection and turns it into the plan, the flight and the alerts.",
    ])  # fmt: skip
    footer(
        d, "Koester, Journal of Search and Rescue (2023): effective sweep width, temperate forest, adult-sized target"
    )
    return im


def scene_figure(title, png: Path, lines, note):
    im, d = canvas()
    d.text((60, 40), title, font=F_H2, fill=FG)
    fig = Image.open(png).convert("RGB")
    fig, _ = fit(fig, 1250, 860)
    im.paste(fig, (60, 140))
    text_block(d, (1360, 180), lines, fnt=F_SMALL, gap=50, width=520)
    footer(d, note)
    return im


def scene_card(kicker, title, lines, note=HONEST):
    im, d = canvas()
    d.text((120, 120), kicker.upper(), font=F_SMALL, fill=ACCENT)
    d.text((120, 170), title, font=F_H2, fill=FG)
    text_block(d, (120, 300), lines)
    footer(d, note)
    return im


class MissionMap:
    """Planned waypoints, geofence and the flown track, in metres around home."""

    def __init__(self, plan: dict, track: list[dict], box=(60, 140, 1260, 1040)):
        self.lat0, self.lon0 = plan["mission"]["plannedHomePosition"][:2]
        self.k = math.cos(math.radians(self.lat0))
        self.wps = [(it["params"][4], it["params"][5]) for it in plan["mission"]["items"] if it["command"] == 16]
        self.fence = plan["geoFence"]["polygons"][0]["polygon"]
        pts = [self.xy(a, b) for a, b in self.fence]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0, x1, y1 = box
        self.s = 0.92 * min((x1 - x0) / (max(xs) - min(xs)), (y1 - y0) / (max(ys) - min(ys)))
        self.cx, self.cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        self.ox, self.oy = (x0 + x1) / 2, (y0 + y1) / 2
        self.track = track

    def xy(self, lat, lon):
        return (lon - self.lon0) * 111_320 * self.k, (lat - self.lat0) * 111_320

    def px(self, lat, lon):
        x, y = self.xy(lat, lon)
        return self.ox + (x - self.cx) * self.s, self.oy - (y - self.cy) * self.s

    def draw(self, d, upto: int):
        f = [self.px(a, b) for a, b in self.fence]
        d.line(f + [f[0]], fill=ALERT, width=3)
        w = [self.px(a, b) for a, b in self.wps]
        for (xa, ya), (xb, yb) in zip(w, w[1:]):  # planned path, dashed
            n = max(1, int(math.hypot(xb - xa, yb - ya) / 14))
            for i in range(0, n, 2):
                t0, t1 = i / n, min(1.0, (i + 1) / n)
                d.line([(xa + (xb - xa) * t0, ya + (yb - ya) * t0), (xa + (xb - xa) * t1, ya + (yb - ya) * t1)],
                       fill=(90, 98, 110), width=2)  # fmt: skip
        hx, hy = self.px(self.lat0, self.lon0)
        d.rectangle([hx - 9, hy - 9, hx + 9, hy + 9], outline=FG, width=3)
        pts = self.track[: upto + 1]
        for a, b in zip(pts, pts[1:]):
            d.line([self.px(a["lat"], a["lon"]), self.px(b["lat"], b["lon"])], fill=PHASE_COL[b["phase"]], width=4)
        if pts:
            x, y = self.px(pts[-1]["lat"], pts[-1]["lon"])
            d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=BG, outline=ACCENT, width=4)


def mission_frames(plan, track, report, seconds=14.0, title="Plan, upload, fly: the whole sortie in ArduPilot's simulator",
                   note="samples/search_example.plan · ArduCopter 4.7.2 SITL · tools/fly_mission.py · flown in CI on every change"):  # fmt: skip
    mm = MissionMap(plan, track)
    n = int(seconds * FPS)
    tmax = float(track[-1]["t_s"])
    idx, j = [], 0
    for f in range(n):  # sample the track evenly in simulated time
        t = tmax * f / (n - 1)
        while j < len(track) - 1 and float(track[j + 1]["t_s"]) <= t:
            j += 1
        idx.append(j)
    for f, i in enumerate(idx):
        im, d = canvas()
        d.text((60, 40), title, font=F_H2, fill=FG)
        mm.draw(d, i)
        p = track[i]
        px = 1340
        d.text((px, 150), f"sim time {float(p['t_s']) / 60:4.1f} min", font=F_BODY, fill=FG)
        d.text((px, 210), f"height {float(p['alt_m']):5.1f} m", font=F_BODY, fill=FG)
        phase = {"transit": "take-off", "search": "searching", "return": "return and landing"}[p["phase"]]
        d.text((px, 270), phase, font=F_BODY, fill=PHASE_COL[p["phase"]])
        y = 370
        for col, lab in ((ALERT, "geofence (uploaded, enforced by the autopilot)"), ((90, 98, 110), "planned tracks"),
                         (ACCENT, "flown: search"), (BLUE, "flown: return home")):  # fmt: skip
            d.rectangle([px, y + 12, px + 40, y + 18], fill=col)
            d.text((px + 55, y), lab, font=F_TINY, fill=MUTED)
            y += 40
        if f > 0.9 * n:
            r = report
            y = 590
            for ln in (f"{r['waypoints_reached']}/{r['waypoints_planned']} search waypoints reached",
                       f"max {r['cross_track_m_max']:.2f} m from the planned tracks",
                       f"{r['positions_outside_fence']} positions outside the fence",
                       f"highest {r['max_alt_m']:.0f} m (ceiling 120 m)",
                       "landed" if r["landed"] else "NOT landed"):  # fmt: skip
                d.text((px, y), "✓", font=F_BODY, fill=GREEN)
                d.text((px + 45, y), ln, font=F_SMALL, fill=FG)
                y += 52
        footer(d, note)
        yield im


def scene_safety(refusal: str, settings: dict):
    im, d = canvas()
    d.text((60, 40), "Flight safety: the autopilot enforces the limits", font=F_H2, fill=FG)
    d.rounded_rectangle([60, 150, 940, 560], 14, fill=(24, 28, 35))
    d.text((90, 175), "$ fly_mission.py --plan plan_at_120m.plan", font=F_TINY, fill=MUTED)
    y = 230
    for ln in refusal.strip().splitlines():
        d.text((90, y), ln, font=F_SMALL, fill=ALERT if "refused" in ln else FG)
        y += 46
    d.text((90, 470), "Checked before anything is sent to the drone.", font=F_TINY, fill=MUTED)
    d.rounded_rectangle([1000, 150, 1860, 560], 14, fill=(24, 28, 35))
    d.text((1030, 175), "set on the autopilot and read back (SITL):", font=F_TINY, fill=MUTED)
    y = 230
    for k, v in settings.items():
        d.text((1030, y), f"{k} = {v:g}", font=F_SMALL, fill=FG)
        y += 46
    text_block(d, (60, 640), [
        "Geofence ceiling at the EU 120 m limit; plans stay 5 m below it.",
        "Found in simulation: a plan at exactly 120 m tripped the fence and aborted.",
        "Return home at the search altitude, not ArduCopter's 15 m default.",
        "The detector only reads telemetry: it never commands the drone.",
    ], fnt=F_SMALL, gap=56)  # fmt: skip
    footer(d, "src/esa/mission.py · tools/fly_mission.py · docs/DEPLOYMENT.md")
    return im


PEOPLE = {"person", "pedestrian", "people", "swimmer"}


def scene_detect(title, images: list[Path], model_path: Path, imgsz: int, captions: list[str], lines: list[str],
                 note: str, conf: float = 0.3):  # fmt: skip
    """Three images side by side with the model's boxes: people in red, everything else in blue."""
    from PIL import ImageDraw
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    im, d = canvas()
    d.text((60, 40), title, font=F_H2, fill=FG)
    x = 60
    for p, cap in zip(images, captions):
        src = Image.open(p).convert("RGB")
        r = model.predict(str(p), imgsz=imgsz, conf=conf, verbose=False)[0]
        tile, s = fit(src, 580, 470)
        td = ImageDraw.Draw(tile)
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()):
            col = ALERT if r.names[c] in PEOPLE else BLUE
            td.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline=col, width=2)
        im.paste(tile, (x, 170))
        d.text((x, 660), cap, font=F_TINY, fill=MUTED)
        x += 610
    text_block(d, (60, 740), lines, fnt=F_SMALL, gap=56)
    footer(d, note)
    return im


def scene_results(R: Path):
    """Detection on held-out data: the first CPU fine-tunes against the GPU-trained models, all three domains."""
    plt = mpl_dark()
    z = pd.read_csv(R / "zeroshot.csv").drop_duplicates("tag", keep="last").set_index("tag")
    sea = pd.read_csv(R / "seadronessee.csv").drop_duplicates(["tag", "split"], keep="last").set_index("tag")
    hit = pd.read_csv(R / "hituav_shared.csv").drop_duplicates("tag", keep="last").set_index("tag")
    groups = ["Land, by day\n(VisDrone)", "At sea\n(SeaDronesSee)", "Night, thermal\n(HIT-UAV)"]
    before = [z.loc["ft_y11n_1280", "AP"], sea.loc["ft_y11n_1280", "AP"], hit.loc["ft_y11n_640_test", "AP"]]
    after = [z.loc["gpu_y11n_1280", "AP"], sea.loc["gpu_y11n_1280", "AP"], hit.loc["gpu_y11n_640_test", "AP"]]
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    xs, w = range(3), 0.38
    for off, vals, col, lab in ((-w / 2, before, "#5c6670", "3-5 CPU epochs"), (w / 2, after, "#ffc400",
                                "60-120 GPU epochs")):  # fmt: skip
        bars = ax.bar([x + off for x in xs], vals, w * 0.94, color=col, label=lab)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=18)
    ax.set_xticks(list(xs), groups)
    ax.set_ylabel("COCO AP on held-out images")
    ax.set_ylim(0, max(after) * 1.25)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    tmp = Path(tempfile.gettempdir()) / "esa-results.png"
    fig.savefig(tmp, facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    hold = pd.read_csv(R / "visdrone_holdout.csv").drop_duplicates(["tag", "sequence"], keep="last")
    h = {t: hold[(hold.tag == t) & (hold.sequence == "all")].iloc[0] for t in ("ft_y11n_ep3", "gpu_y11n_640")}
    al = {f: pd.read_csv(R / f).query("clip == 'all' and dedupe_radius_m == 3").iloc[0]
          for f in ("alert_load.csv", "alert_load_gpu.csv")}  # fmt: skip
    env = pd.read_csv(R / "land_detection_by_altitude_hit_y11n_640_gpu.csv").set_index("alt_m").loc["<=115"]
    im, d = canvas()
    d.text((60, 40), "Detection, measured on data the models never saw", font=F_H2, fill=FG)
    ch = Image.open(tmp).convert("RGB")
    ch.thumbnail((1150, 800))
    im.paste(ch, (40, 140))
    text_block(d, (1240, 180), [
        f"Land, held-out video: {h['ft_y11n_ep3'].recall:.0%} → {h['gpu_y11n_640'].recall:.0%} of objects found",
        f"Sea, real clips: {al['alert_load.csv'].unique_swimmers_alerted} → "
        f"{al['alert_load_gpu.csv'].unique_swimmers_alerted} of {al['alert_load_gpu.csv'].labelled_swimmers} "
        "swimmers raise an alert",
        f"Night, ≤ 115 m: {float(env.recall):.0%} of people found, {float(env.false_alarms_per_image):.2f} false "
        "alarms per image",
        "",
        "Weak spots stay in the README: life rings, people mislabelled as cars at 130 m at night",
    ], fnt=F_SMALL, gap=50, width=640)  # fmt: skip
    footer(d, "Same test sets and scripts before and after · test images never uploaded for training · "
              "every number checked in CI")  # fmt: skip
    return im


def scene_tak(alert: dict):
    xml = alert_event(alert)
    pretty = [p for ln in re.sub(r"(<[^/][^>]*>)", r"\n\1", xml).strip().splitlines() for p in
              ([ln[i:i + 118] for i in range(0, len(ln), 118)] or [""])]  # fmt: skip
    im, d = canvas()
    d.text((60, 40), "To the field team: Cursor on Target for ATAK / WinTAK / TAK Server", font=F_H2, fill=FG)
    d.rounded_rectangle([60, 150, 1860, 760], 14, fill=(24, 28, 35))
    y = 175
    for ln in pretty[:13]:
        d.text((90, y), ln, font=F_TINY, fill=FG)
        y += 44
    text_block(d, (60, 800), [
        "One marker per person, even when both cameras see them; 'unknown', never 'hostile'.",
        "Confirm updates the marker; reject deletes it. Tested against a real TAK server in CI.",
    ], fnt=F_SMALL, gap=56)  # fmt: skip
    footer(d, "src/esa/cot.py · tests/tak_e2e.py · TLS to TAK is the next step (docs/DEPLOYMENT.md)")
    return im


def scene_end(repo_url):
    im, d = canvas()
    d.text((120, 330), "Open source, measured, and honest about what is not done yet", font=F_H2, fill=FG)
    d.rectangle([120, 420, 320, 426], fill=ACCENT)
    d.text((120, 470), repo_url, font=F_BODY, fill=ACCENT)
    d.text((120, 540), "Code, data splits, results files, CI, deployment and field-test plan", font=F_SMALL, fill=MUTED)
    footer(d, HONEST)
    return im


def scene_engineering(n_tests: int, n_numbers: int):
    im, d = canvas()
    d.text((120, 120), "ENGINEERING", font=F_SMALL, fill=ACCENT)
    d.text((120, 170), "Checked on every change, before it reaches main", font=F_H2, fill=FG)
    text_block(d, (120, 300), [
        f"{n_tests} unit and server tests; nothing merges without them passing",
        "The operator app in a real browser: login, no script injection, no policy violations",
        "The ROS 2 nodes on real images, four camera and telemetry modes, and RGB + thermal end to end",
        "Both search missions flown take-off to landing in ArduPilot's simulator",
        "Alerts delivered to a real TAK server",
        f"{n_numbers} numbers in the README checked against their results files",
        "Model files checksummed; CI actions pinned and read-only; a security policy",
    ])  # fmt: skip
    footer(
        d, "GitHub Actions: CI, ROS 2, SITL flight · branches and pull requests only · security audit fixes S-1 S-2 S-9"
    )
    return im


# ---------------------------------------------------------------- assembly
def run(cmd):
    subprocess.run(cmd, check=True)


def clip_segment(src: Path, out: Path, title: str, note: str, max_s: float | None = None):
    """A recorded clip centred on the dark background with a title above and a note below."""
    im, d = canvas()
    d.text((60, 40), title, font=F_H2, fill=FG)
    footer(d, note)
    frame = Path(tempfile.gettempdir()) / "esa-demo-frame.png"
    im.save(frame)
    dur = ["-t", str(max_s)] if max_s else []
    run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(frame), "-i", str(src), *dur,
         "-filter_complex", "[1:v]scale=1720:860:force_original_aspect_ratio=decrease,fps=30[v];"
         "[0:v][v]overlay=(W-w)/2:140:shortest=1,format=yuv420p",
         "-r", str(FPS), "-c:v", "libx264", "-crf", "20", "-an", str(out)])  # fmt: skip


def held(frames, last_s: float):
    """Each frame for one video frame; the last one held for `last_s` seconds. Streams: one image in memory."""
    prev = None
    for f in frames:
        if prev is not None:
            yield prev, 1 / FPS
        prev = f
    if prev is not None:
        yield prev, last_s


def stills_segment(frames, out: Path, work: Path):
    """(image, seconds) pairs, from a list or a generator; each image is written to disk as soon as it arrives."""
    lst = work / f"{out.stem}.txt"
    with lst.open("w") as fh:
        for i, (img, secs) in enumerate(frames):
            p = work / f"{out.stem}_{i:05d}.png"
            img.save(p)
            fh.write(f"file '{p}'\nduration {secs}\n")
        fh.write(f"file '{p}'\n")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf",
         f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-crf", "20", str(out)])  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitl-track", required=True)
    ap.add_argument("--sitl-report", required=True)
    ap.add_argument("--plan", default=str(REPO / "samples/search_example.plan"))
    ap.add_argument("--land-track", required=True)
    ap.add_argument("--land-report", required=True)
    ap.add_argument("--land-plan", default=str(REPO / "samples/land_search_example.plan"))
    ap.add_argument("--refusal", required=True)
    ap.add_argument("--review-clip", required=True)
    ap.add_argument("--dual-clip", required=True)
    ap.add_argument("--sea-clip", required=True)
    ap.add_argument("--sea-run", required=True)
    ap.add_argument("--sea-tag", default="sea_DJI_0055_bytetrack_gpu")
    ap.add_argument("--thermal-dir", required=True)
    ap.add_argument("--land-dir", required=True, help="held-out VisDrone frames (datasets/demo_holdout)")
    ap.add_argument("--land-model", required=True, help="VisDrone GPU model (not distributed: no dataset licence)")
    ap.add_argument("--repo-url", default="github.com/mustafarkhan/edge-situational-awareness")
    ap.add_argument("--short", action="store_true", help="the 75-second cut")
    ap.add_argument("--out", default=str(REPO / "outputs/project_v3.mp4"))
    a = ap.parse_args()

    R = REPO / "benchmarks/results"
    work = Path(tempfile.gettempdir()) / "esa-demo"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    segs: list[Path] = []
    short = a.short

    def stills(name, frames):
        out = work / f"{len(segs):02d}_{name}.mp4"
        stills_segment(frames, out, work)
        segs.append(out)

    def clip(name, src, title, note, max_s=None):
        out = work / f"{len(segs):02d}_{name}.mp4"
        clip_segment(Path(src), out, title, note, max_s)
        segs.append(out)

    def load_track(path):
        return [{**r, **{k: float(r[k]) for k in ("t_s", "lat", "lon", "alt_m")}} for r in csv.DictReader(open(path))]

    # the question, and what the detectors measure on held-out data
    stills("intro", [(scene_title(a.repo_url), 4 if short else 5)] + ([] if short else [(scene_question(), 9)]))
    stills("results", [(scene_results(R), 9 if short else 11)])

    # the plan: sea by day, land at night
    alt = pd.read_csv(R / "search_altitude.csv")
    row = alt[(alt.size_m == 0.5) & (alt.fps == 4.0) & (alt.alt_m == 100)].iloc[0]
    row20 = alt[(alt.size_m == 0.5) & (alt.fps == 4.0) & (alt.alt_m == 20)].iloc[0]
    ex = json.loads((R / "search_example.json").read_text())["example"]
    lex = json.loads((R / "land_search_example.json").read_text())
    land = pd.read_csv(R / "land_search_altitude.csv")
    lbest = land[land.alt_m == lex["alt_m"]].sort_values("km2_h_at_pod80").iloc[-1]
    if not short:
        stills("plan", [
            (scene_figure("How high to fly, from measured detection", REPO / "docs/figures/search_altitude.png", [
                "At sea, by day (person's head and shoulders):",
                f"{row20.km2_h_at_pod80:.2f} km² per hour at 20 m",
                f"{row.km2_h_at_pod80:.2f} km² per hour at 100 m",
                f"{row.p_per_pass:.2f} chance of detection per pass",
                "",
                f"Example: {ex['area_km2']} km² at {ex['alt_m']} m,",
                f"{ex['tracks']} tracks {ex['track_spacing_m']:.0f} m apart, {ex['flight_min']:.0f} min, 80% POD",
            ], "Detection curve measured on 4,221 unseen swimmers (SeaDronesSee) · benchmarks/search_planning.py"), 9),
            (scene_figure("On land at night: the thermal lens sets the altitude", REPO / "docs/figures/land_search.png", [
                f"{lbest.hfov_deg:.0f}° lens at {lex['alt_m']} m:",
                f"{lbest.km2_h_at_pod80:.2f} km² per hour",
                f"{lbest.p_per_pass:.2f} chance of detection per pass",
                "",
                f"Example: {lex['area_km2']} km², {lex['tracks']} tracks,",
                f"{lex['track_spacing_m']:.0f} m apart, {lex['flight_min']:.0f} min",
                "",
                "Planned 5 m under the 120 m ceiling.",
            ], "HIT-UAV thermal (public) · GPU-trained thermal model · benchmarks/land_search.py"), 8),
        ])  # fmt: skip

    # fly them, safely
    report = json.loads(Path(a.sitl_report).read_text())
    stills("mission_sea", held(mission_frames(json.loads(Path(a.plan).read_text()), load_track(a.sitl_track), report,
                                              seconds=9.0 if short else 13.0,
                                              title="At sea: the whole sortie in ArduPilot's simulator"), 2.5))  # fmt: skip
    if not short:
        lrep = json.loads(Path(a.land_report).read_text())
        stills("mission_land", held(mission_frames(
            json.loads(Path(a.land_plan).read_text()), load_track(a.land_track), lrep, seconds=10.0,
            title="On land at night: the thermal search, flown",
            note="samples/land_search_example.plan · ArduCopter 4.7.2 SITL · flown in CI on every change"), 2.5))  # fmt: skip
        stills("safety", [(scene_safety(Path(a.refusal).read_text(), report["autopilot_settings"]), 9)])

    # detect: sea, night, land by day
    tr = pd.read_csv(R / "tracking_sea.csv").drop_duplicates("tag", keep="last")
    summ = tr[tr.tag == a.sea_tag].iloc[0].to_dict()
    run_ = [json.loads(x) for x in Path(a.sea_run).read_text().splitlines()]
    clip_dir = Path(a.sea_clip)
    telem = [json.loads(x) for x in (clip_dir / "telemetry.jsonl").read_text().splitlines()]
    fr = sorted((clip_dir / "frames").iterdir())
    step = 3 if short else 2
    sea = ((scene_sea_track(fp, rec, telem[rec["frame"]], summ, clip_dir.name, i, len(fr)), step / 30)
           for i, (rec, fp) in enumerate(zip(run_, fr)) if i % step == 0)  # fmt: skip
    stills("sea", sea)
    hit = pd.read_csv(R / "hituav_shared.csv").drop_duplicates("tag", keep="last").set_index("tag")
    imgs = [p for p in sorted(Path(a.thermal_dir, "images/test").glob("1_*"))
            if int(p.stem.split("_")[1]) <= MAX_PLAN_ALT_M]  # night, inside the flight envelope  # fmt: skip
    picks = [imgs[len(imgs) * k // 3 + 5] for k in range(3)]
    stills("thermal", [(scene_detect(
        "At night: the thermal model", picks, REPO / "models/hit_y11n_640_gpu.pt", 640,
        [p.stem.split("_")[1] + " m altitude" for p in picks],
        [f"Night AP {hit.loc['gpu_y11n_640_test_night', 'AP']:.3f}, against "
         f"{hit.loc['coco_y11n_640_test_night', 'AP']:.3f} for a colour-trained model",
         "People in red, vehicles in blue · night test images from held-out flights, at or below 115 m",
         "Above 120 m it labels many people as vehicles: a known weak spot (README §10b)"],
        "HIT-UAV (public, CC0) · flight-disjoint split · models/hit_y11n_640_gpu.pt (120 GPU epochs)"), 6 if short else 7)])  # fmt: skip
    limgs = sorted(Path(a.land_dir, "images").glob("*.jpg"))
    lpicks = [limgs[len(limgs) * k // 3 + 5] for k in range(3)]
    hold = pd.read_csv(R / "visdrone_holdout.csv").drop_duplicates(["tag", "sequence"], keep="last")
    hrow = hold[(hold.tag == "gpu_y11n_640") & (hold.sequence == "all")].iloc[0]
    stills("land", [(scene_detect(
        "On land by day: people and vehicles", lpicks, Path(a.land_model), 1280,
        [f"held-out sequence {p.stem.split('_')[0]}" for p in lpicks],
        [f"{hrow.recall:.0%} of {int(hrow.objects):,} labelled objects found in two sequences removed before training",
         "People in red, vehicles in blue"],
        "VisDrone (research use; model not redistributed) · GPU-trained YOLO11n at 1280 px · "
        "benchmarks/holdout_recall.py", conf=0.25), 6 if short else 7)])  # fmt: skip

    # the operator and the field team
    clip("review", a.review_clip, "The operator decides: every alert is confirmed or rejected",
         "Real recorded sea flight replayed through the ground station · operator logged in with a token · "
         "decisions in a tamper-evident log", max_s=12 if short else None)  # fmt: skip
    if not short:
        clip("dual", a.dual_clip, "RGB + thermal: one person, one alert",
             "Integration-test replay (synthetic positions): the same person 2 m apart on both cameras is merged; "
             "a second person stays separate")  # fmt: skip
    sample = {"track_id": 7, "uid": "rgb-7", "label": "person", "lat": 61.49, "lon": 23.735,
              "message": "person detected (track 7)", "cameras": ["rgb", "thermal"]}  # fmt: skip
    stills("tak", [(scene_tak(sample), 6 if short else 8)])

    # how it is built, and what comes next
    n_tests = int(re.search(r"(\d+) tests? collected", subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=REPO, capture_output=True, text=True).stdout).group(1))  # fmt: skip
    n_numbers = int(re.search(r"\((\d+) numbers checked\)", subprocess.run(
        [sys.executable, "benchmarks/check_readme.py"], cwd=REPO, capture_output=True, text=True).stdout).group(1))  # fmt: skip
    end = [(scene_end(a.repo_url), 5)]
    if not short:
        end = [
            (scene_engineering(n_tests, n_numbers), 9),
            (scene_card("Next", "What has not been done yet", [
                "Measure speed, power and INT8 accuracy on a Jetson Orin Nano",
                "Fix the night person/car confusion above 120 m; detect life rings and vests",
                "Field tests on a university drone, following the deployment plan",
                "Publish measured sweep widths for AI drone detection: the benchmark",
            ]), 8),
        ] + end  # fmt: skip
    stills("end", end)

    lst = work / "segments.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", a.out])
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.out],
                         capture_output=True, text=True).stdout.strip()  # fmt: skip
    print(f"video -> {a.out} ({float(dur):.1f} s)")


if __name__ == "__main__":
    main()
