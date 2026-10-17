"""The v3 demo video: the whole system, from the search plan to the operator, built only from real outputs.

  python docs/video/make_demo.py --sitl-track out/sitl_sea_track.csv --sitl-report out/sitl_sea.json \
      --refusal out/refusal.txt --review-clip out/review_v3.mp4 --dual-clip out/dual_v3.mp4 \
      --sea-clip outputs/sea_mot/DJI_0055 --sea-run outputs/sea_track_DJI_0055_bytetrack.jsonl \
      --thermal-dir datasets/hituav_yolo --out outputs/demo_v3.mp4

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
)
from PIL import Image  # noqa: E402
from sea_scenes import scene_sea_track  # noqa: E402

from esa.cot import alert_event  # noqa: E402

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
    d.text((120, 570), "Land, sea and night · RGB + thermal · PX4 / ArduPilot · built for NVIDIA Jetson",
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


def mission_frames(plan, track, report, seconds=14.0):
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
        d.text((60, 40), "Plan, upload, fly: the whole sortie in ArduPilot's simulator", font=F_H2, fill=FG)
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
        footer(
            d,
            "samples/search_example.plan · ArduCopter 4.7.2 SITL · tools/fly_mission.py · flown in CI on every change",
        )
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


def scene_thermal(images: list[Path], model_path: Path, ap_note: str):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    im, d = canvas()
    d.text((60, 40), "At night: the thermal model", font=F_H2, fill=FG)
    x = 60
    for p in images:
        src = Image.open(p).convert("RGB")
        r = model.predict(str(p), imgsz=640, conf=0.3, verbose=False)[0]
        tile, s = fit(src, 580, 470)
        from PIL import ImageDraw

        td = ImageDraw.Draw(tile)
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()):
            col = ALERT if r.names[c] == "person" else BLUE
            td.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline=col, width=3)
        im.paste(tile, (x, 170))
        d.text((x, 660), p.stem.split("_")[1] + " m altitude", font=F_TINY, fill=MUTED)
        x += 610
    text_block(d, (60, 740), [ap_note, "People in red, vehicles in blue · night test images from held-out flights"],
               fnt=F_SMALL, gap=56)  # fmt: skip
    footer(d, "HIT-UAV (public, CC0) · flight-disjoint split · models/hit_y11n_ep5.pt (5 CPU epochs)")
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


def stills_segment(frames: list[tuple[Image.Image, float]], out: Path, work: Path):
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
    ap.add_argument("--refusal", required=True)
    ap.add_argument("--review-clip", required=True)
    ap.add_argument("--dual-clip", required=True)
    ap.add_argument("--sea-clip", required=True)
    ap.add_argument("--sea-run", required=True)
    ap.add_argument("--sea-tag", default="sea_DJI_0055_bytetrack")
    ap.add_argument("--thermal-dir", required=True)
    ap.add_argument("--repo-url", default="github.com/mustafarkhan/edge-situational-awareness")
    ap.add_argument("--out", default=str(REPO / "outputs/demo_v3.mp4"))
    a = ap.parse_args()

    R = REPO / "benchmarks/results"
    work = Path(tempfile.gettempdir()) / "esa-demo"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    segs: list[Path] = []

    def stills(name, frames):
        out = work / f"{len(segs):02d}_{name}.mp4"
        stills_segment(frames, out, work)
        segs.append(out)

    def clip(name, src, title, note, max_s=None):
        out = work / f"{len(segs):02d}_{name}.mp4"
        clip_segment(Path(src), out, title, note, max_s)
        segs.append(out)

    # 1-3: the question and the plan
    alt = pd.read_csv(R / "search_altitude.csv")
    row = alt[(alt.size_m == 0.5) & (alt.fps == 4.0) & (alt.alt_m == 100)].iloc[0]
    row20 = alt[(alt.size_m == 0.5) & (alt.fps == 4.0) & (alt.alt_m == 20)].iloc[0]
    ex = json.loads((R / "search_example.json").read_text())["example"]
    land = pd.read_csv(R / "land_search_altitude.csv")
    l45 = land[(land.hfov_deg == 45.0) & (land.alt_m == 60)].iloc[0]
    stills("intro", [(scene_title(a.repo_url), 5), (scene_question(), 9)])
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
            f"45° lens at 60 m: {l45.km2_h_at_pod80:.2f} km² per hour",
            f"{l45.p_per_pass:.2f} chance of detection per pass",
            "",
            "Planned 5 m under the 120 m ceiling.",
        ], "HIT-UAV thermal (public) · benchmarks/land_search.py"), 7),
    ])  # fmt: skip

    # 4-5: fly it, safely
    plan = json.loads(Path(a.plan).read_text())
    track = [
        {**r, **{k: float(r[k]) for k in ("t_s", "lat", "lon", "alt_m")}} for r in csv.DictReader(open(a.sitl_track))
    ]
    report = json.loads(Path(a.sitl_report).read_text())
    frames = [(f, 1 / FPS) for f in mission_frames(plan, track, report)]
    frames[-1] = (frames[-1][0], 3.0)  # hold the result
    stills("mission", frames)
    stills("safety", [(scene_safety(Path(a.refusal).read_text(), report["autopilot_settings"]), 9)])

    # 6-7: detect
    tr = pd.read_csv(R / "tracking_sea.csv").drop_duplicates("tag", keep="last")
    summ = tr[tr.tag == a.sea_tag].iloc[0].to_dict()
    run_ = [json.loads(x) for x in Path(a.sea_run).read_text().splitlines()]
    clip_dir = Path(a.sea_clip)
    telem = [json.loads(x) for x in (clip_dir / "telemetry.jsonl").read_text().splitlines()]
    fr = sorted((clip_dir / "frames").iterdir())
    sea = [(scene_sea_track(fp, rec, telem[rec["frame"]], summ, clip_dir.name, i, len(fr)), 1 / 15)
           for i, (rec, fp) in enumerate(zip(run_, fr)) if i % 2 == 0]  # fmt: skip
    stills("sea", sea)
    hit = pd.read_csv(R / "hituav_shared.csv").drop_duplicates("tag", keep="last")
    night = hit[hit.tag == "ft_y11n_640_test_night"].iloc[0]
    coco_night = hit[hit.tag == "coco_y11n_640_test_night"].iloc[0]
    imgs = sorted(Path(a.thermal_dir, "images/test").glob("1_*"))
    picks = [imgs[len(imgs) * k // 3 + 5] for k in range(3)]
    stills("thermal", [(scene_thermal(picks, REPO / "models/hit_y11n_ep5.pt",
                                      f"Night AP {night.AP:.3f}, against {coco_night.AP:.3f} for a colour-trained model"),
                        7)])  # fmt: skip

    # 8-10: the operator and the field team
    clip("review", a.review_clip, "The operator decides: every alert is confirmed or rejected",
         "Real recorded sea flight replayed through the ground station · operator logged in with a token · "
         "decisions in a tamper-evident log")  # fmt: skip
    clip("dual", a.dual_clip, "RGB + thermal: one person, one alert",
         "Integration-test replay (synthetic positions): the same person 2 m apart on both cameras is merged; "
         "a second person stays separate")  # fmt: skip
    sample = {"track_id": 7, "uid": "rgb-7", "label": "person", "lat": 61.49, "lon": 23.735,
              "message": "person detected (track 7)", "cameras": ["rgb", "thermal"]}  # fmt: skip
    stills("tak", [(scene_tak(sample), 8)])

    # 11-13: how it is built, and what comes next
    n_tests = int(re.search(r"(\d+) tests? collected", subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=REPO, capture_output=True, text=True).stdout).group(1))  # fmt: skip
    n_numbers = int(re.search(r"\((\d+) numbers checked\)", subprocess.run(
        [sys.executable, "benchmarks/check_readme.py"], cwd=REPO, capture_output=True, text=True).stdout).group(1))  # fmt: skip
    stills("end", [
        (scene_engineering(n_tests, n_numbers), 9),
        (scene_card("Next", "What has not been done yet", [
            "Score the GPU-trained models (trained overnight; not yet evaluated)",
            "Measure speed, power and INT8 accuracy on a Jetson Orin Nano",
            "Field tests on a university drone, following the deployment plan",
            "Publish measured sweep widths for AI drone detection: the benchmark",
        ]), 8),
        (scene_end(a.repo_url), 5),
    ])  # fmt: skip

    lst = work / "segments.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", a.out])
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.out],
                         capture_output=True, text=True).stdout.strip()  # fmt: skip
    print(f"demo -> {a.out} ({float(dur):.1f} s)")


if __name__ == "__main__":
    main()
