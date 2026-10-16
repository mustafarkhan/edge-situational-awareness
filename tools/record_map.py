"""Screen-record the real web app replaying a run, for the demo video.

Starts the FastAPI WebSocket backend in replay mode, serves the built React app, opens it in
headless Chromium (Playwright) and records the page. Offline map style, so no tile server is needed.

  cd web/frontend && npm install && npx vite build && cd ../..
  python tools/record_map.py --replay outputs/track_slow_botsort.jsonl --seconds 14 --out outputs/map_recording.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--out", default=str(REPO / "outputs" / "map_recording.mp4"))
    ap.add_argument("--center", default=None, help="lat,lon to pin the map on (e.g. the flight's centre)")
    ap.add_argument("--zoom", type=float, default=None)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    dist = REPO / "web" / "frontend" / "dist"
    if not dist.exists():
        sys.exit("build the frontend first: cd web/frontend && npm install && npx vite build")
    backend = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "web/backend/server.py"),
            "--replay",
            args.replay,
            "--rate",
            str(args.rate),
            "--port",
            "8000",
            "--once",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    static = subprocess.Popen(
        [sys.executable, "-m", "http.server", "5173", "--directory", str(dist)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    vid_dir = Path(tempfile.mkdtemp(prefix="esa-rec-"))
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            )
            ctx = browser.new_context(
                viewport={"width": args.width, "height": args.height},
                color_scheme="dark",
                record_video_dir=str(vid_dir),
                record_video_size={"width": args.width, "height": args.height},
            )
            page = ctx.new_page()
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            # the backend waits for the first viewer before replaying, so the recording starts at frame 0
            url = "http://localhost:5173/?offline=1"
            if args.center:
                url += f"&center={args.center}"
            if args.zoom:
                url += f"&zoom={args.zoom}"
            page.goto(url)
            page.wait_for_timeout(int(args.seconds * 1000))
            page.screenshot(path=str(Path(args.out).with_suffix(".png")))
            ctx.close()
            browser.close()
            if errors:
                print("browser console errors:", errors[:5])
        webm = next(vid_dir.glob("*.webm"))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(webm),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                str(args.out),
            ],
            check=True,
        )
        print("recording ->", args.out)
    finally:
        backend.terminate()
        static.terminate()
        shutil.rmtree(vid_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
