"""Screenshot the operator review loop in the real map app (for the README).

Replays the recorded flight, waits for alerts, rejects the first and confirms the second through the actual
buttons, then saves a screenshot. Needs the built frontend (cd web/frontend && npx vite build) and Playwright.

  python tools/screenshot_review.py --out docs/figures/operator_review.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=str(REPO / "web/sample/sea_DJI_0055.jsonl"))
    ap.add_argument("--out", default=str(REPO / "docs/figures/operator_review.png"))
    ap.add_argument("--center", default="47.672414,9.268977")
    ap.add_argument("--zoom", default="17.9")
    ap.add_argument("--video", default=None, help="also record the session to this .webm/.mp4 (for the demo video)")
    a = ap.parse_args()
    reviews = Path("/tmp/esa_reviews.jsonl")
    reviews.unlink(missing_ok=True)
    procs = [
        subprocess.Popen(
            [sys.executable, str(REPO / "web/backend/server.py"), "--replay", a.replay, "--rate", "15", "--once"]
            + ["--port", "8000", "--reviews", str(reviews)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        subprocess.Popen(
            [sys.executable, "-m", "http.server", "5173", "--directory", str(REPO / "web/frontend/dist")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
    ]
    time.sleep(3)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch()
            vp = {"width": 1500, "height": 860}
            vid_dir = Path(a.out).parent / "_rec"
            ctx = b.new_context(viewport=vp, record_video_dir=str(vid_dir), record_video_size=vp) if a.video else b
            page = ctx.new_page() if a.video else b.new_page(viewport=vp)
            page.goto(f"http://localhost:5173/?offline=1&center={a.center}&zoom={a.zoom}")
            page.wait_for_selector("button.reject", timeout=60_000)
            page.wait_for_timeout(1500)  # let a second alert arrive
            page.locator("button.reject").first.click()
            page.wait_for_selector("li.reject", timeout=10_000)
            page.locator("button.confirm").first.click()
            page.wait_for_selector("li.confirm", timeout=10_000)
            page.wait_for_timeout(2500)
            page.screenshot(path=a.out)
            if a.video:
                video = page.video
                ctx.close()
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", video.path(), "-r", "30", a.video], check=True
                )
                for f in vid_dir.iterdir():
                    f.unlink()
                vid_dir.rmdir()
            b.close()
        print("screenshot ->", a.out)
        print("audit log:", reviews.read_text())
    finally:
        for pr in procs:
            pr.terminate()


if __name__ == "__main__":
    main()
