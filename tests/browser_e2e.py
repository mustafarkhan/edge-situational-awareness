"""Browser end-to-end test of the operator map (built app + ground-station server, headless Chromium).

Checks, against the real built app:
  1. with the operator link (#token=...) the map goes live and alerts arrive;
  2. a detection whose label carries script is shown as text when clicked, and the script never runs;
  3. a review is accepted and attributed to the operator the token belongs to;
  4. without a token, or with a wrong one, the app receives nothing;
  5. the page reports no Content-Security-Policy violations.

  cd web/frontend && npx vite build && cd ../.. && python tests/browser_e2e.py
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from esa.access import AuditLog, new_token

REPO = Path(__file__).resolve().parents[1]
LAT, LON = 61.4900, 23.7350
PAYLOAD = '<img src=x onerror="window.__xss=1">'


def wait_port(port: int, timeout: float = 20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"port {port}")


def replay_file(path: Path):
    det = {"track_id": 7, "label": PAYLOAD, "conf": 0.9, "lat": LAT, "lon": LON}
    alert = {**det, "message": f"{PAYLOAD} detected (track 7)"}
    with path.open("w") as fh:
        for i in range(400):
            rec = {"frame": i, "t": i * 0.1, "fps": 10.0, "drone": {"lat": LAT, "lon": LON}, "detections": [det],
                   "alerts": [alert] if i == 2 else []}  # fmt: skip
            fh.write(json.dumps(rec) + "\n")


def main() -> int:
    from playwright.sync_api import sync_playwright

    dist = REPO / "web/frontend/dist"
    if not (dist / "index.html").exists():
        sys.exit("build the app first: cd web/frontend && npx vite build")
    work = Path(tempfile.mkdtemp())
    replay, reviews, token = work / "xss.jsonl", work / "reviews.jsonl", new_token()
    replay_file(replay)
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(REPO / "web/backend/server.py"),
                "--replay",
                str(replay),
                "--rate",
                "10",
                "--port",
                "8000",
                "--reviews",
                str(reviews),
                "--token",
                token,
            ]
        ),  # fmt: skip
        subprocess.Popen(
            [sys.executable, "-m", "http.server", "5173", "--bind", "127.0.0.1", "--directory", str(dist)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),  # fmt: skip
    ]
    checks: dict[str, bool] = {}
    try:
        wait_port(8000)
        wait_port(5173)
        base = f"http://localhost:5173/?offline=1&center={LAT},{LON}&zoom=19"
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            )

            # 4. no token, wrong token
            page = browser.new_page()
            page.goto(base)
            page.wait_for_timeout(1500)
            checks["no token: app asks for the operator link"] = "no operator token" in page.inner_text(".status")
            page.close()
            page = browser.new_page()  # a fresh page: changing only the #fragment would not reload the app
            page.goto(base + "#token=" + "x" * 32)
            page.wait_for_selector("text=not authorised", timeout=10_000)
            checks["wrong token: not authorised, no alerts"] = page.locator("button.reject").count() == 0
            page.close()

            # 1-3, 5. operator link
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            csp_errors = []
            page.on("console", lambda m: csp_errors.append(m.text) if "Content Security Policy" in m.text else None)
            page.goto(base + "#token=" + token)
            page.wait_for_selector("text=live", timeout=15_000)
            checks["token is removed from the address bar"] = "token" not in page.url
            page.wait_for_selector("button.reject", timeout=20_000)
            checks["alert arrives, label shown as text"] = PAYLOAD in page.inner_text("body")

            canvas = page.locator("canvas.maplibregl-canvas")
            box = canvas.bounding_box()
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)  # the detection is centred
            page.wait_for_selector(".maplibregl-popup-content", timeout=10_000)
            popup = page.inner_text(".maplibregl-popup-content")
            checks["popup shows the label as text"] = PAYLOAD in popup
            page.wait_for_timeout(1000)
            checks["script in the label never ran"] = page.evaluate("window.__xss === undefined")
            checks["no <img> injected into the popup"] = page.locator(".maplibregl-popup-content img").count() == 0

            page.locator("button.reject").first.click()
            page.wait_for_selector("li.reject", timeout=10_000)
            checks["no Content-Security-Policy violations"] = not csp_errors
            browser.close()
        time.sleep(0.5)
        recs = [json.loads(line) for line in reviews.read_text().splitlines()] if reviews.exists() else []
        checks["review logged for the token's operator"] = len(recs) == 1 and recs[0]["operator"] == "operator"
        checks["audit log chain intact"] = AuditLog.verify(reviews)[0] if recs else False
    finally:
        for pr in procs:
            pr.terminate()
    for name, ok in checks.items():
        print(f"{'OK  ' if ok else 'FAIL'} {name}")
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
