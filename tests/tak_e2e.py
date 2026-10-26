"""End-to-end TAK test: ground-station bridge -> real TAK server (taky) -> a TAK client.

1. start taky, an open-source TAK server, on 127.0.0.1:8087 (plain TCP)
2. connect a listening TAK client to it (as ATAK would)
3. start web/backend/server.py replaying the recorded real flight, forwarding to tcp://127.0.0.1:8087
4. connect to the map WebSocket as an operator; wait for an alert; reject it
5. check the client received: the drone's position, the swimmer alert, and the delete for the rejected one;
   and that the rejection is in the audit log

python tests/tak_e2e.py --taky /path/to/venv/bin/taky       (CI: .github/workflows/ci.yml, job "tak")
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOKEN = secrets.token_urlsafe(24)


def wait_port(port: int, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.3)
    raise TimeoutError(f"port {port} never opened")


def tak_client(received: list[str], stop: threading.Event):
    s = socket.create_connection(("127.0.0.1", 8087), timeout=1)
    # announce ourselves like a TAK client does (a position report), so the server routes traffic to us
    sys.path.insert(0, str(REPO / "src"))
    from esa.cot import event

    s.sendall(event("test-client", "a-f-G-U-C", 61.5, 23.7, callsign="test-client", how="h-e").encode())
    buf = ""
    while not stop.is_set():
        try:
            chunk = s.recv(65536)
        except TimeoutError:
            continue
        if not chunk:
            break
        buf += chunk.decode(errors="replace")
        while "</event>" in buf:
            ev, buf = buf.split("</event>", 1)
            received.append(ev[ev.find("<event") :] + "</event>")
    s.close()


async def operator(port: int, log: list[dict]):
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        await ws.send(json.dumps({"type": "auth", "token": TOKEN}))
        hello = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert hello == {"type": "hello", "operator": "operator"}, hello
        t0 = time.time()
        while time.time() - t0 < 60:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("alerts"):
                a = msg["alerts"][0]
                await ws.send(json.dumps({"type": "review", "uid": a["uid"], "decision": "reject"}))
                log.append(a)
            if msg.get("type") == "review":
                log.append(msg)
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taky", required=True, help="path to the taky executable")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp())
    taky = subprocess.Popen([a.taky, "-l", "info"], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    server = None
    try:
        wait_port(8087)
        received, stop = [], threading.Event()
        th = threading.Thread(target=tak_client, args=(received, stop), daemon=True)
        th.start()
        time.sleep(1)
        reviews = work / "reviews.jsonl"
        server = subprocess.Popen(
            [sys.executable, str(REPO / "web/backend/server.py"), "--replay", str(REPO / "web/sample/sea_DJI_0055.jsonl"),
             "--rate", "30", "--once", "--port", "8765", "--tak", "tcp://127.0.0.1:8087", "--reviews", str(reviews),
             "--token", TOKEN],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )  # fmt: skip
        wait_port(8765)
        log: list[dict] = []
        asyncio.run(operator(8765, log))
        alert = log[0]
        uid = f"esa-drone-1-swimmer-{alert['uid']}"  # camera-scoped: esa-drone-1-swimmer-rgb-<track>
        # TAK messages go out from a background thread and through the TAK server: wait for the delete to arrive
        # (up to 20 s on a slow runner) instead of a fixed pause
        t0 = time.time()
        while time.time() - t0 < 20 and not any('type="t-x-d-d"' in e and f'uid="{uid}"' in e for e in list(received)):
            time.sleep(0.2)
        stop.set()

        types = [re.search(r'type="([^"]+)"', e).group(1) for e in received]
        checks = {
            "drone position reached the TAK client": "a-f-A-M-H-Q" in types,
            "swimmer alert reached the TAK client": any(f'uid="{uid}"' in e and 'type="a-u-S"' in e for e in received),
            "rejection deleted the marker": any('type="t-x-d-d"' in e and f'uid="{uid}"' in e for e in received),
            "rejection is in the audit log": reviews.exists() and '"reject"' in reviews.read_text(),
        }
        print(f"TAK client received {len(received)} events: {sorted(set(types))}")
        for k, v in checks.items():
            print(("OK   " if v else "FAIL ") + k)
            if not v and os.environ.get("GITHUB_ACTIONS"):  # an annotation: readable without the full job log
                print(f"::error title=TAK end to end::{k} failed; client received {len(received)} events: "
                      f"{sorted(set(types))}")  # fmt: skip
        sample = next((e for e in received if f'uid="{uid}"' in e), "")
        print("example alert as received by the client:\n" + sample)
        return 0 if all(checks.values()) else 1
    finally:
        for p in (server, taky):
            if p and p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # a timeout or crash before the checks: name it in an annotation too
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error title=TAK end to end::{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
        raise
