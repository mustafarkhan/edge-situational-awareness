"""Ground-station bridge: detections -> browser map, operator reviews, and TAK (ATAK / WinTAK / TAK Server).

Sources (one of):
  --replay outputs/run1.jsonl   replay a file from tools/detect_video.py (laptop, no ROS)
  --ros                          subscribe to /tracks_geo from the ROS 2 detector node

Access (esa.access): every WebSocket client first sends {"type": "auth", "token": "..."}; nothing is sent to it
before that. Browsers may only connect from allowed origins (--allow-origin). The server listens on this machine
only (127.0.0.1) unless --host says otherwise, and a non-local --host needs operator tokens given explicitly and,
for anything but a lab bench, TLS (--ssl-certfile/--ssl-keyfile, then the app connects with wss://).

Operator loop: the map app sends {"type": "review", "track_id": .., "decision": "confirm" | "reject"}.
The decision is attributed to the operator the token belongs to, appended to a hash-chained audit log (--reviews),
broadcast to every connected operator, forwarded to TAK (confirm: marker updated to "CONFIRMED"; reject: marker
deleted) and, with --ros, published on /alert_reviews so the detector suppresses new alerts at that spot
(esa.alerts.AlertManager.reject).

  python web/backend/server.py --replay web/sample/sea_DJI_0055.jsonl --rate 15 --tak udp://239.2.3.1:6969
It prints an operator link (http://localhost:5173/#token=...) for the React app in web/frontend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from esa.access import (  # noqa: E402
    DEFAULT_ORIGINS,
    AuditLog,
    OperatorDirectory,
    RateLimiter,
    is_loopback,
    new_token,
    origin_allowed,
    parse_review,
)
from esa.cot import CotSender, alert_event, delete_event, drone_event  # noqa: E402

AUTH_TIMEOUT_S = 5.0
MAX_ALERTS = 5000  # remembered alerts (for reviews); the oldest are forgotten first
MAX_MESSAGE_BYTES = 64 * 1024

clients: set[WebSocket] = set()  # authenticated operators only
queue: asyncio.Queue | None = None  # created on startup
ARGS = argparse.Namespace(ros=False, replay=None, rate=10.0, once=False, tak=None, drone_uid="esa-drone-1",
                          reviews="outputs/reviews.jsonl", ce=15.0, allow_origin=list(DEFAULT_ORIGINS))  # fmt: skip
OPERATORS = OperatorDirectory()  # set in main(); empty = nobody can connect
STATE = {"alerts": OrderedDict(), "reviews": {}, "tak": None, "tak_sent": 0, "last_drone_cot": 0.0, "ros_pub": None,
         "audit": None}  # fmt: skip


@asynccontextmanager
async def lifespan(_app):
    global queue
    queue = asyncio.Queue(maxsize=100)
    if ARGS.tak:
        STATE["tak"] = CotSender(ARGS.tak)
    if ARGS.reviews:
        STATE["audit"] = AuditLog(ARGS.reviews)
    asyncio.create_task(broadcaster())
    if ARGS.ros:
        threading.Thread(target=ros_thread, args=(asyncio.get_running_loop(),), daemon=True).start()
    elif ARGS.replay:
        asyncio.create_task(replay(ARGS.replay, ARGS.rate))
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    if not origin_allowed(websocket.headers.get("origin"), ARGS.allow_origin):
        await websocket.close(code=1008, reason="origin not allowed")  # before accept: the handshake is refused
        return
    await websocket.accept()
    operator = await authenticate(websocket)
    if operator is None:
        await websocket.close(code=1008, reason="authentication required")
        return
    await websocket.send_text(json.dumps({"type": "hello", "operator": operator}))
    for r in STATE["reviews"].values():  # a newly connected operator sees decisions already made
        await websocket.send_text(json.dumps(r))
    clients.add(websocket)
    limiter = RateLimiter()
    try:
        while True:
            text = await websocket.receive_text()
            if not limiter.allow(time.monotonic()):
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            review = parse_review(msg, STATE["alerts"])
            if review is not None:
                await handle_review(*review, operator=operator)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


async def authenticate(websocket: WebSocket) -> str | None:
    """The first message must be {"type": "auth", "token": ...}, within AUTH_TIMEOUT_S."""
    try:
        msg = json.loads(await asyncio.wait_for(websocket.receive_text(), AUTH_TIMEOUT_S))
    except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect, KeyError):
        return None
    if not isinstance(msg, dict) or msg.get("type") != "auth":
        return None
    return OPERATORS.authenticate(msg.get("token"))


async def handle_review(tid: int, decision: str, operator: str) -> dict | None:
    alert = STATE["alerts"].get(tid)
    if alert is None:
        return None
    review = {
        "type": "review",
        "track_id": tid,
        "decision": decision,
        "label": alert["label"],
        "lat": alert.get("lat"),
        "lon": alert.get("lon"),
        "operator": operator,  # from the token, never from the client's message
        "time": time.time(),
    }
    STATE["reviews"][tid] = review
    if STATE["audit"] is not None:
        STATE["audit"].append(review)
    tak_send_review(alert, decision)
    if STATE["ros_pub"] is not None:
        STATE["ros_pub"](json.dumps(review))
    await broadcast(json.dumps(review))
    return review


def tak_send(xml: str):
    if STATE["tak"] is None:
        return
    try:
        STATE["tak"].send(xml)
        STATE["tak_sent"] += 1
    except OSError as e:  # never let TAK problems stop the map
        print(f"TAK send failed: {e}", file=sys.stderr)


def tak_send_review(alert: dict, decision: str):
    if alert.get("lat") is None:
        return
    uid = f"{ARGS.drone_uid}-{alert['label']}-{alert['track_id']}"
    if decision == "confirm":
        tak_send(alert_event(alert, ARGS.drone_uid, ARGS.ce, confirmed=True, stale_s=3600))
    else:
        tak_send(delete_event(uid, alert["lat"], alert["lon"]))


def on_frame(text: str):
    """Side effects of one detection frame: remember alerts, forward alerts and the drone to TAK."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return
    if msg.get("frame") == 0:  # a replay restarted: forget the previous pass
        STATE["alerts"].clear()
        STATE["reviews"].clear()
    for a in msg.get("alerts") or []:
        STATE["alerts"][a["track_id"]] = a
        STATE["alerts"].move_to_end(a["track_id"])
        while len(STATE["alerts"]) > MAX_ALERTS:
            STATE["alerts"].popitem(last=False)
        if a.get("lat") is not None:
            tak_send(alert_event(a, ARGS.drone_uid, ARGS.ce))
    d = msg.get("drone")
    now = time.monotonic()
    if d and now - STATE["last_drone_cot"] >= 1.0:  # drone position to TAK at 1 Hz
        STATE["last_drone_cot"] = now
        tak_send(drone_event(ARGS.drone_uid, d["lat"], d["lon"], d.get("alt_m")))


@app.get("/health")
def health():
    return {
        "clients": len(clients),
        "source": "ros" if ARGS.ros else ARGS.replay,
        "tak": bool(ARGS.tak),
        "tak_messages_sent": STATE["tak_sent"],
        "alerts": len(STATE["alerts"]),
        "reviews": len(STATE["reviews"]),
    }


async def broadcast(msg: str):
    for c in list(clients):
        try:
            await c.send_text(msg)
        except Exception:
            clients.discard(c)


async def broadcaster():
    while True:
        msg = await queue.get()
        on_frame(msg)
        await broadcast(msg)


async def replay(path: str, rate: float):
    lines = Path(path).read_text().splitlines()
    while True:  # loop forever so the demo keeps running
        while not clients:  # start each pass only once someone is watching, so viewers see frame 0
            await asyncio.sleep(0.1)
        for line in lines:
            await queue.put(line)
            await asyncio.sleep(1.0 / rate)
        if ARGS.once:  # single pass (used for recordings): leave the last frame on screen
            return


def _offer(msg: str):
    """Drop frames if the browser can't keep up: stale positions are worse than gaps."""
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        pass


def ros_thread(loop):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class Bridge(Node):
        def __init__(self):
            super().__init__("web_bridge")
            self.create_subscription(String, "tracks_geo", lambda m: loop.call_soon_threadsafe(_offer, m.data), 10)
            pub = self.create_publisher(String, "alert_reviews", 10)
            STATE["ros_pub"] = lambda text: pub.publish(String(data=text))

    rclpy.init()
    rclpy.spin(Bridge())


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay")
    src.add_argument("--ros", action="store_true")
    ap.add_argument("--rate", type=float, default=10.0, help="replay frames per second")
    ap.add_argument("--once", action="store_true", help="replay a single pass instead of looping")
    ap.add_argument("--tak", default=None, help="udp://239.2.3.1:6969 (ATAK mesh) or tcp://takserver:8087")
    ap.add_argument("--drone-uid", default="esa-drone-1")
    ap.add_argument("--ce", type=float, default=15.0, help="position uncertainty radius sent to TAK (m)")
    ap.add_argument("--reviews", default="outputs/reviews.jsonl", help="operator decisions audit log (hash-chained)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on; this machine only by default")
    ap.add_argument("--port", type=int, default=8000)
    who = ap.add_mutually_exclusive_group()
    who.add_argument("--token", help="one shared operator token (or set ESA_TOKEN); default: a new random one")
    who.add_argument("--operators", help="file of '<name> <token>' lines, one per operator")
    ap.add_argument(
        "--allow-origin",
        action="append",
        default=None,
        help=f"browser origin allowed to connect (repeatable); default {', '.join(DEFAULT_ORIGINS)}",
    )
    ap.add_argument("--ssl-certfile", help="TLS certificate: serve wss:// instead of ws://")
    ap.add_argument("--ssl-keyfile", help="TLS private key")
    a = ap.parse_args(argv)
    a.allow_origin = a.allow_origin or list(DEFAULT_ORIGINS)
    return a


def operators_from_args(a) -> tuple[OperatorDirectory, str | None]:
    """The operator directory, and the token if one was generated here (so it can be shown once)."""
    if a.operators:
        return OperatorDirectory.from_file(a.operators), None
    token = a.token or os.environ.get("ESA_TOKEN")
    if token:
        return OperatorDirectory.single(token), None
    token = new_token()
    return OperatorDirectory.single(token), token


def main(argv=None):
    global ARGS, OPERATORS
    ARGS = parse_args(argv)
    OPERATORS, generated = operators_from_args(ARGS)
    if not is_loopback(ARGS.host):
        if generated:
            sys.exit("--host is not this machine: give operator tokens explicitly (--operators or --token/ESA_TOKEN)")
        if not ARGS.ssl_certfile:
            print("WARNING: listening on the network without TLS: positions and tokens travel in cleartext",
                  file=sys.stderr)  # fmt: skip
    if generated:
        print(f"operator link: http://localhost:5173/#token={generated}", file=sys.stderr, flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, ws_max_size=MAX_MESSAGE_BYTES,
                ssl_certfile=ARGS.ssl_certfile, ssl_keyfile=ARGS.ssl_keyfile)  # fmt: skip


if __name__ == "__main__":
    main()
