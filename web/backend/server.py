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
import queue as queue_mod
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
from esa.fusion import AlertFusion, alert_uid  # noqa: E402

AUTH_TIMEOUT_S = 5.0
MAX_ALERTS = 5000  # remembered alerts (for reviews); the oldest are forgotten first
MAX_MESSAGE_BYTES = 64 * 1024
SEND_TIMEOUT_S = 2.0  # a browser that cannot take a message within this is dropped, not waited for

clients: set[WebSocket] = set()  # authenticated operators only
queue: asyncio.Queue | None = None  # created on startup
ARGS = argparse.Namespace(ros=False, replay=None, rate=10.0, once=False, tak=None, drone_uid="esa-drone-1",
                          reviews="outputs/reviews.jsonl", ce=15.0, allow_origin=list(DEFAULT_ORIGINS),
                          camera_topics=["tracks_geo"], fuse_radius=5.0, fuse_window=30.0)  # fmt: skip
OPERATORS = OperatorDirectory()  # set in main(); empty = nobody can connect
STATE = {"alerts": OrderedDict(), "reviews": {}, "tak": None, "tak_sent": 0, "tak_dropped": 0, "last_drone_cot": 0.0,
         "ros_pub": None, "audit": None, "fusion": AlertFusion()}  # fmt: skip


@asynccontextmanager
async def lifespan(_app):
    global queue
    queue = asyncio.Queue(maxsize=100)
    STATE["fusion"] = AlertFusion(ARGS.fuse_radius, ARGS.fuse_window, MAX_ALERTS)
    if ARGS.tak:
        STATE["tak"] = TakWorker(ARGS.tak)
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


async def handle_review(uid: str, decision: str, operator: str) -> dict | None:
    alert = STATE["alerts"].get(uid)
    if alert is None:
        return None
    review = {
        "type": "review",
        "uid": uid,
        "decision": decision,
        "label": alert["label"],
        "cameras": alert.get("cameras", [alert.get("camera")]),
        "lat": alert.get("lat"),
        "lon": alert.get("lon"),
        "operator": operator,  # from the token, never from the client's message
        "time": time.time(),
    }
    STATE["reviews"][uid] = review
    if STATE["audit"] is not None:
        STATE["audit"].append(review)
    tak_send_review(alert, decision)
    if STATE["ros_pub"] is not None:
        STATE["ros_pub"](json.dumps(review))
    await broadcast(json.dumps(review))
    return review


class TakWorker:
    """Sends Cursor-on-Target from its own thread, so a slow or unreachable TAK server never stalls the map.

    Messages wait in a bounded queue; when it is full the oldest are dropped (a stale position is worth less than
    a current one) and counted in /health.
    """

    def __init__(self, url: str, maxsize: int = 1000):
        self.url, self.q = url, queue_mod.Queue(maxsize=maxsize)
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, xml: str):
        while True:
            try:
                self.q.put_nowait(xml)
                return
            except queue_mod.Full:
                try:
                    self.q.get_nowait()
                    STATE["tak_dropped"] += 1
                except queue_mod.Empty:
                    pass

    def _run(self):
        sender = None
        while True:
            xml = self.q.get()
            try:
                if sender is None:
                    sender = CotSender(self.url)
                sender.send(xml)
                STATE["tak_sent"] += 1
            except OSError as e:  # never let TAK problems stop the map; reconnect on the next message
                print(f"TAK send failed: {e}", file=sys.stderr)
                sender = None
                time.sleep(1.0)


def tak_send(xml: str):
    if STATE["tak"] is not None:
        STATE["tak"].send(xml)


def tak_send_review(alert: dict, decision: str):
    if alert.get("lat") is None:
        return
    if decision == "confirm":
        tak_send(alert_event(alert, ARGS.drone_uid, ARGS.ce, confirmed=True, stale_s=3600))
    else:
        tak_send(delete_event(f"{ARGS.drone_uid}-{alert['label']}-{alert['uid']}", alert["lat"], alert["lon"]))


def on_frame(text: str) -> list[str]:
    """One detection frame from one camera -> the messages for the operators.

    Detections and alerts get camera-scoped IDs ("rgb-5"). An alert that another camera has already raised for the
    same spot (esa.fusion) does not reach the operator again: an "alert_update" says which cameras now see it.
    New alerts, and the drone at 1 Hz, also go to TAK.
    """
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(msg, dict):
        return []
    camera = str(msg.get("camera") or "rgb")
    msg["camera"] = camera
    if msg.get("frame") == 0 and not ARGS.ros:  # a replay restarted: forget the previous pass
        STATE["alerts"].clear()
        STATE["reviews"].clear()
        STATE["fusion"].clear()
    for d in msg.get("detections") or []:
        if d.get("track_id") is not None:
            d["uid"] = alert_uid(camera, d["track_id"])
    now, new_alerts, updates = time.monotonic(), [], []
    for a in msg.get("alerts") or []:
        fa, is_new = STATE["fusion"].add(camera, a["track_id"], a.get("lat"), a.get("lon"), now)
        if is_new:
            a.update(uid=fa.uid, camera=camera, cameras=sorted(fa.cameras))
            STATE["alerts"][fa.uid] = a
            while len(STATE["alerts"]) > MAX_ALERTS:
                STATE["alerts"].popitem(last=False)
            new_alerts.append(a)
            if a.get("lat") is not None:
                tak_send(alert_event(a, ARGS.drone_uid, ARGS.ce))
        elif fa.uid in STATE["alerts"]:
            stored = STATE["alerts"][fa.uid]
            if stored.get("cameras") != sorted(fa.cameras):
                stored["cameras"] = sorted(fa.cameras)
                updates.append(json.dumps({"type": "alert_update", "uid": fa.uid, "cameras": stored["cameras"]}))
    msg["alerts"] = new_alerts
    d = msg.get("drone")
    if d and now - STATE["last_drone_cot"] >= 1.0:  # drone position to TAK at 1 Hz
        STATE["last_drone_cot"] = now
        tak_send(drone_event(ARGS.drone_uid, d["lat"], d["lon"], d.get("alt_m")))
    return [json.dumps(msg), *updates]


@app.get("/health")
def health():
    return {
        "clients": len(clients),
        "source": "ros" if ARGS.ros else ARGS.replay,
        "tak": bool(ARGS.tak),
        "tak_messages_sent": STATE["tak_sent"],
        "tak_messages_dropped": STATE["tak_dropped"],
        "alerts": len(STATE["alerts"]),
        "reviews": len(STATE["reviews"]),
    }


async def _send(c: WebSocket, msg: str):
    try:
        await asyncio.wait_for(c.send_text(msg), SEND_TIMEOUT_S)
    except Exception:  # gone, or too slow: drop it; the app reconnects and catches up
        clients.discard(c)


async def broadcast(msg: str):
    """To every operator at once, so one slow tablet does not delay the others."""
    await asyncio.gather(*(_send(c, msg) for c in list(clients)))


async def broadcaster():
    while True:
        for out in on_frame(await queue.get()):
            await broadcast(out)


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
            for topic in ARGS.camera_topics:  # one per camera: tracks_geo, or /rgb/tracks_geo /thermal/tracks_geo
                self.create_subscription(String, topic, lambda m: loop.call_soon_threadsafe(_offer, m.data), 10)
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
    ap.add_argument("--camera-topics", nargs="+", default=["tracks_geo"],
                    help="--ros: one tracks_geo topic per camera, e.g. /rgb/tracks_geo /thermal/tracks_geo")  # fmt: skip
    ap.add_argument("--fuse-radius", type=float, default=5.0,
                    help="m: alerts from different cameras this close (and --fuse-window s apart) are one alert")  # fmt: skip
    ap.add_argument("--fuse-window", type=float, default=30.0)
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
