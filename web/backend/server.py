"""Ground-station bridge: detections -> browser map, operator reviews, and TAK (ATAK / WinTAK / TAK Server).

Sources (one of):
  --replay outputs/run1.jsonl   replay a file from tools/detect_video.py (laptop, no ROS)
  --ros                          subscribe to /tracks_geo from the ROS 2 detector node

Operator loop: the map app sends {"type": "review", "track_id": .., "decision": "confirm" | "reject"}.
Each review is appended to an audit log (--reviews), broadcast to every connected operator, forwarded to TAK
(confirm: marker updated to "CONFIRMED"; reject: marker deleted) and, with --ros, published on /alert_reviews so
the detector suppresses new alerts at that spot (esa.alerts.AlertManager.reject).

  python web/backend/server.py --replay web/sample/sea_DJI_0055.jsonl --rate 15 --tak udp://239.2.3.1:6969
Then open the React app (web/frontend), which connects to ws://localhost:8000/ws.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from esa.cot import CotSender, alert_event, delete_event, drone_event  # noqa: E402

clients: set[WebSocket] = set()
queue: asyncio.Queue | None = None  # created on startup
ARGS = argparse.Namespace(ros=False, replay=None, rate=10.0, once=False, tak=None, drone_uid="esa-drone-1",
                          reviews="outputs/reviews.jsonl", ce=15.0)  # fmt: skip
STATE = {"alerts": {}, "reviews": {}, "tak": None, "tak_sent": 0, "last_drone_cot": 0.0, "ros_pub": None}


@asynccontextmanager
async def lifespan(_app):
    global queue
    queue = asyncio.Queue(maxsize=100)
    if ARGS.tak:
        STATE["tak"] = CotSender(ARGS.tak)
    asyncio.create_task(broadcaster())
    if ARGS.ros:
        threading.Thread(target=ros_thread, args=(asyncio.get_running_loop(),), daemon=True).start()
    elif ARGS.replay:
        asyncio.create_task(replay(ARGS.replay, ARGS.rate))
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    # a newly connected operator sees decisions already made
    for r in STATE["reviews"].values():
        await websocket.send_text(json.dumps(r))
    try:
        while True:
            text = await websocket.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "review":
                await handle_review(msg)
    except WebSocketDisconnect:
        clients.discard(websocket)


async def handle_review(msg: dict) -> dict | None:
    tid, decision = msg.get("track_id"), msg.get("decision")
    alert = STATE["alerts"].get(tid)
    if alert is None or decision not in ("confirm", "reject"):
        return None
    review = {
        "type": "review",
        "track_id": tid,
        "decision": decision,
        "label": alert["label"],
        "lat": alert.get("lat"),
        "lon": alert.get("lon"),
        "operator": str(msg.get("operator", "operator"))[:40],
        "time": time.time(),
    }
    STATE["reviews"][tid] = review
    if ARGS.reviews:
        Path(ARGS.reviews).parent.mkdir(parents=True, exist_ok=True)
        with open(ARGS.reviews, "a") as fh:
            fh.write(json.dumps(review) + "\n")
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
        "tak": ARGS.tak,
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
    ap.add_argument("--reviews", default="outputs/reviews.jsonl", help="operator decisions audit log")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args(argv)


if __name__ == "__main__":
    ARGS = parse_args()
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
