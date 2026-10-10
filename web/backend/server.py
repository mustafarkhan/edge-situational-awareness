"""WebSocket bridge between detections and the browser map.

Two sources:
  --replay outputs/run1.jsonl   replay a file from tools/detect_video.py (laptop, no ROS)
  --ros                          subscribe to /tracks_geo from the ROS 2 detector node

  python web/backend/server.py --replay outputs/run1.jsonl --rate 10
Then open the React app (web/frontend) which connects to ws://localhost:8000/ws.
"""
import argparse
import asyncio
import json
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()
clients: set[WebSocket] = set()
queue: asyncio.Queue = None  # created on startup
ARGS = None


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive; we don't expect messages
    except WebSocketDisconnect:
        clients.discard(websocket)


@app.get("/health")
def health():
    return {"clients": len(clients), "source": "ros" if ARGS.ros else ARGS.replay}


async def broadcaster():
    while True:
        msg = await queue.get()
        for c in list(clients):
            try:
                await c.send_text(msg)
            except Exception:
                clients.discard(c)


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
            self.create_subscription(String, "tracks_geo",
                                     lambda m: loop.call_soon_threadsafe(_offer, m.data), 10)

    rclpy.init()
    rclpy.spin(Bridge())


@app.on_event("startup")
async def startup():
    global queue
    queue = asyncio.Queue(maxsize=100)
    asyncio.create_task(broadcaster())
    if ARGS.ros:
        threading.Thread(target=ros_thread, args=(asyncio.get_running_loop(),), daemon=True).start()
    else:
        asyncio.create_task(replay(ARGS.replay, ARGS.rate))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay")
    src.add_argument("--ros", action="store_true")
    ap.add_argument("--rate", type=float, default=10.0, help="replay frames per second")
    ap.add_argument("--once", action="store_true", help="replay a single pass instead of looping")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ARGS = ap.parse_args()
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
