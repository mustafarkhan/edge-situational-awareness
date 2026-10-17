"""Connect to the ground station as an operator and check what the RGB + thermal setup delivers. For test_dual.sh.

  python3 check_ground_station.py --token TOKEN --seconds 90 --out result.json

Passes when frames arrive from both cameras, detections carry camera-scoped IDs, at least one alert is raised, and
at least one alert is reported as seen by both cameras (both cameras watch the same scene in the test).
"""

import argparse
import asyncio
import json
import sys
import time

import websockets


async def collect(url: str, token: str, seconds: float) -> dict:
    seen = {"cameras": set(), "uids": set(), "alerts": [], "fused": [], "frames": 0}
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "auth", "token": token}))
        t0 = time.time()
        while time.time() - t0 < seconds:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
            except asyncio.TimeoutError:
                continue
            if msg.get("type") == "alert_update":
                seen["fused"].append(msg)
            elif "frame" in msg:
                seen["frames"] += 1
                seen["cameras"].add(msg.get("camera"))
                seen["uids"].update(d["uid"] for d in msg.get("detections", []) if "uid" in d)
                seen["alerts"] += msg.get("alerts", [])
            if seen["fused"] and len(seen["cameras"]) == 2:
                break
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--token", required=True)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = asyncio.run(collect(a.url, a.token, a.seconds))
    checks = {
        "frames from the RGB camera": "rgb" in s["cameras"],
        "frames from the thermal camera": "thermal" in s["cameras"],
        "detections carry camera-scoped IDs": any(u.startswith("rgb-") for u in s["uids"])
        and any(u.startswith("thermal-") for u in s["uids"]),
        "an alert reached the operator": bool(s["alerts"]),
        "a person seen by both cameras is one alert": any(f["cameras"] == ["rgb", "thermal"] for f in s["fused"]),
    }
    result = {"frames": s["frames"], "alerts": [a["uid"] for a in s["alerts"]], "fused": s["fused"], "checks": checks}
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    for name, ok in checks.items():
        print(f"{'OK  ' if ok else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
