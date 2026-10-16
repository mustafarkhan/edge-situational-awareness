"""Upload a search mission (.plan from benchmarks/search_planning.py or land_search.py) to a PX4 or ArduPilot drone.

  python tools/fly_mission.py --connect udp:127.0.0.1:14550 --plan samples/search_example.plan          # upload only
  python tools/fly_mission.py --connect tcp:127.0.0.1:5760 --plan ... --sitl --fly --report out.json    # simulator

Safety by design:
  * on a real vehicle this tool only UPLOADS the mission and the geofence. Arming, take-off and starting the mission
    stay with the pilot in the ground station, as the operating rules require;
  * --fly refuses to run unless --sitl is given AND the autopilot reports itself as a software-in-the-loop
    simulator (SIM_* parameters present on ArduPilot, or a SITL build on PX4);
  * the geofence from the plan is uploaded and enabled, so the autopilot itself keeps the drone inside the
    approved area whatever the software above it does.

With --fly it arms, runs the mission, records the flown track and checks it: every waypoint reached, maximum
distance from the planned path, and whether the drone ever left the fence. Used by the ArduPilot SITL test in CI.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from pymavlink import mavutil

EARTH = 111_320.0
MISSION, FENCE = 0, 1  # MAV_MISSION_TYPE_*


def plan_items(plan: dict) -> list[tuple[int, int, list[float]]]:
    """(command, frame, params[7]) for each mission item in a QGroundControl .plan."""
    out = []
    for it in plan["mission"]["items"]:
        p = [0.0 if v is None else float(v) for v in it["params"]]
        out.append((int(it["command"]), int(it["frame"]), p))
    return out


def fence_items(plan: dict) -> list[tuple[int, int, list[float]]]:
    out = []
    for poly in plan.get("geoFence", {}).get("polygons", []):
        cmd = 5001 if poly.get("inclusion", True) else 5002  # MAV_CMD_NAV_FENCE_POLYGON_VERTEX_(IN|EX)CLUSION
        for lat, lon in poly["polygon"]:
            out.append((cmd, 0, [len(poly["polygon"]), 0, 0, 0, lat, lon, 0]))
    return out


def upload(m, items, mission_type: int, timeout: float = 20.0):
    """MAVLink mission protocol: COUNT -> autopilot requests each item -> ACK."""
    m.mav.mission_clear_all_send(m.target_system, m.target_component, mission_type)
    m.recv_match(type="MISSION_ACK", blocking=True, timeout=3)
    m.mav.mission_count_send(m.target_system, m.target_component, len(items), mission_type)
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = m.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], blocking=True, timeout=5)
        if msg is None:
            continue
        if msg.get_type() == "MISSION_ACK":
            if msg.type != 0:
                raise RuntimeError(f"autopilot rejected the upload (MAV_MISSION_RESULT {msg.type})")
            return
        cmd, frame, p = items[msg.seq]
        is_pos = frame in (3, 6, 0) and cmd not in (178, 20)
        m.mav.mission_item_int_send(
            m.target_system, m.target_component, msg.seq, frame, cmd, 0, 1,
            p[0], p[1], p[2], p[3],
            int(round(p[4] * 1e7)) if is_pos else int(p[4]),
            int(round(p[5] * 1e7)) if is_pos else int(p[5]),
            p[6], mission_type,
        )  # fmt: skip
    raise TimeoutError("mission upload timed out")


def wait_param(m, name: str, timeout: float = 5.0):
    """The PARAM_VALUE for `name` (others may arrive first, e.g. replies to earlier requests)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg is not None and msg.param_id.rstrip("\x00") == name:
            return msg
    return None


def set_param(m, name: str, value: float):
    for _ in range(3):
        m.mav.param_set_send(
            m.target_system, m.target_component, name.encode(), value, mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        msg = wait_param(m, name, 3)
        if msg is not None and abs(msg.param_value - value) < 1e-3:
            return
    raise RuntimeError(f"autopilot did not accept {name}={value}")


def is_sitl(m) -> bool:
    """ArduPilot SITL has SIM_* parameters; real autopilots do not."""
    m.mav.param_request_read_send(m.target_system, m.target_component, b"SIM_SPEEDUP", -1)
    return wait_param(m, "SIM_SPEEDUP") is not None


def dist_to_segment(p, a, b) -> float:
    """Metres from point p to segment a-b (lat/lon, local flat-earth)."""
    k = math.cos(math.radians(a[0]))
    ax, ay, bx, by = a[1] * EARTH * k, a[0] * EARTH, b[1] * EARTH * k, b[0] * EARTH
    px, py = p[1] * EARTH * k, p[0] * EARTH
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-9)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", required=True, help="udp:127.0.0.1:14550, tcp:127.0.0.1:5760, /dev/ttyACM0 ...")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--sitl", action="store_true", help="the target is a simulator (required for --fly)")
    ap.add_argument("--fly", action="store_true", help="SIMULATOR ONLY: arm, run the mission, check the flown track")
    ap.add_argument("--timeout", type=float, default=900.0, help="--fly: give up after this many seconds")
    ap.add_argument("--report", default=None, help="--fly: write the check results here (JSON)")
    a = ap.parse_args()

    plan = json.loads(Path(a.plan).read_text())
    m = mavutil.mavlink_connection(a.connect, source_system=250)
    hb = m.wait_heartbeat(timeout=30)
    if hb is None:
        sys.exit("no heartbeat from the autopilot")
    ardupilot = hb.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
    print(f"connected: {'ArduPilot' if ardupilot else 'PX4' if hb.autopilot == 12 else hb.autopilot}, "
          f"system {m.target_system}")  # fmt: skip

    items = plan_items(plan)
    if ardupilot:  # ArduPilot keeps mission item 0 for home; it is overwritten by the real home position
        lat0, lon0, _ = plan["mission"]["plannedHomePosition"]
        items = [(16, 0, [0, 0, 0, 0, lat0, lon0, 0])] + items
    fence = fence_items(plan)
    if fence:
        upload(m, fence, FENCE)
        if ardupilot:
            set_param(m, "FENCE_TYPE", 4 | 1)  # polygon + max altitude (not the default 300 m circle around home)
            set_param(m, "FENCE_ALT_MAX", max(p[6] for c, f, p in items if c == 16) + 30)
            set_param(m, "FENCE_ACTION", 1)  # RTL on breach
            set_param(m, "FENCE_ENABLE", 1)
        print(f"geofence uploaded: {len(fence)} vertices")
    upload(m, items, MISSION)
    n_wp = sum(1 for c, _, _ in items if c == 16) - (1 if ardupilot else 0)
    print(f"mission uploaded: {len(items)} items ({n_wp} search waypoints)")
    if not a.fly:
        print("upload only: arm and start the mission from the ground station")
        return

    if not a.sitl or not is_sitl(m):
        sys.exit("--fly is for simulators only: give --sitl, and the autopilot must report SIM_* parameters")
    deadline = time.time() + 180
    while time.time() < deadline and not m.motors_armed():  # retry until the EKF is happy enough to arm
        m.set_mode("GUIDED")
        m.arducopter_arm()
        t_arm = time.time() + 5
        while time.time() < t_arm and not m.motors_armed():
            m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
    if not m.motors_armed():
        sys.exit("could not arm in the simulator")
    m.set_mode("AUTO")
    m.mav.command_long_send(m.target_system, m.target_component, 300, 0, 0, 0, 0, 0, 0, 0, 0)  # MISSION_START
    print("armed, mission started")

    m.mav.request_data_stream_send(m.target_system, m.target_component, 0, 4, 1)  # all streams at 4 Hz
    wps = [(p[4], p[5]) for c, _, p in plan_items(plan) if c == 16]
    fence_poly = plan["geoFence"]["polygons"][0]["polygon"] if plan["geoFence"]["polygons"] else None
    track, reached, events, t0 = [], set(), [], time.time()
    first_wp_seq = next(i for i, (c, _, _) in enumerate(items) if c == 16 and not (ardupilot and i == 0))
    search_start = None
    last_seq = len(items) - 1
    while time.time() - t0 < a.timeout:
        msg = m.recv_match(
            type=["GLOBAL_POSITION_INT", "MISSION_ITEM_REACHED", "HEARTBEAT", "STATUSTEXT"], blocking=True, timeout=5
        )
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT":
            events.append(msg.text)
            print("autopilot:", msg.text, flush=True)
        elif msg.get_type() == "HEARTBEAT" and msg.get_srcSystem() == m.target_system:
            if track and max(t[2] for t in track) > 5 and not m.motors_armed():
                events.append("disarmed before the mission finished")
                break
        elif msg.get_type() == "GLOBAL_POSITION_INT":
            track.append((msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000))
        elif msg.get_type() == "MISSION_ITEM_REACHED":
            reached.add(msg.seq)
            if msg.seq == first_wp_seq:
                search_start = len(track)  # the search legs begin at the first search waypoint
            print(f"reached item {msg.seq}", flush=True)
            if msg.seq >= last_seq - 1:  # last search waypoint reached; RTL follows
                break
    # check the flown search legs (from the first search waypoint on, at search altitude) against the planned path
    alt = wps and [p[6] for c, _, p in plan_items(plan) if c == 16][0]
    legs = [t for t in track[search_start or 0 :] if t[2] > 0.8 * alt]
    dev = [min(dist_to_segment(t, wps[i], wps[i + 1]) for i in range(len(wps) - 1)) for t in legs]
    outside = 0
    if fence_poly:
        s_, w_ = fence_poly[0]
        n_, e_ = fence_poly[2]
        outside = sum(1 for la, lo, _ in track if not (s_ <= la <= n_ and w_ <= lo <= e_))
    wp_seqs = {i for i, (c, _, _) in enumerate(items) if c == 16 and not (ardupilot and i == 0)}
    rep = {
        "autopilot": "ArduPilot" if ardupilot else "PX4",
        "waypoints_planned": len(wps),
        "waypoints_reached": len(wp_seqs & reached),
        "track_points": len(track),
        "cross_track_m_median": round(sorted(dev)[len(dev) // 2], 2) if dev else None,
        "cross_track_m_max": round(max(dev), 2) if dev else None,
        "positions_outside_fence": outside,
        "wall_seconds": round(time.time() - t0, 1),
        "autopilot_messages": events[-20:],
    }
    print(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))
    ok = rep["waypoints_reached"] == len(wps) and outside == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
