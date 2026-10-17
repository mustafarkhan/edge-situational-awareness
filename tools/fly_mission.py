"""Upload a search mission (.plan from benchmarks/search_planning.py or land_search.py) to a PX4 or ArduPilot drone.

  python tools/fly_mission.py --connect udp:127.0.0.1:14550 --plan samples/search_example.plan          # upload only
  python tools/fly_mission.py --connect tcp:127.0.0.1:5760 --plan ... --sitl --fly --report out.json    # simulator

Before anything is sent, the plan is checked (esa.mission.check_plan): altitude limit (120 m by default), an
inclusion geofence containing every waypoint, take-off first and return-to-launch last. A plan that fails is refused.

Then the autopilot is made to enforce the same limits itself: geofence uploaded and enabled, fence ceiling at the
altitude limit, return-to-launch on breach, and a return altitude for the site (--rtl-alt; default the search
altitude, which the drone has just flown over) instead of the autopilot's own default. Each setting is read back.

On a real vehicle the tool only uploads; arming and starting stay with the pilot in the ground station. --fly runs
only when --sitl is given AND the autopilot reports simulator parameters (ArduPilot SITL: SIM_*). It flies the whole
sortie to landing and checks the track: every waypoint reached, distance from the planned path, never outside the
fence, never above the altitude limit, landed.

Tested end to end in CI against ArduPilot SITL. The PX4 parameter names follow PX4's documentation but have not yet
been exercised against PX4 SITL.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

from pymavlink import mavutil

from esa.mission import (
    LEGACY_PARAMS,
    NAV_WAYPOINT,
    SafetyLimits,
    autopilot_params,
    check_plan,
    encode,
    fence_items,
    flight_report,
    mission_for,
)

MISSION, FENCE = 0, 1  # MAV_MISSION_TYPE_*
PX4_INT_PARAMS = {"GF_ACTION"}  # PX4 stores these as INT32; sent bit-cast into the float field, as MAVLink requires


def upload(m, items, mission_type: int, timeout: float = 20.0):
    """MAVLink mission protocol: COUNT -> the autopilot requests each item -> ACK."""
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
        frame, cmd, p1, p2, p3, p4, x, y, z = encode(items[msg.seq])
        m.mav.mission_item_int_send(m.target_system, m.target_component, msg.seq, frame, cmd, 0, 1,
                                    p1, p2, p3, p4, x, y, z, mission_type)  # fmt: skip
    raise TimeoutError("mission upload timed out")


def wait_param(m, name: str, timeout: float = 5.0):
    """The PARAM_VALUE for `name` (others may arrive first, e.g. replies to earlier requests)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg is not None and msg.param_id.rstrip("\x00") == name:
            return msg
    return None


def set_param(m, name: str, value: float, ardupilot: bool):
    """Set and read back. ArduPilot takes every value as a float; PX4 integer parameters are bit-cast."""
    if not ardupilot and name in PX4_INT_PARAMS:
        wire, ptype = struct.unpack("<f", struct.pack("<i", int(value)))[0], mavutil.mavlink.MAV_PARAM_TYPE_INT32
    else:
        wire, ptype = float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    for _ in range(3):
        m.mav.param_set_send(m.target_system, m.target_component, name.encode(), wire, ptype)
        msg = wait_param(m, name, 3)
        if msg is not None and abs(msg.param_value - wire) < 1e-3:
            return
    raise RuntimeError(f"autopilot did not accept {name}={value}")


def param_exists(m, name: str) -> bool:
    m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
    return wait_param(m, name) is not None


def is_sitl(m) -> bool:
    """ArduPilot SITL has SIM_* parameters; real autopilots do not."""
    return param_exists(m, "SIM_SPEEDUP")


def fly(m, plan: dict, items, ardupilot: bool, limits: SafetyLimits, timeout: float) -> dict:
    """Arm in the simulator, run the mission to landing, and check the flown track."""
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
    m.mav.request_data_stream_send(m.target_system, m.target_component, 0, 4, 1)  # all streams at 4 Hz
    print("armed, mission started", flush=True)

    wp_seqs = [i for i, (c, _, _) in enumerate(items) if c == NAV_WAYPOINT and not (ardupilot and i == 0)]
    track, reached, events, search_start, search_end, landed = [], set(), [], 0, None, False
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type=["GLOBAL_POSITION_INT", "MISSION_ITEM_REACHED", "HEARTBEAT", "STATUSTEXT"],
                           blocking=True, timeout=5)  # fmt: skip
        if msg is None:
            continue
        kind = msg.get_type()
        if kind == "STATUSTEXT":
            events.append(msg.text)
        elif kind == "GLOBAL_POSITION_INT":
            track.append((msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000))
        elif kind == "MISSION_ITEM_REACHED":
            reached.add(msg.seq)
            if msg.seq == wp_seqs[0]:
                search_start = len(track)  # the search legs begin at the first search waypoint
            if msg.seq == wp_seqs[-1]:
                search_end = len(track)  # ... and end at the last; the return home follows
        elif kind == "HEARTBEAT" and msg.get_srcSystem() == m.target_system:
            if track and max(t[2] for t in track) > 5 and not m.motors_armed():
                landed = track[-1][2] < 1.0  # disarmed on the ground after the return
                break
    rep = flight_report(plan, track, search_start, len(set(wp_seqs) & reached), landed, limits, search_end)
    rep["autopilot"] = "ArduPilot" if ardupilot else "PX4"
    rep["sim_wall_seconds"] = round(time.time() - t0, 1)
    if not rep["ok"]:  # keep the autopilot's last messages only when they are needed to explain a failure
        rep["autopilot_messages"] = events[-20:]
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", required=True, help="udp:127.0.0.1:14550, tcp:127.0.0.1:5760, /dev/ttyACM0 ...")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--max-alt", type=float, default=SafetyLimits.max_alt_m,
                    help="altitude ceiling above take-off, m (EU open category: 120); plans must stay 5 m below it")  # fmt: skip
    ap.add_argument("--rtl-alt", type=float, default=None,
                    help="return-to-launch altitude, m (default: the search altitude); clear of obstacles on the way home")  # fmt: skip
    ap.add_argument("--sitl", action="store_true", help="the target is a simulator (required for --fly)")
    ap.add_argument("--fly", action="store_true", help="SIMULATOR ONLY: arm, run the mission, check the flown track")
    ap.add_argument("--timeout", type=float, default=1200.0, help="--fly: give up after this many seconds")
    ap.add_argument("--report", default=None, help="--fly: write the check results here (JSON)")
    a = ap.parse_args(argv)

    plan = json.loads(Path(a.plan).read_text())
    limits = SafetyLimits(max_alt_m=a.max_alt, rtl_alt_m=a.rtl_alt)
    problems = check_plan(plan, limits)
    if problems:
        sys.exit("plan refused:\n  - " + "\n  - ".join(problems))

    m = mavutil.mavlink_connection(a.connect, source_system=250)
    hb = m.wait_heartbeat(timeout=30)
    if hb is None:
        sys.exit("no heartbeat from the autopilot")
    ardupilot = hb.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
    print(f"connected: {'ArduPilot' if ardupilot else 'PX4'}, system {m.target_system}", flush=True)

    items = mission_for(plan, ardupilot)
    upload(m, fence_items(plan), FENCE)
    params = autopilot_params(plan, limits, ardupilot)
    for name, value in list(params.items()):
        if ardupilot and name in LEGACY_PARAMS and not param_exists(m, name):  # older ArduCopter
            old, scale = LEGACY_PARAMS[name]
            params[old] = params.pop(name) * scale
            name, value = old, params[old]
        set_param(m, name, value, ardupilot)
    upload(m, items, MISSION)
    print(f"geofence and mission uploaded; autopilot settings read back: {params}", flush=True)
    if not a.fly:
        print("upload only: arm and start the mission from the ground station")
        return
    if not a.sitl or not is_sitl(m):
        sys.exit("--fly is for simulators only: give --sitl, and the autopilot must report SIM_* parameters")
    rep = fly(m, plan, items, ardupilot, limits, a.timeout)
    rep["autopilot_settings"] = params
    print(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
