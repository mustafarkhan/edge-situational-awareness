"""Search missions as the autopilot receives them: encoding, safety checks, autopilot settings, flight checks.

No MAVLink connection here, so everything can be unit-tested. `tools/fly_mission.py` is the thin layer that talks
to PX4 or ArduPilot.

Safety rules (checked before anything is uploaded; a plan that breaks one is refused):
  * no waypoint above the altitude limit (the EU open-category limit, 120 m above the take-off point, by default);
  * the plan carries an inclusion geofence and every waypoint lies inside it;
  * the plan starts with a take-off and ends with return-to-launch.
The autopilot is then told the same limits, so it enforces them itself: a fence ceiling at the altitude limit (not
above it), return-to-launch on a fence breach, and a return altitude chosen for the site rather than the
autopilot's default (15 m on ArduCopter, low enough to hit trees and masts on the way home).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_M_PER_DEG = 111_320.0
EU_MAX_ALT_M = 120.0
# Plans stay this far below the fence ceiling. Found in SITL: a mission flown exactly at the 120 m ceiling breached
# the altitude fence on normal altitude-hold overshoot, and the autopilot aborted the search.
CEILING_MARGIN_M = 5.0
MAX_PLAN_ALT_M = EU_MAX_ALT_M - CEILING_MARGIN_M

NAV_WAYPOINT, NAV_RTL, NAV_TAKEOFF, DO_CHANGE_SPEED = 16, 20, 22, 178
FENCE_INCLUSION, FENCE_EXCLUSION = 5001, 5002
FRAME_GLOBAL, FRAME_MISSION, FRAME_GLOBAL_RELATIVE_ALT = 0, 2, 3
POSITIONAL = {NAV_WAYPOINT, NAV_TAKEOFF, FENCE_INCLUSION, FENCE_EXCLUSION}


@dataclass(frozen=True)
class SafetyLimits:
    max_alt_m: float = EU_MAX_ALT_M  # the ceiling (fence), above the take-off point
    rtl_alt_m: float | None = None  # None: the search altitude (cleared during the search)
    margin_m: float = CEILING_MARGIN_M  # waypoints and the return stay this far below the ceiling

    @property
    def plan_ceiling(self) -> float:
        return self.max_alt_m - self.margin_m

    def return_alt(self, search_alt_m: float) -> float:
        alt = self.rtl_alt_m if self.rtl_alt_m is not None else search_alt_m
        return min(alt, self.plan_ceiling)


DEFAULT_LIMITS = SafetyLimits()

Item = tuple[int, int, list[float]]  # (MAV_CMD, MAV_FRAME, params[7]: p1..p4, lat/x, lon/y, alt/z)


def plan_items(plan: dict) -> list[Item]:
    """Mission items of a QGroundControl .plan, in order."""
    out = []
    for it in plan["mission"]["items"]:
        p = [0.0 if v is None else float(v) for v in it["params"]]
        out.append((int(it["command"]), int(it["frame"]), p))
    return out


def fence_items(plan: dict) -> list[Item]:
    """Geofence polygons of a .plan as MAVLink fence items (MAV_MISSION_TYPE_FENCE)."""
    out = []
    for poly in plan.get("geoFence", {}).get("polygons", []):
        cmd = FENCE_INCLUSION if poly.get("inclusion", True) else FENCE_EXCLUSION
        for lat, lon in poly["polygon"]:
            out.append((cmd, FRAME_GLOBAL, [len(poly["polygon"]), 0, 0, 0, lat, lon, 0]))
    return out


def mission_for(plan: dict, ardupilot: bool) -> list[Item]:
    """The items to upload. ArduPilot reserves item 0 for home (overwritten with the real home position)."""
    items = plan_items(plan)
    if ardupilot:
        lat0, lon0, _ = plan["mission"]["plannedHomePosition"]
        items = [(NAV_WAYPOINT, FRAME_GLOBAL, [0, 0, 0, 0, lat0, lon0, 0]), *items]
    return items


def encode(item: Item) -> tuple:
    """(frame, command, p1, p2, p3, p4, x, y, z) for MISSION_ITEM_INT: positions as degrees x 1e7 integers."""
    cmd, frame, p = item
    if cmd in POSITIONAL:
        x, y = int(round(p[4] * 1e7)), int(round(p[5] * 1e7))
    else:
        x, y = int(p[4]), int(p[5])
    return (frame, cmd, p[0], p[1], p[2], p[3], x, y, float(p[6]))


def search_waypoints(plan: dict) -> list[tuple[float, float, float]]:
    return [(p[4], p[5], p[6]) for c, _, p in plan_items(plan) if c == NAV_WAYPOINT]


def fence_polygon(plan: dict) -> list[tuple[float, float]] | None:
    polys = [p for p in plan.get("geoFence", {}).get("polygons", []) if p.get("inclusion", True)]
    return [tuple(v) for v in polys[0]["polygon"]] if polys else None


def inside(lat: float, lon: float, poly: list[tuple[float, float]]) -> bool:
    """Point in polygon (ray casting), for the small areas a drone searches."""
    hit = False
    for (la1, lo1), (la2, lo2) in zip(poly, poly[1:] + poly[:1]):
        if (lo1 > lon) != (lo2 > lon) and lat < (la2 - la1) * (lon - lo1) / (lo2 - lo1) + la1:
            hit = not hit
    return hit


def check_plan(plan: dict, limits: SafetyLimits = DEFAULT_LIMITS) -> list[str]:
    """Reasons this plan must not be uploaded; empty when it may be."""
    problems = []
    items = plan_items(plan)
    if not items or items[0][0] != NAV_TAKEOFF:
        problems.append("the mission must start with a take-off")
    if not items or items[-1][0] != NAV_RTL:
        problems.append("the mission must end with return-to-launch")
    wps = search_waypoints(plan)
    if not wps:
        problems.append("the mission has no search waypoints")
    alts = [p[6] for c, _, p in items if c in (NAV_WAYPOINT, NAV_TAKEOFF)]
    if alts and max(alts) > limits.plan_ceiling:
        problems.append(f"altitude {max(alts):g} m leaves less than {limits.margin_m:g} m below the "
                        f"{limits.max_alt_m:g} m ceiling")  # fmt: skip
    if alts and min(alts) <= 0:
        problems.append("a waypoint is at or below the take-off point")
    if limits.rtl_alt_m is not None and not 0 < limits.rtl_alt_m <= limits.plan_ceiling:
        problems.append(f"return altitude {limits.rtl_alt_m:g} m must be above 0 and at most {limits.plan_ceiling:g} m")
    poly = fence_polygon(plan)
    if poly is None:
        problems.append("the plan has no inclusion geofence")
    else:
        out = [i for i, (la, lo, _) in enumerate(wps) if not inside(la, lo, poly)]
        if out:
            problems.append(f"{len(out)} waypoint(s) outside the geofence")
    return problems


# ArduCopter before 4.7 named some parameters differently: new name -> (old name, scale from new units to old)
LEGACY_PARAMS = {"RTL_ALT_M": ("RTL_ALT", 100.0)}


def autopilot_params(plan: dict, limits: SafetyLimits, ardupilot: bool) -> dict[str, float]:
    """Parameters that make the autopilot itself enforce the limits (fence ceiling, breach action, return altitude)."""
    search_alt = max(a for _, _, a in search_waypoints(plan))
    rtl = limits.return_alt(search_alt)
    if ardupilot:  # ArduCopter
        return {
            "FENCE_TYPE": 4 | 1,  # polygon + maximum altitude (not the default 300 m circle around home)
            "FENCE_ALT_MAX": limits.max_alt_m,  # m
            "FENCE_ACTION": 1,  # return to launch on breach
            "RTL_ALT_M": rtl,  # m (ArduCopter 4.7+; older firmware: RTL_ALT in cm, see LEGACY_PARAMS)
            "FENCE_ENABLE": 1,
        }
    return {  # PX4
        "GF_MAX_VER_DIST": limits.max_alt_m,  # m above home
        "GF_ACTION": 3,  # return on breach
        "RTL_RETURN_ALT": rtl,  # m
    }


def _xy(lat: float, lon: float, lat0: float) -> tuple[float, float]:
    return lon * EARTH_M_PER_DEG * math.cos(math.radians(lat0)), lat * EARTH_M_PER_DEG


def dist_to_segment(p, a, b) -> float:
    """Metres from point p to segment a-b ((lat, lon, ...) tuples, local flat-earth)."""
    (px, py), (ax, ay), (bx, by) = _xy(p[0], p[1], a[0]), _xy(a[0], a[1], a[0]), _xy(b[0], b[1], a[0])
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-9)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def flight_report(plan: dict, track: list[tuple[float, float, float]], search_start: int, waypoints_reached: int,
                  landed: bool, limits: SafetyLimits = DEFAULT_LIMITS, search_end: int | None = None) -> dict:  # fmt: skip
    """Check a flown track (lat, lon, metres above home) against its plan. The distance from the planned path is
    measured on the search legs only: from the first to the last search waypoint (not the transit or the return)."""
    wps = search_waypoints(plan)
    search_alt = max(a for _, _, a in wps)
    legs = [t for t in track[search_start:search_end] if t[2] > 0.8 * search_alt]
    dev = sorted(min(dist_to_segment(t, wps[i], wps[i + 1]) for i in range(len(wps) - 1)) for t in legs)
    poly = fence_polygon(plan)
    outside = sum(1 for la, lo, _ in track if poly and not inside(la, lo, poly))
    max_alt = max((t[2] for t in track), default=0.0)
    rep = {
        "waypoints_planned": len(wps),
        "waypoints_reached": waypoints_reached,
        "track_points": len(track),
        "cross_track_m_median": round(dev[len(dev) // 2], 2) if dev else None,
        "cross_track_m_max": round(dev[-1], 2) if dev else None,
        "positions_outside_fence": outside,
        "max_alt_m": round(max_alt, 1),
        "landed": landed,
    }
    rep["ok"] = (
        waypoints_reached == len(wps) and outside == 0 and max_alt <= limits.max_alt_m + 2.0 and landed
    )  # 2 m: altitude-hold overshoot the autopilot itself tolerates
    return rep
