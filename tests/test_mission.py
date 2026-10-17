"""Mission encoding, safety checks and autopilot settings (esa.mission)."""

import copy
import json
from pathlib import Path

import pytest

from esa.mission import (
    FENCE_INCLUSION,
    NAV_RTL,
    NAV_TAKEOFF,
    NAV_WAYPOINT,
    SafetyLimits,
    autopilot_params,
    check_plan,
    encode,
    fence_items,
    flight_report,
    inside,
    mission_for,
    search_waypoints,
)
from esa.search import lawnmower, qgc_plan

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def plan():
    pts = lawnmower(61.49, 23.735, 200, 300, 50.0, heading_deg=0)
    return json.loads(json.dumps(qgc_plan(pts, 100.0, 10.0)))


def test_shipped_example_plans_pass_the_checks():
    for name in ("search_example.plan", "land_search_example.plan"):
        assert check_plan(json.loads((REPO / "samples" / name).read_text())) == []


def test_plan_writer_refuses_altitudes_above_the_limit():
    pts = lawnmower(61.49, 23.735, 100, 100, 50.0)
    with pytest.raises(ValueError):
        qgc_plan(pts, 150.0, 10.0)
    with pytest.raises(ValueError):
        qgc_plan(pts, 0.0, 10.0)
    assert qgc_plan(pts, 150.0, 10.0, max_alt_m=200.0)  # where a higher limit is authorised


def test_check_plan_catches_each_problem(plan):
    def broken(edit):
        p = copy.deepcopy(plan)
        edit(p)
        return " | ".join(check_plan(p))

    def too_high(p):
        p["mission"]["items"][3]["params"][6] = 150.0

    def outside(p):
        p["mission"]["items"][4]["params"][4] += 0.01  # ~1.1 km north

    assert "below the 120 m ceiling" in broken(too_high)
    assert "outside the geofence" in broken(outside)
    assert "no inclusion geofence" in broken(lambda p: p["geoFence"].update(polygons=[]))
    assert "return-to-launch" in broken(lambda p: p["mission"]["items"].pop())
    assert "take-off" in broken(lambda p: p["mission"]["items"].pop(0))
    assert check_plan(plan, SafetyLimits(rtl_alt_m=200.0)) != []
    assert check_plan(plan, SafetyLimits(max_alt_m=90.0)) != []  # a stricter site limit


def test_ardupilot_gets_home_item_and_scaled_positions(plan):
    items = mission_for(plan, ardupilot=True)
    assert items[0][0] == NAV_WAYPOINT and items[1][0] == NAV_TAKEOFF and items[-1][0] == NAV_RTL
    assert len(items) == len(mission_for(plan, ardupilot=False)) + 1
    frame, cmd, *_, x, y, z = encode(items[2])  # the speed command is not positional
    assert cmd == 178 and (x, y) == (0, 0)
    frame, cmd, *_, x, y, z = encode(items[3])  # first waypoint
    lat, lon, alt = search_waypoints(plan)[0]
    assert (x, y) == (round(lat * 1e7), round(lon * 1e7)) and z == alt == 100.0
    fence = fence_items(plan)
    assert len(fence) == 4 and all(c == FENCE_INCLUSION for c, _, _ in fence)
    assert all(isinstance(v, int) and abs(v) > 1e8 for v in encode(fence[0])[6:8])


def test_autopilot_enforces_the_limit_not_more(plan):
    ap = autopilot_params(plan, SafetyLimits(), ardupilot=True)
    assert ap["FENCE_ALT_MAX"] == 120.0  # the legal ceiling, not search altitude + margin
    assert ap["FENCE_TYPE"] == 5 and ap["FENCE_ACTION"] == 1 and ap["FENCE_ENABLE"] == 1
    assert ap["RTL_ALT_M"] == 100.0  # return at the search altitude, not ArduCopter's 15 m default
    assert autopilot_params(plan, SafetyLimits(rtl_alt_m=60.0), True)["RTL_ALT_M"] == 60.0
    px4 = autopilot_params(plan, SafetyLimits(rtl_alt_m=60.0), ardupilot=False)
    assert px4 == {"GF_MAX_VER_DIST": 120.0, "GF_ACTION": 3, "RTL_RETURN_ALT": 60.0}


def test_point_in_polygon():
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert inside(0.5, 0.5, sq) and not inside(1.5, 0.5, sq) and not inside(0.5, -0.1, sq)


def test_flight_report(plan):
    wps = search_waypoints(plan)
    track = [(la, lo, 100.0) for la, lo, _ in wps] + [(wps[0][0], wps[0][1], 0.2)]
    rep = flight_report(plan, track, 0, len(wps), landed=True)
    assert rep["ok"] and rep["cross_track_m_max"] < 0.01 and rep["positions_outside_fence"] == 0
    assert not flight_report(plan, track, 0, len(wps) - 1, landed=True)["ok"]  # a waypoint missed
    assert not flight_report(plan, track, 0, len(wps), landed=False)["ok"]
    high = track + [(wps[0][0], wps[0][1], 130.0)]
    assert not flight_report(plan, high, 0, len(wps), landed=True)["ok"]  # above the limit
    away = track + [(wps[0][0] + 0.05, wps[0][1], 100.0)]
    assert flight_report(plan, away, 0, len(wps), landed=True)["positions_outside_fence"] == 1
