"""Search planning: how high and how fast to fly so that a swimmer is likely to be found.

The trade-off: flying higher widens the camera footprint (more sea per minute) but shrinks each swimmer to fewer
pixels, and small objects are missed (README: recall falls steeply below ~16 px). This module turns a MEASURED
detection curve (probability of detecting a swimmer vs its size in pixels at the model input,
benchmarks/detection_vs_range.py) into the quantities search-and-rescue planners use:

  sweep width W   width of sea effectively searched on one pass (m): footprint width x probability of detection
  sweep rate      W x ground speed (km^2/h): how fast the area gets covered
  POD             probability of detection after covering an area with parallel tracks spaced S apart.
                  1 - exp(-W/S), the exponential ("random search") detection law: conservative compared with a
                  perfectly flown pattern, standard in SAR planning when the sweep is imperfect.

Assumptions (stated so they can be checked): camera pointing straight down, long image side across track, the
frame downscaled so its long side equals the model input; detection probability uniform across the frame;
calm water, daylight, the swimmer's head and shoulders visible. Consecutive looks at the same swimmer are not
independent; `independent_looks` caps how many count (measured on real clips by benchmarks/search_planning.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_M_PER_DEG_LAT = 111_320.0


@dataclass
class DetectionCurve:
    """Probability of detecting one swimmer in one frame, vs its size in pixels at the model input."""

    px: list[float]  # bin centres (median size in each bin), increasing
    p: list[float]  # measured detection probability in each bin
    min_px: float = 0.0  # smallest size with enough measurements; below it the curve is a guess

    @classmethod
    def from_bins(cls, px: list[float], p: list[float], n: list[int], min_px: float = 0.0) -> DetectionCurve:
        """Measured bins -> a curve that never gets worse as objects get bigger.

        Pool-adjacent-violators (isotonic regression, weighted by the number of swimmers per bin): neighbouring
        bins that dip are averaged rather than lifted to the best one, so the curve is not optimistic."""
        blocks = [[pi, ni, 1] for pi, ni in zip(p, n)]  # value, weight, bins merged
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                a, b = blocks[i], blocks[i + 1]
                w = a[1] + b[1]
                blocks[i : i + 2] = [[(a[0] * a[1] + b[0] * b[1]) / w, w, a[2] + b[2]]]
                i = max(i - 1, 0)
            else:
                i += 1
        fitted = [v for v, _, k in blocks for _ in range(k)]
        return cls(list(px), fitted, min_px)

    def measured(self, px: float) -> bool:
        return px >= self.min_px

    def __call__(self, px: float) -> float:
        # linear between measured bins; 0 at 0 px; flat beyond the largest measured size
        return float(np.interp(px, [0.0, *self.px], [0.0, *self.p]))


@dataclass
class Camera:
    hfov_deg: float = 65.5  # horizontal field of view (long side)
    aspect: float = 16 / 9  # long side / short side
    input_px: int = 1280  # model input size (long side)

    def footprint_m(self, alt_m: float) -> tuple[float, float]:
        """(across-track width, along-track length) of a straight-down camera's ground footprint."""
        w = 2 * alt_m * math.tan(math.radians(self.hfov_deg) / 2)
        return w, w / self.aspect

    def pixels_on(self, size_m: float, alt_m: float) -> float:
        """Size in model-input pixels of an object `size_m` across, seen straight down from `alt_m`."""
        return size_m * self.input_px / self.footprint_m(alt_m)[0]


def looks_per_pass(cam: Camera, alt_m: float, speed_mps: float, fps: float) -> float:
    """How many processed frames see a given point as the drone flies over it."""
    along = cam.footprint_m(alt_m)[1]
    return along * fps / max(speed_mps, 1e-6)


def p_pass(p_single: float, looks: float, independent_looks: float) -> float:
    """Probability of at least one detection during one pass, counting at most `independent_looks` looks."""
    n = max(1.0, min(looks, independent_looks))
    return 1 - (1 - p_single) ** n


@dataclass
class PlanPoint:
    alt_m: float
    speed_mps: float
    px_on_target: float
    p_single: float
    looks: float
    p_pass: float
    footprint_w_m: float
    sweep_width_m: float
    sweep_rate_km2_h: float
    in_measured_range: bool


def evaluate(
    curve: DetectionCurve,
    cam: Camera,
    alt_m: float,
    speed_mps: float,
    fps: float,
    swimmer_m: float,
    independent_looks: float,
) -> PlanPoint:
    px = cam.pixels_on(swimmer_m, alt_m)
    ps = curve(px)
    n = looks_per_pass(cam, alt_m, speed_mps, fps)
    pp = p_pass(ps, n, independent_looks)
    fw = cam.footprint_m(alt_m)[0]
    w = fw * pp
    return PlanPoint(alt_m, speed_mps, px, ps, n, pp, fw, w, w * speed_mps * 3.6 / 1000, curve.measured(px))


def max_altitude_measured(curve: DetectionCurve, cam: Camera, swimmer_m: float) -> float:
    """Highest altitude at which a swimmer still covers `curve.min_px` pixels (the edge of the measured data)."""
    return swimmer_m * cam.input_px / (curve.min_px * 2 * math.tan(math.radians(cam.hfov_deg) / 2))


def spacing_for_pod(sweep_width_m: float, pod: float) -> float:
    """Track spacing that gives `pod` under the exponential detection law (POD = 1 - exp(-W/S))."""
    if not 0 < pod < 1:
        raise ValueError("pod must be between 0 and 1")
    return sweep_width_m / -math.log(1 - pod)


def lawnmower(
    lat0: float, lon0: float, width_m: float, length_m: float, spacing_m: float, heading_deg: float = 0.0
) -> list[tuple[float, float]]:
    """Parallel-track ("lawnmower") waypoints covering a width x length rectangle.

    (lat0, lon0) is the starting corner; tracks run along `heading_deg` (0 = north) and step to the right.
    The first track is half a spacing in from the edge, so the edge strip gets the same coverage as the rest.
    """
    n = max(1, math.ceil(width_m / spacing_m))
    h = math.radians(heading_deg)
    fwd = (math.sin(h), math.cos(h))  # east, north
    right = (math.cos(h), -math.sin(h))
    pts = []
    for i in range(n):
        off = min((i + 0.5) * spacing_m, width_m)
        ends = [0.0, length_m] if i % 2 == 0 else [length_m, 0.0]
        for along in ends:
            e = off * right[0] + along * fwd[0]
            nn = off * right[1] + along * fwd[1]
            pts.append(
                (
                    lat0 + nn / EARTH_M_PER_DEG_LAT,
                    lon0 + e / (EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat0))),
                )
            )
    return pts


def path_length_m(pts: list[tuple[float, float]]) -> float:
    total = 0.0
    for (a, b), (c, d) in zip(pts, pts[1:]):
        dn = (c - a) * EARTH_M_PER_DEG_LAT
        de = (d - b) * EARTH_M_PER_DEG_LAT * math.cos(math.radians(a))
        total += math.hypot(dn, de)
    return total


def qgc_plan(pts: list[tuple[float, float]], alt_m: float, speed_mps: float) -> dict:
    """QGroundControl mission (.plan JSON) flying the waypoints at `alt_m` above home, for PX4 or ArduPilot."""
    items = [
        {
            "autoContinue": True,
            "command": 178,  # MAV_CMD_DO_CHANGE_SPEED
            "doJumpId": 1,
            "frame": 2,
            "params": [1, speed_mps, -1, 0, 0, 0, 0],
            "type": "SimpleItem",
        }
    ]
    for i, (lat, lon) in enumerate(pts, start=2):
        items.append(
            {
                "AMSLAltAboveTerrain": None,
                "Altitude": alt_m,
                "AltitudeMode": 1,
                "autoContinue": True,
                "command": 16,  # MAV_CMD_NAV_WAYPOINT
                "doJumpId": i,
                "frame": 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
                "params": [0, 0, 0, None, lat, lon, alt_m],
                "type": "SimpleItem",
            }
        )
    return {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": speed_mps,
            "firmwareType": 12,  # PX4; QGC converts for ArduPilot
            "hoverSpeed": 5,
            "items": items,
            "plannedHomePosition": [pts[0][0], pts[0][1], 0],
            "vehicleType": 2,  # multirotor
            "version": 2,
        },
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }
