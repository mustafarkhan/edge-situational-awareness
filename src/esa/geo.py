"""Pixel -> latitude/longitude by intersecting the camera ray with flat ground (or the sea surface).

Model:
  * pinhole camera, no lens distortion, square pixels, horizontal field of view `hfov_deg`
  * camera points along `heading_deg` (0 = north, clockwise) and is tilted down by `pitch_deg`
    below the horizon (90 = straight down, the default; DJI gimbal pitch -90 corresponds to 90 here)
  * no roll; flat ground at `alt_m` below the camera (exact at sea, approximate over land)

With pitch 90 this reduces exactly to the simple straight-down ("nadir") model this project started
with. benchmarks/geo_sensitivity.py showed why the general model matters: at ~60 m, ignoring a 10 deg
gimbal tilt moves positions by ~12 m, more than a 5 m GPS error.

Pixels at or above the horizon have no ground intersection: pixel_to_latlon returns None for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_M_PER_DEG_LAT = 111_320.0


@dataclass
class CameraGeo:
    lat: float  # drone latitude (deg)
    lon: float  # drone longitude (deg)
    alt_m: float  # height above ground/sea (m)
    hfov_deg: float  # horizontal field of view of the camera (deg)
    heading_deg: float = 0.0  # direction the camera faces; 0 = north, clockwise positive
    pitch_deg: float = 90.0  # camera tilt below the horizon; 90 = straight down

    def ground_width_m(self) -> float:
        """Ground footprint width for a straight-down camera (used for quick estimates)."""
        return 2.0 * self.alt_m * math.tan(math.radians(self.hfov_deg) / 2.0)


def ground_offset_m(px: float, py: float, img_w: int, img_h: int, cam: CameraGeo) -> tuple[float, float] | None:
    """(east, north) metres from the point below the drone to where pixel (px, py) meets the ground."""
    f = (img_w / 2.0) / math.tan(math.radians(cam.hfov_deg) / 2.0)
    x, y = px - img_w / 2.0, py - img_h / 2.0  # image right, image down
    h, p = math.radians(cam.heading_deg), math.radians(cam.pitch_deg)
    fwd = (math.sin(h), math.cos(h), 0.0)  # horizontal heading (east, north, up)
    right = (math.cos(h), -math.sin(h), 0.0)
    axis = (math.cos(p) * fwd[0], math.cos(p) * fwd[1], -math.sin(p))  # optical axis
    up = (math.sin(p) * fwd[0], math.sin(p) * fwd[1], math.cos(p))  # image-top direction
    ray = [x * right[i] - y * up[i] + f * axis[i] for i in range(3)]
    if ray[2] >= -1e-9:
        return None  # at or above the horizon
    t = cam.alt_m / -ray[2]
    return t * ray[0], t * ray[1]


def pixel_to_latlon(px: float, py: float, img_w: int, img_h: int, cam: CameraGeo) -> tuple[float, float] | None:
    """Map an image pixel (e.g. a box's bottom-centre or centre) to (lat, lon), or None above the horizon."""
    off = ground_offset_m(px, py, img_w, img_h, cam)
    if off is None:
        return None
    east, north = off
    lat = cam.lat + north / EARTH_M_PER_DEG_LAT
    lon = cam.lon + east / (EARTH_M_PER_DEG_LAT * math.cos(math.radians(cam.lat)))
    return lat, lon
