"""Pixel -> approximate latitude/longitude for a camera looking straight down.

This is deliberately simple and its assumptions are part of the Limitations section:
  * camera points straight down (nadir), no gimbal pitch/roll
  * flat ground / sea surface, no terrain
  * no lens distortion, square pixels
  * drone position, altitude and heading are known exactly
Error grows quickly with tilt and altitude error. For real use you would take the
drone's telemetry (MAVLink) and the gimbal attitude and do a proper ray-ground intersection.
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
    heading_deg: float = 0.0  # 0 = image top points north, clockwise positive

    def ground_width_m(self) -> float:
        return 2.0 * self.alt_m * math.tan(math.radians(self.hfov_deg) / 2.0)


def pixel_to_latlon(px: float, py: float, img_w: int, img_h: int, cam: CameraGeo) -> tuple[float, float]:
    """Map an image pixel (e.g. box centre) to (lat, lon)."""
    m_per_px = cam.ground_width_m() / img_w
    # offsets from image centre, in metres, in the camera frame (right = +x, up = +y)
    dx = (px - img_w / 2.0) * m_per_px
    dy = (img_h / 2.0 - py) * m_per_px
    # rotate camera frame into east/north using heading
    h = math.radians(cam.heading_deg)
    east = dx * math.cos(h) + dy * math.sin(h)
    north = -dx * math.sin(h) + dy * math.cos(h)
    lat = cam.lat + north / EARTH_M_PER_DEG_LAT
    lon = cam.lon + east / (EARTH_M_PER_DEG_LAT * math.cos(math.radians(cam.lat)))
    return lat, lon
