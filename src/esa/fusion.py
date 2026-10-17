"""Alerts from several cameras on one drone (RGB + thermal) -> one alert per person for the operator.

Each camera's detector raises its own alerts, with its own track IDs: track 5 on the RGB camera and track 5 on the
thermal camera are different objects. Every alert therefore gets a camera-scoped ID ("rgb-5", "thermal-5").

The same person seen by both cameras would otherwise reach the operator twice. An alert from one camera that lands
within `radius_m` of an alert from ANOTHER camera, raised within `window_s`, joins that alert instead of creating a
new one: the operator sees one alert marked as seen by both cameras, and TAK gets one marker. Alerts from the same
camera are never merged here: within one camera that is the detector's job (esa.alerts de-duplication).

The radius covers the two cameras' different geolocation errors (the thermal lens is narrower and its pixels
coarser). It is a starting value, like the 3 m single-camera radius (README, section 11); measure it in the field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

M_PER_DEG = 111_320.0


def alert_uid(camera: str, track_id) -> str:
    return f"{camera}-{track_id}"


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dy = (lat2 - lat1) * M_PER_DEG
    dx = (lon2 - lon1) * M_PER_DEG * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


@dataclass
class FusedAlert:
    uid: str  # the first camera's ID; the one the operator reviews
    lat: float | None
    lon: float | None
    t: float
    cameras: set[str] = field(default_factory=set)
    members: set[str] = field(default_factory=set)  # every camera-scoped ID merged into it


class AlertFusion:
    def __init__(self, radius_m: float = 5.0, window_s: float = 30.0, max_alerts: int = 5000):
        self.radius_m, self.window_s, self.max_alerts = radius_m, window_s, max_alerts
        self.alerts: dict[str, FusedAlert] = {}  # uid -> fused alert, oldest first
        self.member_of: dict[str, str] = {}  # camera-scoped ID -> fused alert uid

    def add(self, camera: str, track_id, lat: float | None, lon: float | None, now: float) -> tuple[FusedAlert, bool]:
        """(the fused alert this one belongs to, True if it is new for the operator)."""
        uid = alert_uid(camera, track_id)
        if uid in self.member_of:  # the same track alerting again
            return self.alerts[self.member_of[uid]], False
        if lat is not None and lon is not None:
            for fa in reversed(self.alerts.values()):  # newest first
                if now - fa.t > self.window_s:
                    break
                if camera in fa.cameras or fa.lat is None:
                    continue
                if _dist_m(lat, lon, fa.lat, fa.lon) <= self.radius_m:
                    fa.cameras.add(camera)
                    fa.members.add(uid)
                    self.member_of[uid] = fa.uid
                    return fa, False
        fa = FusedAlert(uid, lat, lon, now, {camera}, {uid})
        self.alerts[uid] = fa
        self.member_of[uid] = uid
        while len(self.alerts) > self.max_alerts:  # forget the oldest
            old = self.alerts.pop(next(iter(self.alerts)))
            for m in old.members:
                self.member_of.pop(m, None)
        return fa, True

    def clear(self):
        self.alerts.clear()
        self.member_of.clear()
