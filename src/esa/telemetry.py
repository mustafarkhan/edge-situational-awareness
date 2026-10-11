"""Latest drone telemetry, with an age check, turned into the camera model used for geolocation.

The real-footage test (README section 9) showed that geolocation is only as good as the attitude it
is given, and that a confidently wrong position is worse than none. So this class refuses to produce
a camera model when any required value is missing or older than `timeout_s`: the node then publishes
detections without coordinates instead of publishing stale ones.

Independent of ROS so it can be unit-tested; the ROS 2 node feeds it from topic callbacks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from esa.geo import CameraGeo

REQUIRED = ("lat", "lon", "alt_m", "heading_deg", "pitch_deg")


@dataclass
class TelemetryState:
    hfov_deg: float
    timeout_s: float = 1.0
    defaults: dict = field(default_factory=dict)  # values used when a field has no topic (e.g. fixed gimbal)
    _values: dict = field(default_factory=dict)
    _stamps: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, now: float, **values: float) -> None:
        """Record new values (e.g. update(t, lat=.., lon=..)), each stamped with `now` (seconds)."""
        with self._lock:
            for k, v in values.items():
                if k not in REQUIRED:
                    raise KeyError(k)
                self._values[k] = float(v)
                self._stamps[k] = now

    def problems(self, now: float) -> list[str]:
        """Why no camera model can be produced right now (empty list = ready)."""
        out = []
        with self._lock:
            for k in REQUIRED:
                if k in self._values:
                    age = now - self._stamps[k]
                    if age > self.timeout_s:
                        out.append(f"{k} stale ({age:.1f} s)")
                elif k not in self.defaults:
                    out.append(f"{k} missing")
        return out

    def camera(self, now: float) -> CameraGeo | None:
        """Camera model from the latest fresh values, or None if anything required is missing or stale."""
        if self.problems(now):
            return None
        with self._lock:
            v = {k: self._values.get(k, self.defaults.get(k)) for k in REQUIRED}
        return CameraGeo(v["lat"], v["lon"], v["alt_m"], self.hfov_deg, v["heading_deg"], v["pitch_deg"])
