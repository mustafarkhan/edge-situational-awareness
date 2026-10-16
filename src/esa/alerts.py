"""Debounced alerts: only alert once a track has been seen for N frames, and learn from operator rejections.

Raw detections flicker. A single false positive on a wave crest should not page a rescue crew. Requiring
persistence across frames trades a little delay for far fewer false alarms (measured in
benchmarks/alert_load.py).

Operator loop (v2): when an operator rejects an alert (a buoy, a wave pattern, debris), the same spot tends to
fire again as soon as the tracker gives the object a new ID. `reject()` remembers the rejected position and
suppresses new alerts within `suppress_radius_m` of it for `suppress_s` seconds. Confirmed alerts are never
suppressed. Every suppression is counted, so the effect can be measured, not just assumed.

De-duplication (v2): trackers lose and re-create IDs, and every new ID of the same swimmer used to raise a new
alert (about 20 alerts in 5 s for 15 swimmers on a real clip). With `dedupe_radius_m` > 0, a new alert within that
distance of an alert of the same label from the last `dedupe_s` seconds is treated as the same object and not
raised again. benchmarks/alert_load.py measures how many duplicates this removes and whether real swimmers are lost.
"""

from __future__ import annotations

import math
import time

EARTH_M_PER_DEG_LAT = 111_320.0


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dn = (lat2 - lat1) * EARTH_M_PER_DEG_LAT
    de = (lon2 - lon1) * EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat1))
    return math.hypot(dn, de)


class AlertManager:
    def __init__(
        self,
        alert_labels: set[str],
        min_frames: int = 5,
        suppress_radius_m: float = 10.0,
        suppress_s: float = 600.0,
        dedupe_radius_m: float = 0.0,
        dedupe_s: float = 60.0,
    ):
        self.alert_labels = alert_labels
        self.min_frames = min_frames
        self.suppress_radius_m = suppress_radius_m
        self.suppress_s = suppress_s
        self._seen: dict[int, int] = {}
        self._fired: set[int] = set()
        self._rejected: list[tuple[float, float, float, str]] = []  # lat, lon, time, label
        self.suppressed = 0
        self.dedupe_radius_m = dedupe_radius_m
        self.dedupe_s = dedupe_s
        self._raised: list[tuple[float, float, float, str]] = []  # lat, lon, time, label of raised alerts
        self.deduplicated = 0

    def reject(self, lat: float | None, lon: float | None, label: str, now: float | None = None) -> None:
        """Operator says this alert was false. Suppress new alerts of this label near this position."""
        if lat is not None and lon is not None:
            self._rejected.append((lat, lon, time.time() if now is None else now, label))

    def _near_rejected(self, d: dict, now: float) -> bool:
        if d.get("lat") is None or d.get("lon") is None:
            return False
        self._rejected = [r for r in self._rejected if now - r[2] <= self.suppress_s]
        return any(
            r[3] == d["label"] and distance_m(r[0], r[1], d["lat"], d["lon"]) <= self.suppress_radius_m
            for r in self._rejected
        )

    def _near_raised(self, d: dict, now: float) -> bool:
        if self.dedupe_radius_m <= 0 or d.get("lat") is None or d.get("lon") is None:
            return False
        self._raised = [r for r in self._raised if now - r[2] <= self.dedupe_s]
        return any(
            r[3] == d["label"] and distance_m(r[0], r[1], d["lat"], d["lon"]) <= self.dedupe_radius_m
            for r in self._raised
        )

    def update(self, detections: list[dict], now: float | None = None) -> list[dict]:
        """detections: dicts with 'track_id' and 'label' (and lat/lon). Returns newly fired alerts."""
        now = time.time() if now is None else now
        new_alerts = []
        for d in detections:
            tid = d.get("track_id")
            if tid is None or d["label"] not in self.alert_labels:
                continue
            self._seen[tid] = self._seen.get(tid, 0) + 1
            if self._seen[tid] >= self.min_frames and tid not in self._fired:
                self._fired.add(tid)
                if self._near_rejected(d, now):
                    self.suppressed += 1
                    continue
                if self._near_raised(d, now):
                    self.deduplicated += 1
                    continue
                if d.get("lat") is not None and d.get("lon") is not None:
                    self._raised.append((d["lat"], d["lon"], now, d["label"]))
                new_alerts.append(
                    {
                        "track_id": tid,
                        "label": d["label"],
                        "lat": d.get("lat"),
                        "lon": d.get("lon"),
                        "message": f"{d['label']} detected (track {tid})",
                    }
                )
        return new_alerts
