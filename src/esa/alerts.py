"""Debounced alerts: only alert once a track has been seen for N frames.

Raw detections flicker. A single false positive on a wave crest should not page a
rescue crew. Requiring persistence across frames trades a little delay for far fewer
false alarms -- a deliberate design trade-off.
"""

from __future__ import annotations


class AlertManager:
    def __init__(self, alert_labels: set[str], min_frames: int = 5):
        self.alert_labels = alert_labels
        self.min_frames = min_frames
        self._seen: dict[int, int] = {}
        self._fired: set[int] = set()

    def update(self, detections: list[dict]) -> list[dict]:
        """detections: dicts with 'track_id' and 'label'. Returns newly fired alerts."""
        new_alerts = []
        for d in detections:
            tid = d.get("track_id")
            if tid is None or d["label"] not in self.alert_labels:
                continue
            self._seen[tid] = self._seen.get(tid, 0) + 1
            if self._seen[tid] >= self.min_frames and tid not in self._fired:
                self._fired.add(tid)
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
