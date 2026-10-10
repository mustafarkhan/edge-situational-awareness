"""Read power, temperature and GPU load from `tegrastats` on a Jetson.

Runs tegrastats in the background and stores timestamped samples, so a benchmark can
ask for the average power over exactly the window in which it was timing inference.
On a non-Jetson machine `available` is False and summaries return None.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time

_POWER = re.compile(r"VDD_IN (\d+)mW")
_TEMP = re.compile(r"(\w+)@(-?[\d.]+)C")
_GPU = re.compile(r"GR3D_FREQ (\d+)%")


class TegraStats:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.available = shutil.which("tegrastats") is not None
        self.samples: list[dict] = []
        self._proc = None
        self._thread = None

    def start(self):
        if not self.available:
            return self
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)], stdout=subprocess.PIPE, text=True
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            s = {"t": time.time()}
            if m := _POWER.search(line):
                s["power_w"] = int(m.group(1)) / 1000.0
            if m := _GPU.search(line):
                s["gpu_load"] = int(m.group(1))
            temps = {k: float(v) for k, v in _TEMP.findall(line)}
            if temps:
                s["max_temp_c"] = max(temps.values())
            self.samples.append(s)

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def summary(self, t0: float, t1: float) -> dict | None:
        win = [s for s in self.samples if t0 <= s["t"] <= t1 and "power_w" in s]
        if not win:
            return None
        return {
            "power_w_mean": sum(s["power_w"] for s in win) / len(win),
            "power_w_max": max(s["power_w"] for s in win),
            "temp_c_max": max((s.get("max_temp_c", float("nan")) for s in win), default=float("nan")),
            "gpu_load_mean": sum(s.get("gpu_load", 0) for s in win) / len(win),
            "n_samples": len(win),
        }
