"""Cursor on Target (CoT): send alerts and the drone's position to TAK (ATAK, WinTAK, iTAK, TAK Server).

TAK is the map that many rescue, police, fire and defence teams already carry on phones and tablets. Speaking
its protocol means a detection appears on the map the crew already uses, instead of in a separate app.

CoT is a small XML message: who/what (uid, type), where (point lat/lon + circular error), when (time, start,
stale), plus free-form <detail>. Types follow MIL-STD-2525 letters: "a-u-S" = atom, unknown affiliation, Sea
surface (a person or object in the water, not yet identified); "a-u-G" = unknown, Ground (a person or vehicle on land);
"a-f-A-M-H-Q" = friendly, air, rotary-wing drone. The sea / ground choice follows the detected label.
A "t-x-d-d" event deletes a marker (used when the operator rejects an alert).

Transports: UDP (default ATAK mesh multicast udp://239.2.3.1:6969, or unicast) and plain TCP to a TAK server
(tcp://host:8087). TLS (port 8089, client certificates) is not implemented.
"""

from __future__ import annotations

import socket
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

UNKNOWN = "9999999.0"  # CoT convention for "not known" height / linear error
SEA_LABELS = {"swimmer", "boat", "jetski", "life_saving_appliances", "buoy"}


def cot_type_for(label: str) -> str:
    """Unknown object on the sea surface for maritime classes, on the ground for everything else."""
    return "a-u-S" if label in SEA_LABELS else "a-u-G"


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def event(
    uid: str,
    cot_type: str,
    lat: float,
    lon: float,
    *,
    ce_m: float = UNKNOWN,
    hae_m: float | str = UNKNOWN,
    callsign: str | None = None,
    remarks: str | None = None,
    stale_s: float = 120.0,
    how: str = "m-g",  # machine-generated
    now: datetime | None = None,
    extra: dict[str, dict[str, str]] | None = None,
) -> str:
    """One CoT event as an XML string."""
    now = now or datetime.now(timezone.utc)
    ev = ET.Element(
        "event",
        version="2.0",
        uid=uid,
        type=cot_type,
        how=how,
        time=_iso(now),
        start=_iso(now),
        stale=_iso(now + timedelta(seconds=stale_s)),
    )
    ET.SubElement(ev, "point", lat=f"{lat:.7f}", lon=f"{lon:.7f}", hae=str(hae_m), ce=str(ce_m), le=UNKNOWN)
    detail = ET.SubElement(ev, "detail")
    if callsign:
        ET.SubElement(detail, "contact", callsign=callsign)
    if remarks:
        ET.SubElement(detail, "remarks").text = remarks
    for tag, attrs in (extra or {}).items():
        ET.SubElement(detail, tag, **attrs)
    return ET.tostring(ev, encoding="unicode")


def alert_event(alert: dict, drone_uid: str = "esa-drone-1", ce_m: float = 15.0, confirmed: bool = False, **kw) -> str:
    """CoT for an alert from esa.alerts.AlertManager (needs lat/lon). ce_m: position uncertainty radius."""
    status = "CONFIRMED by operator" if confirmed else "unconfirmed, awaiting operator review"
    return event(
        uid=f"{drone_uid}-{alert['label']}-{alert['track_id']}",
        cot_type=cot_type_for(alert["label"]),
        lat=alert["lat"],
        lon=alert["lon"],
        ce_m=ce_m,
        callsign=f"{alert['label'].capitalize()} {alert['track_id']}{' ✓' if confirmed else ' ?'}",
        remarks=f"{alert['message']} by drone {drone_uid}; {status}. Automatic detection, verify before acting.",
        extra={"link": {"uid": drone_uid, "relation": "p-p", "type": "a-f-A-M-H-Q"}},
        **kw,
    )


def drone_event(drone_uid: str, lat: float, lon: float, alt_m: float | None = None, **kw) -> str:
    return event(
        uid=drone_uid,
        cot_type="a-f-A-M-H-Q",
        lat=lat,
        lon=lon,
        hae_m=f"{alt_m:.1f}" if alt_m is not None else UNKNOWN,
        callsign=drone_uid,
        stale_s=kw.pop("stale_s", 10.0),
        **kw,
    )


def delete_event(target_uid: str, lat: float = 0.0, lon: float = 0.0, **kw) -> str:
    """Ask TAK clients to remove a marker (operator rejected the alert)."""
    return event(
        uid=f"{target_uid}-delete",
        cot_type="t-x-d-d",
        lat=lat,
        lon=lon,
        extra={"link": {"uid": target_uid, "relation": "none", "type": "none"}, "__forcedelete": {}},
        stale_s=kw.pop("stale_s", 20.0),
        **kw,
    )


class CotSender:
    """Send CoT XML to udp://host:port (unicast or multicast) or tcp://host:port (TAK server, no TLS)."""

    def __init__(self, url: str):
        u = urlparse(url)
        if u.scheme not in ("udp", "tcp") or not u.hostname or not u.port:
            raise ValueError(f"expected udp://host:port or tcp://host:port, got {url!r}")
        self.scheme, self.addr = u.scheme, (u.hostname, u.port)
        self.sock = None
        self._connect()

    def _connect(self):
        if self.scheme == "udp":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)  # stay on the local network
        else:
            self.sock = socket.create_connection(self.addr, timeout=5)

    def send(self, xml: str) -> None:
        data = xml.encode()
        if self.scheme == "udp":
            self.sock.sendto(data, self.addr)
            return
        try:
            self.sock.sendall(data)
        except OSError:  # TAK server restarted: reconnect once
            self._connect()
            self.sock.sendall(data)

    def close(self):
        if self.sock:
            self.sock.close()
