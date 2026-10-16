"""Small unit tests for the parts that are pure logic. Run: pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from _common import yolo_row  # noqa: E402

from esa.alerts import AlertManager  # noqa: E402
from esa.geo import CameraGeo, pixel_to_latlon  # noqa: E402


def test_centre_pixel_is_drone_position():
    cam = CameraGeo(60.0, 25.0, 100, 90)
    assert pixel_to_latlon(500, 400, 1000, 800, cam) == pytest.approx((60.0, 25.0))


def test_top_of_image_is_north_when_heading_zero():
    cam = CameraGeo(60.0, 25.0, 100, 90)  # 90 deg FOV at 100 m -> 200 m wide footprint
    lat, lon = pixel_to_latlon(500, 0, 1000, 800, cam)  # top edge = 80 m north
    assert lat > 60.0 and lon == pytest.approx(25.0)
    assert (lat - 60.0) * 111_320 == pytest.approx(80, rel=1e-3)


def test_heading_90_rotates_top_to_east():
    cam = CameraGeo(60.0, 25.0, 100, 90, heading_deg=90)
    lat, lon = pixel_to_latlon(500, 0, 1000, 800, cam)
    assert lat == pytest.approx(60.0) and lon > 25.0


def test_yolo_row_normalises_and_clips():
    assert yolo_row(2, 0, 0, 100, 50, 200, 100) == "2 0.250000 0.250000 0.500000 0.500000"
    assert yolo_row(0, -10, 0, 20, 10, 100, 100).startswith("0 0.050000")  # clipped at left edge
    assert yolo_row(0, 150, 0, 10, 10, 100, 100) is None  # outside image


def test_alert_fires_once_after_persistence():
    am = AlertManager({"swimmer"}, min_frames=3)
    d = [{"track_id": 7, "label": "swimmer"}, {"track_id": 8, "label": "boat"}]
    assert am.update(d) == [] and am.update(d) == []
    fired = am.update(d)
    assert len(fired) == 1 and fired[0]["track_id"] == 7
    assert am.update(d) == []  # no repeat


def test_overall_ap_works_with_large_max_dets(tmp_path):
    """Regression test: pycocotools' summary returns AP=-1 when maxDets doesn't contain 100.
    Scoring ground truth against itself must give AP=1.0 at maxDets=500."""
    import json

    from esa.coco_utils import coco_evaluate

    gt = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": i + 1, "image_id": 1, "category_id": 1, "bbox": [i * 9, i * 9, 8, 8], "area": 64, "iscrowd": 0}
            for i in range(10)
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    preds = [{"image_id": 1, "category_id": 1, "bbox": a["bbox"], "score": 1.0} for a in gt["annotations"]]
    m = coco_evaluate(p, preds, max_dets=500)
    assert m["AP"] == pytest.approx(1.0) and m["AP_small"] == pytest.approx(1.0)


def test_iou_distance_matches_hand_calculation():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from tracking_eval import iou_distance

    d = iou_distance([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 10, 10], [50, 50, 5, 5]])
    assert d[0, 0] == pytest.approx(0.0)
    assert np.isnan(d[0, 1])  # IoU 1/3 < 0.5 -> not allowed to match
    assert np.isnan(d[0, 2])
    assert iou_distance([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_tilt_zero_equals_nadir_model():
    """With no tilt, the full ray-intersection model must agree with the pipeline's nadir model."""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from geo_sensitivity import tilted_ground_point

    cam = CameraGeo(60.0, 25.0, 60, 70)
    for px, py in [(0, 0), (480, 270), (959, 539), (100, 400)]:
        assert tilted_ground_point(px, py, 960, 540, cam, 0.0) == pytest.approx(
            pixel_to_latlon(px, py, 960, 540, cam), abs=1e-9
        )


def test_tilt_shifts_image_centre_by_alt_tan_angle():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from geo_sensitivity import tilted_ground_point

    cam = CameraGeo(60.0, 25.0, 60, 70)
    lat, _ = tilted_ground_point(480, 270, 960, 540, cam, 10.0)
    assert (lat - 60.0) * 111_320 == pytest.approx(60 * np.tan(np.radians(10)), rel=1e-6)


def test_pitch_90_is_the_nadir_model():
    """The general ray model must reproduce the original straight-down formula exactly at pitch 90."""
    import math

    cam = CameraGeo(60.0, 25.0, 80, 70, heading_deg=33.0)
    m_per_px = cam.ground_width_m() / 1000
    for px, py in [(0, 0), (500, 400), (999, 799), (120, 650)]:
        dx, dy = (px - 500) * m_per_px, (400 - py) * m_per_px
        h = math.radians(33.0)
        east, north = dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h)
        lat, lon = pixel_to_latlon(px, py, 1000, 800, cam)
        assert (lat - 60.0) * 111_320 == pytest.approx(north, abs=1e-6)
        assert (lon - 25.0) * 111_320 * math.cos(math.radians(60.0)) == pytest.approx(east, abs=1e-6)


def test_tilted_camera_centre_lands_alt_over_tan_pitch_ahead():
    import math

    cam = CameraGeo(60.0, 25.0, 50, 70, heading_deg=90.0, pitch_deg=30.0)  # facing east, 30 deg below horizon
    lat, lon = pixel_to_latlon(480, 270, 960, 540, cam)
    east = (lon - 25.0) * 111_320 * math.cos(math.radians(60.0))
    assert east == pytest.approx(50 / math.tan(math.radians(30)), rel=1e-6)
    assert lat == pytest.approx(60.0, abs=1e-9)


def test_pixel_above_horizon_has_no_position():
    cam = CameraGeo(60.0, 25.0, 50, 70, pitch_deg=10.0)  # 10 deg below horizon; top of image looks above it
    assert pixel_to_latlon(480, 0, 960, 540, cam) is None


def test_independent_tilt_implementation_agrees():
    """benchmarks/geo_sensitivity.py has its own ray code; both must agree (tilt t from nadir = pitch 90 - t)."""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from geo_sensitivity import tilted_ground_point

    for tilt in (5, 20, 45):
        cam = CameraGeo(60.0, 25.0, 60, 70)
        tilted = CameraGeo(60.0, 25.0, 60, 70, pitch_deg=90 - tilt)
        for px, py in [(100, 100), (480, 270), (900, 500)]:
            assert pixel_to_latlon(px, py, 960, 540, tilted) == pytest.approx(
                tilted_ground_point(px, py, 960, 540, cam, tilt), abs=1e-9
            )


# --- live telemetry (ROS 2 node) ---------------------------------------------------------------
from esa.telemetry import TelemetryState  # noqa: E402


def test_telemetry_needs_position_and_altitude():
    t = TelemetryState(hfov_deg=70.0, defaults={"heading_deg": 0.0, "pitch_deg": 90.0})
    assert t.camera(0.0) is None
    t.update(0.0, lat=61.5, lon=23.76)
    assert "alt_m missing" in t.problems(0.1)
    t.update(0.1, alt_m=40.0)
    cam = t.camera(0.2)
    assert (cam.lat, cam.lon, cam.alt_m, cam.pitch_deg) == (61.5, 23.76, 40.0, 90.0)


def test_stale_telemetry_gives_no_camera():
    t = TelemetryState(hfov_deg=70.0, timeout_s=1.0, defaults={"heading_deg": 0.0, "pitch_deg": 90.0})
    t.update(0.0, lat=61.5, lon=23.76, alt_m=40.0)
    assert t.camera(0.9) is not None
    assert t.camera(1.5) is None  # better no position than a stale one
    assert any("stale" in p for p in t.problems(1.5))


def test_gimbal_topic_overrides_default_and_then_must_stay_fresh():
    t = TelemetryState(hfov_deg=70.0, timeout_s=1.0, defaults={"heading_deg": 0.0, "pitch_deg": 90.0})
    t.update(0.0, lat=61.5, lon=23.76, alt_m=40.0, pitch_deg=45.0)
    assert t.camera(0.5).pitch_deg == 45.0
    t.update(2.0, lat=61.5, lon=23.76, alt_m=40.0)  # gimbal topic went silent
    assert t.camera(2.1) is None


def test_unknown_telemetry_field_is_rejected():
    with pytest.raises(KeyError):
        TelemetryState(hfov_deg=70.0).update(0.0, yaw=1.0)


# --- real-footage geolocation helpers (benchmarks/geo_real.py) ---------------------------------
sys.path.insert(0, str(ROOT / "benchmarks"))
import geo_real  # noqa: E402


def test_metres_between_points_one_arcsecond_apart():
    assert geo_real.metres((60.0, 25.0), (60.0 + 1 / 3600, 25.0)) == pytest.approx(30.92, abs=0.01)


def test_smoothing_turns_sample_and_hold_into_a_ramp_and_unwraps_heading():
    tel = [{"lat": 0.0, "lon": 0.0, "alt_m": 10.0, "heading_deg": h, "pitch_deg": 90.0} for h in (358, 358, 2, 2)]
    tel[2]["alt_m"] = tel[3]["alt_m"] = 13.0
    out = geo_real.smooth_telemetry(tel)
    assert [o["alt_m"] for o in out] == pytest.approx([10.0, 11.5, 13.0, 13.0])
    # 358 -> 2 is a 4 deg turn through north, not a 356 deg turn the other way
    assert out[1]["heading_deg"] == pytest.approx(0.0, abs=1e-9) or out[1]["heading_deg"] == pytest.approx(360.0)


# --- search planning (src/esa/search.py) -------------------------------------------------------
import json as _json  # noqa: E402
import math as _math  # noqa: E402

from esa.search import (  # noqa: E402
    Camera,
    DetectionCurve,
    evaluate,
    lawnmower,
    p_pass,
    path_length_m,
    qgc_plan,
    spacing_for_pod,
)


def test_detection_curve_is_isotonic_not_optimistic():
    # a noisy dip must not make bigger objects worse, and must not be lifted to the best bin either
    c = DetectionCurve.from_bins([5, 10, 20], [0.2, 0.6, 0.4], [100, 100, 300])
    assert c(0) == 0.0 and c(5) == pytest.approx(0.2)
    assert c(10) == pytest.approx(c(20)) == pytest.approx((0.6 * 100 + 0.4 * 300) / 400)
    assert c(100) == pytest.approx(c(20))
    assert not c.measured(1) or c.min_px == 0


def test_pixels_on_target_halves_when_altitude_doubles():
    cam = Camera(hfov_deg=90.0, input_px=1000)  # footprint = 2 * alt
    assert cam.footprint_m(50)[0] == pytest.approx(100.0)
    assert cam.pixels_on(0.5, 50) == pytest.approx(5.0)
    assert cam.pixels_on(0.5, 100) == pytest.approx(2.5)


def test_looks_are_capped_by_independence():
    assert p_pass(0.5, looks=10, independent_looks=1) == pytest.approx(0.5)
    assert p_pass(0.5, looks=10, independent_looks=2) == pytest.approx(0.75)
    assert p_pass(0.5, looks=0.3, independent_looks=5) == pytest.approx(0.5)  # at least one look


def test_sweep_width_is_footprint_times_pass_probability():
    cam = Camera(hfov_deg=90.0, input_px=1000, aspect=1.0)
    p = evaluate(DetectionCurve([5.0], [0.4]), cam, alt_m=50, speed_mps=10, fps=1, swimmer_m=0.5, independent_looks=1)
    assert p.sweep_width_m == pytest.approx(100 * 0.4)
    assert p.sweep_rate_km2_h == pytest.approx(40 * 10 * 3.6 / 1000)


def test_spacing_for_pod_inverts_the_exponential_law():
    s = spacing_for_pod(30.0, 0.8)
    assert 1 - _math.exp(-30.0 / s) == pytest.approx(0.8)


def test_lawnmower_covers_the_box_and_alternates_direction():
    pts = lawnmower(61.5, 23.7, width_m=100, length_m=400, spacing_m=25, heading_deg=0)
    assert len(pts) == 8  # 4 tracks x 2 ends
    assert pts[1][0] > pts[0][0] and pts[3][0] < pts[2][0]  # north, then back south
    assert path_length_m(pts) == pytest.approx(4 * 400 + 3 * 25, rel=1e-3)


def test_qgc_plan_takes_off_searches_returns_and_is_fenced():
    pts = lawnmower(61.5, 23.7, 50, 100, 25)
    plan = _json.loads(_json.dumps(qgc_plan(pts, 30.0, 8.0, fence_margin_m=50)))
    items = plan["mission"]["items"]
    assert plan["fileType"] == "Plan"
    assert [i["command"] for i in items] == [22, 178] + [16] * len(pts) + [20]  # takeoff, speed, search, RTL
    assert items[2]["params"][4:] == [pts[0][0], pts[0][1], 30.0]
    fence = plan["geoFence"]["polygons"][0]
    assert fence["inclusion"] and len(fence["polygon"]) == 4
    s_, w_ = fence["polygon"][0]
    n_, e_ = fence["polygon"][2]
    assert all(s_ < la < n_ and w_ < lo < e_ for la, lo in pts)  # every waypoint inside the fence


# --- TAK / Cursor on Target and the operator loop ------------------------------------------------
import socket as _socket  # noqa: E402
import xml.etree.ElementTree as _ET  # noqa: E402
from datetime import datetime as _dt  # noqa: E402
from datetime import timezone as _tz  # noqa: E402

from esa.cot import CotSender, alert_event, delete_event, drone_event  # noqa: E402

ALERT = {
    "track_id": 7,
    "label": "swimmer",
    "lat": 61.4978123,
    "lon": 23.7610456,
    "message": "swimmer detected (track 7)",
}


def test_alert_cot_has_required_fields():
    now = _dt(2026, 10, 15, 12, 0, 0, tzinfo=_tz.utc)
    ev = _ET.fromstring(alert_event(ALERT, "drone-A", ce_m=12.0, now=now, stale_s=60))
    assert ev.tag == "event" and ev.get("version") == "2.0"
    assert ev.get("type") == "a-u-S" and ev.get("uid") == "drone-A-swimmer-7"
    assert ev.get("time") == "2026-10-15T12:00:00.000Z" and ev.get("stale") == "2026-10-15T12:01:00.000Z"
    pt = ev.find("point")
    assert float(pt.get("lat")) == pytest.approx(ALERT["lat"]) and float(pt.get("ce")) == 12.0
    assert "unconfirmed" in ev.find("detail/remarks").text
    assert ev.find("detail/link").get("uid") == "drone-A"


def test_confirm_and_delete_events():
    ok = _ET.fromstring(alert_event(ALERT, "drone-A", confirmed=True))
    assert "CONFIRMED" in ok.find("detail/remarks").text
    de = _ET.fromstring(delete_event("drone-A-swimmer-7"))
    assert de.get("type") == "t-x-d-d" and de.find("detail/link").get("uid") == "drone-A-swimmer-7"
    assert de.find("detail/__forcedelete") is not None
    dr = _ET.fromstring(drone_event("drone-A", 61.5, 23.7, alt_m=40.0))
    assert dr.get("type") == "a-f-A-M-H-Q" and dr.find("point").get("hae") == "40.0"


def test_cot_over_udp_arrives_intact():
    rx = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    tx = CotSender(f"udp://127.0.0.1:{rx.getsockname()[1]}")
    xml = alert_event(ALERT)
    tx.send(xml)
    assert rx.recv(65535).decode() == xml
    tx.close()
    rx.close()


def test_land_alerts_are_ground_markers():
    land = {**ALERT, "label": "person"}
    assert _ET.fromstring(alert_event(land)).get("type") == "a-u-G"
    assert _ET.fromstring(alert_event(ALERT)).get("type") == "a-u-S"


def test_bad_tak_url_is_rejected():
    with pytest.raises(ValueError):
        CotSender("http://example.com")


def test_rejected_spot_suppresses_new_alerts_nearby_only():
    am = AlertManager({"swimmer"}, min_frames=1, suppress_radius_m=10, suppress_s=600)
    near = {"track_id": 2, "label": "swimmer", "lat": 61.0, "lon": 23.0}
    am.reject(61.0, 23.0, "swimmer", now=0)
    assert am.update([near], now=10) == [] and am.suppressed == 1
    far = {"track_id": 3, "label": "swimmer", "lat": 61.001, "lon": 23.0}  # ~111 m away
    assert len(am.update([far], now=10)) == 1
    later = {"track_id": 4, "label": "swimmer", "lat": 61.0, "lon": 23.0}
    assert len(am.update([later], now=700)) == 1  # rejection has expired


def test_dedupe_merges_a_new_track_id_at_the_same_spot_only():
    am = AlertManager({"swimmer"}, min_frames=1, dedupe_radius_m=3, dedupe_s=60)
    assert len(am.update([{"track_id": 1, "label": "swimmer", "lat": 61.0, "lon": 23.0}], now=0)) == 1
    same = {"track_id": 9, "label": "swimmer", "lat": 61.00001, "lon": 23.0}  # ~1.1 m away: same swimmer, new ID
    assert am.update([same], now=5) == [] and am.deduplicated == 1
    other = {"track_id": 10, "label": "swimmer", "lat": 61.0001, "lon": 23.0}  # ~11 m away: another swimmer
    assert len(am.update([other], now=5)) == 1
    again = {"track_id": 11, "label": "swimmer", "lat": 61.0, "lon": 23.0}
    assert len(am.update([again], now=100)) == 1  # older than dedupe_s: raise again
    off = AlertManager({"swimmer"}, min_frames=1)  # default: off, as before
    off.update([{"track_id": 1, "label": "swimmer", "lat": 61.0, "lon": 23.0}], now=0)
    assert len(off.update([same], now=1)) == 1
