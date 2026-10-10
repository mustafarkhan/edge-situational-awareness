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
    assert yolo_row(0, 150, 0, 10, 10, 100, 100) is None                   # outside image


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
    gt = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
          "annotations": [{"id": i + 1, "image_id": 1, "category_id": 1, "bbox": [i * 9, i * 9, 8, 8],
                           "area": 64, "iscrowd": 0} for i in range(10)],
          "categories": [{"id": 1, "name": "person"}]}
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
    assert np.isnan(d[0, 1])          # IoU 1/3 < 0.5 -> not allowed to match
    assert np.isnan(d[0, 2])
    assert iou_distance([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_tilt_zero_equals_nadir_model():
    """With no tilt, the full ray-intersection model must agree with the pipeline's nadir model."""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from geo_sensitivity import tilted_ground_point
    cam = CameraGeo(60.0, 25.0, 60, 70)
    for px, py in [(0, 0), (480, 270), (959, 539), (100, 400)]:
        assert tilted_ground_point(px, py, 960, 540, cam, 0.0) == pytest.approx(
            pixel_to_latlon(px, py, 960, 540, cam), abs=1e-9)


def test_tilt_shifts_image_centre_by_alt_tan_angle():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from geo_sensitivity import tilted_ground_point
    cam = CameraGeo(60.0, 25.0, 60, 70)
    lat, _ = tilted_ground_point(480, 270, 960, 540, cam, 10.0)
    assert (lat - 60.0) * 111_320 == pytest.approx(60 * np.tan(np.radians(10)), rel=1e-6)
