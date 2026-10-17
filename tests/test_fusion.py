"""Alerts from RGB + thermal cameras fused per person (esa.fusion)."""

from esa.fusion import AlertFusion, alert_uid

LAT, LON = 61.4900, 23.7350
M = 1 / 111_320.0  # one metre of latitude, in degrees


def test_ids_are_camera_scoped():
    f = AlertFusion()
    a, new_a = f.add("rgb", 5, LAT, LON, 0.0)
    b, new_b = f.add("thermal", 5, LAT + 100 * M, LON, 1.0)  # same track number, different person
    assert new_a and new_b and a.uid == "rgb-5" and b.uid == "thermal-5"
    assert alert_uid("thermal", 5) == "thermal-5"


def test_same_person_on_both_cameras_is_one_alert():
    f = AlertFusion(radius_m=5.0)
    a, _ = f.add("rgb", 3, LAT, LON, 0.0)
    b, new = f.add("thermal", 9, LAT + 3 * M, LON, 2.0)  # 3 m away, 2 s later
    assert not new and b is a and a.cameras == {"rgb", "thermal"} and a.members == {"rgb-3", "thermal-9"}
    again, new = f.add("thermal", 9, LAT, LON, 3.0)  # the thermal track alerting again
    assert again is a and not new


def test_not_merged_when_far_late_same_camera_or_unlocated():
    f = AlertFusion(radius_m=5.0, window_s=30.0)
    f.add("rgb", 1, LAT, LON, 0.0)
    assert f.add("thermal", 2, LAT + 8 * M, LON, 1.0)[1]  # 8 m away: two people
    assert f.add("thermal", 3, LAT, LON, 40.0)[1]  # same spot but 40 s later
    g = AlertFusion()
    g.add("rgb", 1, LAT, LON, 0.0)
    assert g.add("rgb", 2, LAT, LON, 1.0)[1]  # same camera: the detector's own de-duplication decides
    assert g.add("thermal", 7, None, None, 1.0)[1]  # no position (camera above the horizon): cannot match


def test_memory_is_bounded():
    f = AlertFusion(max_alerts=10)
    for i in range(25):
        f.add("rgb", i, LAT + i * 100 * M, LON, float(i))
    assert len(f.alerts) == 10 and "rgb-0" not in f.member_of and "rgb-24" in f.member_of
