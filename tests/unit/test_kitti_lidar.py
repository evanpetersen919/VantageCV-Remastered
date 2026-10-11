"""Tests for KITTI labels and calibration of the synthetic LiDAR scans (kitti_lidar module)."""

import math
from typing import Any, Dict, Tuple

import numpy as np

from src.export import kitti_lidar as kl


def _box(
    x: float,
    z: float,
    ry: float = 0.0,
    hwl: Tuple[float, float, float] = (1.5, 1.7, 4.0),
    y: float = 1.71,
) -> Dict[str, Any]:
    return {"location": [x, y, z], "dimensions_hwl": list(hwl), "rotation_y": ry}


def _annotation(category: int, box: Dict[str, Any], visibility: float = 1.0) -> Dict[str, Any]:
    return {"category_id": category, "box3d": box, "visibility_fraction": visibility}


def test_projection_matches_kitti_intrinsics() -> None:
    """A point 20 m ahead and 2 m right projects where KITTI's P2 puts it."""
    p = kl.kitti_projection() @ np.array([2.0, 1.71, 20.0, 1.0])
    assert p[0] / p[2] == np.float64(kl.KITTI_FX * 2.0 / 20.0 + kl.KITTI_CX)
    assert p[1] / p[2] == np.float64(kl.KITTI_FX * 1.71 / 20.0 + kl.KITTI_CY)


def test_calibration_text_has_velodyne_transform() -> None:
    """The calib file carries P2, an identity R0_rect and the velodyne-to-camera map."""
    text = kl.calibration()
    fields = {line.split(":")[0]: line.split(":")[1].split() for line in text.strip().split("\n")}
    assert float(fields["P2"][0]) == kl.KITTI_FX
    assert [float(v) for v in fields["Tr_velo_to_cam"][:4]] == [0.0, -1.0, 0.0, 0.0]
    assert [float(v) for v in fields["Tr_velo_to_cam"][8:]] == [1.0, 0.0, 0.0, 0.0]


def test_car_ahead_is_labelled_with_exact_dimensions() -> None:
    """A car 20 m ahead gives one Car line with its height, width and length and no truncation."""
    lines = kl.frame_labels([_annotation(3, _box(1.0, 20.0))])
    assert len(lines) == 1
    fields = lines[0].split()
    assert fields[0] == "Car" and float(fields[1]) == 0.0 and fields[2] == "0"
    assert [float(v) for v in fields[8:11]] == [1.5, 1.7, 4.0]
    assert [float(v) for v in fields[11:14]] == [1.0, 1.71, 20.0]


def test_object_outside_the_kitti_view_is_not_labelled() -> None:
    """KITTI's field is 81.6 degrees wide: a car 20 m ahead and 30 m to the side is outside."""
    assert not kl.frame_labels([_annotation(3, _box(30.0, 20.0))])
    assert not kl.frame_labels([_annotation(3, _box(0.0, -10.0))])


def test_object_on_the_edge_is_truncated() -> None:
    """A car straddling the right edge of the image is kept with a truncation between 0 and 1."""
    edge_x = 20.0 * (kl.KITTI_WIDTH - kl.KITTI_CX) / kl.KITTI_FX  # metres at the right edge
    lines = kl.frame_labels([_annotation(3, _box(edge_x, 20.0))])
    assert len(lines) == 1
    assert 0.2 < float(lines[0].split()[1]) < 0.8


def test_rider_and_bike_become_one_cyclist() -> None:
    """A rider and the bike beneath them are one Cyclist with the bike's footprint, rider's top."""
    rider = _box(0.0, 15.0, ry=0.5, hwl=(1.6, 0.6, 0.8), y=1.70)
    bike = _box(0.1, 15.1, ry=0.5, hwl=(1.0, 0.6, 1.8), y=1.71)
    lines = kl.frame_labels([_annotation(10, rider), _annotation(2, bike)])
    assert len(lines) == 1
    fields = lines[0].split()
    assert fields[0] == "Cyclist"
    height, width, length = (float(v) for v in fields[8:11])
    assert abs(height - (1.71 - (1.70 - 1.6))) < 0.011  # bottom of the bike to the rider's head
    assert (width, length) == (0.6, 1.8)


def test_parked_bike_is_not_labelled_and_lone_rider_is() -> None:
    """A bike with no rider is skipped; a rider with no bike is still a Cyclist (bike hidden)."""
    parked = kl.frame_labels([_annotation(2, _box(0.0, 15.0, hwl=(1.0, 0.6, 1.8)))])
    assert not parked
    lone = kl.frame_labels([_annotation(10, _box(0.0, 15.0, hwl=(1.6, 0.6, 0.8)))])
    assert len(lone) == 1 and lone[0].startswith("Cyclist")


def test_occlusion_level_follows_visibility_and_alpha_is_wrapped() -> None:
    """Visibility 0.5 is occlusion level 2; alpha lies in [-pi, pi]."""
    line = kl.frame_labels([_annotation(3, _box(2.0, 25.0, ry=3.0), visibility=0.5)])[0]
    fields = line.split()
    assert fields[2] == "2"
    assert -math.pi <= float(fields[3]) <= math.pi


def test_points_in_box_respects_rotation_and_margin() -> None:
    """Points inside a rotated box are in; a point 0.3 m outside is out until the margin grows."""
    box = _box(0.0, 20.0, ry=math.pi / 2, hwl=(1.5, 2.0, 4.0))  # length now along camera z
    inside = np.array([[0.0, 1.0, 21.9]])  # 1.9 m along the length (half length 2.0)
    outside = np.array([[0.0, 1.0, 22.3]])
    assert kl.points_in_box(inside, box, 0.0).all()
    assert not kl.points_in_box(outside, box, 0.0).any()
    assert kl.points_in_box(outside, box, 0.4).all()


def test_empty_frame_gets_one_dont_care_line() -> None:
    """OpenPCDet's info builder cannot read an empty label file; KITTI's ignore marker is used."""
    assert kl.with_dont_care([]) == [kl.DONT_CARE_LINE]
    assert kl.with_dont_care(["Car 0.00 0 0.00 1 2 3 4 1 1 1 0 0 10 0"]) == [
        "Car 0.00 0 0.00 1 2 3 4 1 1 1 0 0 10 0"
    ]
    assert len(kl.DONT_CARE_LINE.split()) == 15


def test_conventions_cap_van_and_pedestrian_footprint() -> None:
    """Range cap drops far labels, long Cars become Van, pedestrians take the KITTI footprint."""
    conventions = kl.Conventions(
        range_cap_m={"Car": 30.0, "Pedestrian": 25.0},
        van_length_m=4.53,
        pedestrian_footprint_m=(0.63, 0.82),
    )
    short_car = _annotation(3, _box(1.0, 20.0, hwl=(1.5, 1.7, 4.0)))
    long_car = _annotation(3, _box(1.0, 20.0, hwl=(1.8, 1.9, 5.0)))
    far_car = _annotation(3, _box(1.0, 40.0, hwl=(1.5, 1.7, 4.0)))
    pedestrian = _annotation(1, _box(0.5, 15.0, hwl=(1.75, 0.5, 0.7)))
    far_pedestrian = _annotation(1, _box(0.5, 28.0, hwl=(1.75, 0.5, 0.7)))
    annotations = [short_car, long_car, far_car, pedestrian, far_pedestrian]
    lines = kl.frame_labels(annotations, conventions)
    assert sorted(line.split()[0] for line in lines) == ["Car", "Pedestrian", "Van"]
    ped = next(line for line in lines if line.startswith("Pedestrian")).split()
    assert [float(v) for v in ped[8:11]] == [1.75, 0.63, 0.82]


def test_no_conventions_changes_nothing() -> None:
    """The default call is the original behaviour: the same lines as with an empty Conventions."""
    annotations = [_annotation(3, _box(1.0, 60.0, hwl=(1.8, 1.9, 5.0)))]
    assert kl.frame_labels(annotations) == kl.frame_labels(annotations, kl.Conventions())
