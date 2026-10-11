"""KITTI-format labels and calibration for the depth-derived LiDAR scans (preregistration, stage 2).

A scan is sampled from the engine's depth at the camera's position (``src/sensors/depth_lidar.py``),
so the LiDAR and camera share one origin. The labels follow KITTI's camera-frame convention
(``src/export/box3d_formats.py``) and are projected through KITTI's own measured front-camera
intrinsics (721.5377 px focal length, principal point (609.5593, 172.854), 1242 x 375: the P2 of
the downloaded calibration files), so a box is kept, truncated or dropped exactly as KITTI's own
image would treat it. Objects outside that 81.6 x 29.2 degree view are not labelled, as in KITTI.

``Cyclist`` is KITTI's single box around a rider and their bicycle; this generator labels the two
separately, so a rider is paired with the bike whose 3D box lies within ``PAIR_DISTANCE_M`` of it
(on the ground plane) and the pair becomes one box with the bike's footprint and orientation and
the height from the wheel contact to the rider's head. A rider with no bike is labelled by their
own box; a bike without a rider is not labelled (parked bikes are not KITTI objects).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt

from src.export.box3d_formats import CameraBox3D, calibration_text, kitti_label_line

KITTI_FX = 721.5377
KITTI_CX = 609.5593
KITTI_CY = 172.854
KITTI_WIDTH, KITTI_HEIGHT = 1242, 375
MIN_DEPTH_M = 0.1
PAIR_DISTANCE_M = 1.0
MIN_VISIBLE_AREA_SHARE = 0.1
MIN_BOX_HEIGHT_PX = 5.0

# category ids of the riders profile (src/ground_truth/categories.py RIDER_PROFILE) -> KITTI type
TYPE_BY_CATEGORY = {1: "Pedestrian", 3: "Car", 6: "Misc", 8: "Truck"}
RIDER_ID, BIKE_ID = 10, 2

# velodyne x forward, y left, z up -> camera x right, y down, z forward
TR_VELO_TO_CAM = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


def kitti_projection() -> npt.NDArray[np.float64]:
    """The 3 x 4 projection of KITTI's front camera (no stereo offset: the sensor is the camera)."""
    return np.array(
        [[KITTI_FX, 0.0, KITTI_CX, 0.0], [0.0, KITTI_FX, KITTI_CY, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )


def calibration() -> str:
    """The text of a KITTI ``calib`` file for the synthetic scans.

    ``Tr_velo_to_cam`` maps the scan's Velodyne frame (x forward, y left, z up, origin at the
    camera) to the camera frame; ``Tr_imu_to_velo`` is an identity placeholder (no IMU).
    """
    base = calibration_text(kitti_projection()).split("\n")
    row = " ".join(f"{v:.12e}" for v in TR_VELO_TO_CAM.reshape(-1))
    return "\n".join(
        line if not line.startswith("Tr_velo_to_cam") else f"Tr_velo_to_cam: {row}" for line in base
    )


def box_corners(box: Dict[str, Any]) -> npt.NDArray[np.float64]:
    """The 8 corners of a KITTI-convention box (``location`` is the bottom centre), (8, 3)."""
    height, width, length = (float(v) for v in box["dimensions_hwl"])
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * length / 2.0
    y = np.array([0, 0, 0, 0, -1, -1, -1, -1]) * height
    z = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * width / 2.0
    ry = float(box["rotation_y"])
    cos, sin = math.cos(ry), math.sin(ry)
    rotated = np.stack([cos * x + sin * z, y, -sin * x + cos * z], axis=1)
    return np.asarray(rotated + np.array(box["location"], dtype=np.float64), dtype=np.float64)


def project_box(
    box: Dict[str, Any],
) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
    """(clipped 2D box ``x1 y1 x2 y2``, truncation) in KITTI's image, or ``None`` if not kept.

    Not kept: a corner behind the camera plane, nothing inside the image, or less than 10% of
    the projected area (or 5 px of height) inside it.
    """
    corners = box_corners(box)
    if corners[:, 2].min() < MIN_DEPTH_M:
        return None
    u = KITTI_FX * corners[:, 0] / corners[:, 2] + KITTI_CX
    v = KITTI_FX * corners[:, 1] / corners[:, 2] + KITTI_CY
    x1, x2, y1, y2 = float(u.min()), float(u.max()), float(v.min()), float(v.max())
    full = max(x2 - x1, 1e-9) * max(y2 - y1, 1e-9)
    cx1, cx2 = max(x1, 0.0), min(x2, float(KITTI_WIDTH))
    cy1, cy2 = max(y1, 0.0), min(y2, float(KITTI_HEIGHT))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    share = (cx2 - cx1) * (cy2 - cy1) / full
    if share < MIN_VISIBLE_AREA_SHARE or cy2 - cy1 < MIN_BOX_HEIGHT_PX:
        return None
    return (cx1, cy1, cx2, cy2), 1.0 - share


def points_in_box(
    points: npt.NDArray[np.float64], box: Dict[str, Any], margin: float
) -> npt.NDArray[np.bool_]:
    """Which label-frame ``points`` lie inside the KITTI-convention ``box`` grown by ``margin``.

    ``box`` holds ``location`` (bottom centre), ``dimensions_hwl`` and ``rotation_y``; the length
    axis is the box's x before rotation about the down axis and the width axis its z.
    """
    height, width, length = (float(v) for v in box["dimensions_hwl"])
    centre = np.array(box["location"], dtype=np.float64) - np.array([0.0, height / 2.0, 0.0])
    ry = float(box["rotation_y"])
    cos, sin = np.cos(ry), np.sin(ry)
    rotation = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    local = (points - centre) @ rotation  # rows: R^T (p - c)
    half = np.array([length / 2.0, height / 2.0, width / 2.0]) + margin
    return np.asarray((np.abs(local) <= half).all(axis=1))


def _alpha(box: Dict[str, Any]) -> float:
    x, _, z = box["location"]
    angle = float(box["rotation_y"]) - math.atan2(x, z)
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def cyclist_box(rider: Dict[str, Any], bike: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """KITTI's ``Cyclist`` box from a rider's 3D box and, if there is one, their bike's.

    With a bike: the bike's footprint, orientation and bottom, with the height up to the higher
    of the two tops. Without: the rider's own box.
    """
    if bike is None:
        return dict(rider)
    top = min(
        rider["location"][1] - rider["dimensions_hwl"][0],
        bike["location"][1] - bike["dimensions_hwl"][0],
    )
    bottom = max(rider["location"][1], bike["location"][1])
    return {
        "location": [bike["location"][0], bottom, bike["location"][2]],
        "dimensions_hwl": [bottom - top, bike["dimensions_hwl"][1], bike["dimensions_hwl"][2]],
        "rotation_y": bike["rotation_y"],
    }


def _ground_distance(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    a, b = first["location"], second["location"]
    return math.hypot(a[0] - b[0], a[2] - b[2])


def frame_labels(  # pylint: disable=too-many-locals
    annotations: Sequence[Dict[str, Any]]
) -> List[str]:
    """The ``label_2`` lines of one frame from its annotations (riders profile ids).

    Each annotation needs ``box3d``, ``category_id``, ``visibility_fraction`` and ``bbox``.
    """
    lines: List[str] = []
    items = [a for a in annotations if a.get("box3d") is not None]
    bikes = [a for a in items if a["category_id"] == BIKE_ID]
    used_bikes: Set[int] = set()
    entries: List[Tuple[str, Dict[str, Any], float]] = []
    for annotation in items:
        category = annotation["category_id"]
        if category in TYPE_BY_CATEGORY:
            entries.append(
                (
                    TYPE_BY_CATEGORY[category],
                    annotation["box3d"],
                    float(annotation["visibility_fraction"]),
                )
            )
        elif category == RIDER_ID:
            partner = None
            for index, bike in enumerate(bikes):
                if index in used_bikes:
                    continue
                if _ground_distance(annotation["box3d"], bike["box3d"]) <= PAIR_DISTANCE_M:
                    partner = index
                    break
            if partner is not None:
                used_bikes.add(partner)
            bike_box = bikes[partner]["box3d"] if partner is not None else None
            entries.append(
                (
                    "Cyclist",
                    cyclist_box(annotation["box3d"], bike_box),
                    float(annotation["visibility_fraction"]),
                )
            )
    for kitti_type, box, visibility in entries:
        projected = project_box(box)
        if projected is None:
            continue
        (x1, y1, x2, y2), truncation = projected
        camera_box = CameraBox3D(
            tuple(box["location"]),
            tuple(box["dimensions_hwl"]),
            float(box["rotation_y"]),
            _alpha(box),
        )
        lines.append(
            kitti_label_line(
                kitti_type, truncation, visibility, (x1, y1, x2 - x1, y2 - y1), camera_box
            )
        )
    return lines
