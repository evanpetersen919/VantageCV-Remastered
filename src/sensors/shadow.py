"""Beams hidden behind an inserted object (docs/lidar/insertion_preregistration.md, section 2).

When a synthetic object is pasted into a real scan, a real sensor would have returned the object
instead of whatever lies behind it along the same beams. OpenPCDet's database sampler removes only
the points *inside* the pasted box. ``visible_mask`` also removes the scene points in the box's
shadow: those whose direction (azimuth, elevation from the sensor at the origin of the LiDAR frame,
x forward, y left, z up) falls inside the convex hull of the box's projected corners and whose range
is not less than the box's nearest corner minus ``margin_m``.

The hull of the projected corners is an upper bound of the object's silhouette, so a little
background around its edges can be removed too. Points hidden by a nearer *real* surface are not
handled here.
"""

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module


def _directions(xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Azimuth and elevation (degrees) and range (metres) of each ``x y z`` row."""
    xyz = xyz.astype(np.float64)
    horizontal = np.hypot(xyz[:, 0], xyz[:, 1])
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    elevation = np.degrees(np.arctan2(xyz[:, 2], horizontal))
    return np.column_stack([azimuth, elevation, np.linalg.norm(xyz, axis=1)])


def visible_mask(
    points_xyz: npt.NDArray[np.float64],
    corners: npt.NDArray[np.float64],
    margin_m: float = 0.3,
) -> npt.NDArray[np.bool_]:
    """``True`` for scene points that stay; ``False`` for those in a pasted box's shadow.

    ``points_xyz`` is ``(N, 3)`` (extra columns are ignored by passing only ``[:, :3]``) and
    ``corners`` is ``(B, 8, 3)``, the corners of the pasted boxes. A box with a corner behind the
    sensor (x <= 0) is skipped.
    """
    keep = np.ones(points_xyz.shape[0], dtype=bool)
    if points_xyz.shape[0] == 0 or corners.shape[0] == 0:
        return keep
    scene = _directions(points_xyz)
    forward = points_xyz[:, 0] > 0.0
    for box in corners:
        if (box[:, 0] <= 0.0).any():
            continue
        projected = _directions(box)
        low, high = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
        candidate = (
            keep
            & forward
            & (scene[:, 0] >= low[0])
            & (scene[:, 0] <= high[0])
            & (scene[:, 1] >= low[1])
            & (scene[:, 1] <= high[1])
            & (scene[:, 2] >= projected[:, 2].min() - margin_m)
        )
        if not candidate.any():
            continue
        try:
            hull = ConvexHull(projected[:, :2])
        except RuntimeError:  # degenerate projection (QhullError is a RuntimeError)
            continue
        inside = (
            scene[candidate, :2] @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9
        ).all(axis=1)
        hidden = np.flatnonzero(candidate)[inside]
        keep[hidden] = False
    return keep
