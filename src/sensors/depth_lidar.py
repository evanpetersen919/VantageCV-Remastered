"""A LiDAR scan sampled from a rendered depth image (docs/lidar/preregistration.md, section 2).

The engine's depth capture holds, for every pixel, the distance to the nearest surface along the
camera's viewing axis. A rotating LiDAR is a set of beams at fixed azimuth and elevation angles
from one point; for a sensor at the camera's position, each beam is a ray through the camera, so
it can be read from the depth image: project the beam into the image, take the depth there, and
convert it to the range along the beam. What the camera cannot see produces no return, so a scan
covers the camera's field of view only (a forward sector), not 360 degrees.

Conventions: camera frame x right, y down, z forward (OpenCV); pixel ``i`` covers ``[i, i + 1)``
and the principal point is the image centre. ``to_velodyne_frame`` converts to the KITTI Velodyne
frame (x forward, y left, z up). The sensor model is a model, not a measurement of any device:
``channels`` elevations spaced uniformly over the vertical field, one ray per azimuth step, no
dropout, no noise, no intensity (the depth carries none).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class BeamPattern:
    """Beam directions of the simulated sensor, as angles from the camera's viewing axis.

    ``channels`` elevations are spaced uniformly over ``vertical_fov_deg`` and azimuths over
    ``horizontal_fov_deg`` at ``azimuth_step_deg``; both fields are centred on the axis.
    """

    channels: int = 64
    vertical_fov_deg: float = 73.74
    horizontal_fov_deg: float = 106.0
    azimuth_step_deg: float = 0.1
    max_range_m: float = 120.0

    def elevations_deg(self) -> npt.NDArray[np.float64]:
        """Channel elevations, bottom to top."""
        half = self.vertical_fov_deg / 2.0
        return np.linspace(-half, half, self.channels)

    def azimuths_deg(self) -> npt.NDArray[np.float64]:
        """Azimuth samples, left to right."""
        half = self.horizontal_fov_deg / 2.0
        count = int(np.floor(self.horizontal_fov_deg / self.azimuth_step_deg + 1e-9)) + 1
        return np.linspace(-half, half, count)


@dataclass
class Scan:
    """The returns of one scan; arrays are parallel (one entry per point).

    ``points`` are in the camera frame, in metres. ``pixel_x`` and ``pixel_y`` are the integer
    pixel each return was read from (so class and instance maps can be read at the same place).
    """

    points: npt.NDArray[np.float32]
    ranges: npt.NDArray[np.float32]
    channel: npt.NDArray[np.int32]
    pixel_x: npt.NDArray[np.int32]
    pixel_y: npt.NDArray[np.int32]


def _read_depth(  # pylint: disable=too-many-locals
    depth_m: npt.NDArray[np.floating[Any]],
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    interpolate: bool,
    edge_ratio: float,
) -> npt.NDArray[np.float64]:
    """Depth at continuous pixel coordinates ``(u, v)`` (pixel ``i`` covers ``[i, i + 1)``).

    Nearest pixel by default. With ``interpolate`` the four pixel centres around the point are
    blended in *inverse* depth (exact across a plane, where depth itself is hyperbolic) when all
    four have a surface and their depths agree within ``edge_ratio`` of the nearest one; across an
    object edge the nearest pixel is used, so no surface is invented between two objects.
    """
    height, width = depth_m.shape
    column = np.clip(np.floor(u).astype(np.int64), 0, width - 1)
    row = np.clip(np.floor(v).astype(np.int64), 0, height - 1)
    nearest = depth_m[row, column].astype(np.float64)
    if not interpolate:
        return nearest
    cu, cv = u - 0.5, v - 0.5
    c0, r0 = np.floor(cu).astype(np.int64), np.floor(cv).astype(np.int64)
    fx, fy = cu - c0, cv - r0
    corners = []
    for dr in (0, 1):
        for dc in (0, 1):
            rr = np.clip(r0 + dr, 0, height - 1)
            cc = np.clip(c0 + dc, 0, width - 1)
            corners.append(depth_m[rr, cc].astype(np.float64))
    stack = np.stack(corners)
    positive = (stack > 0.0).all(axis=0)
    safe = np.where(stack > 0.0, stack, 1.0)
    spread_ok = (safe.max(axis=0) - safe.min(axis=0)) <= edge_ratio * safe.min(axis=0)
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
    blended = 1.0 / (weights / safe).sum(axis=0)
    use = positive & spread_ok & (nearest > 0.0)
    return np.where(use, blended, nearest)


def sample_scan(  # pylint: disable=too-many-locals
    depth_m: npt.NDArray[np.floating[Any]],
    focal_px: float,
    pattern: Optional[BeamPattern] = None,
    interpolate: bool = True,
    edge_ratio: float = 0.05,
) -> Scan:
    """The scan of ``depth_m`` (metres along the viewing axis, 0 = no surface) for ``pattern``.

    A beam returns when its pixel is inside the image, the depth there is positive and its range
    does not exceed ``pattern.max_range_m``. See ``_read_depth`` for ``interpolate`` and
    ``edge_ratio`` (a **choice**: depths of neighbouring pixels within 5% are one surface).
    """
    pattern = pattern or BeamPattern()
    height, width = depth_m.shape
    elevation = np.radians(pattern.elevations_deg())[:, None]
    azimuth = np.radians(pattern.azimuths_deg())[None, :]
    # unit beam direction in the camera frame (x right, y down, z forward)
    direction_x = np.sin(azimuth) * np.cos(elevation)
    direction_y = -np.sin(elevation) * np.ones_like(azimuth)
    direction_z = np.cos(azimuth) * np.cos(elevation)
    forward = direction_z > 1e-6
    safe_z = np.where(forward, direction_z, 1.0)
    u = width / 2.0 + focal_px * direction_x / safe_z
    v = height / 2.0 + focal_px * direction_y / safe_z
    inside = forward & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    column_c = np.clip(np.floor(u).astype(np.int64), 0, width - 1)
    row_c = np.clip(np.floor(v).astype(np.int64), 0, height - 1)
    z = np.where(inside, _read_depth(depth_m, u, v, interpolate, edge_ratio), 0.0)
    ranges = z / safe_z  # unit direction: range = z / d_z
    hit = inside & (z > 0.0) & (ranges <= pattern.max_range_m)
    channel_index = np.broadcast_to(np.arange(pattern.channels)[:, None], hit.shape)
    points = np.stack([direction_x, direction_y, direction_z], axis=-1) * ranges[..., None]
    return Scan(
        points=points[hit].astype(np.float32),
        ranges=ranges[hit].astype(np.float32),
        channel=channel_index[hit].astype(np.int32),
        pixel_x=column_c[hit].astype(np.int32),
        pixel_y=row_c[hit].astype(np.int32),
    )


def to_velodyne_frame(points_camera: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    """Camera frame (x right, y down, z forward) to Velodyne frame (x forward, y left, z up)."""
    x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
    return np.asarray(np.stack([z, -x, -y], axis=-1), dtype=np.float32)


def write_kitti_bin(path: Path, points_velodyne: npt.NDArray[np.floating[Any]]) -> None:
    """KITTI ``.bin``: float32 ``x y z intensity`` per point. Intensity is 0: the depth has none."""
    data = np.zeros((points_velodyne.shape[0], 4), dtype=np.float32)
    data[:, :3] = points_velodyne
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(path)
