"""Tests for the depth-derived LiDAR sampler (``src/sensors/depth_lidar.py``)."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from src.sensors.depth_lidar import BeamPattern, sample_scan, to_velodyne_frame, write_kitti_bin

WIDTH, HEIGHT = 1920, 1080
FOCAL = (HEIGHT / 2.0) / np.tan(np.radians(73.74 / 2.0))
PATTERN = BeamPattern(channels=32, azimuth_step_deg=0.5, max_range_m=200.0)


def _wall(distance: float) -> npt.NDArray[np.float32]:
    return np.full((HEIGHT, WIDTH), distance, dtype=np.float32)


def _ground(height_m: float) -> npt.NDArray[np.float32]:
    """Depth of a level ground plane ``height_m`` below the camera (0 above the horizon)."""
    rows = np.arange(HEIGHT) + 0.5 - HEIGHT / 2.0
    z = np.where(rows > 0.5, height_m * FOCAL / np.maximum(rows, 1e-9), 0.0)
    return np.repeat(z[:, None], WIDTH, axis=1).astype(np.float32)


def test_wall_points_lie_on_the_wall_plane() -> None:
    """Every return has camera-frame z equal to the wall's distance, and range z / cos."""
    scan = sample_scan(_wall(10.0), FOCAL, PATTERN)
    assert len(scan.points) > 0
    assert np.allclose(scan.points[:, 2], 10.0, atol=1e-4)
    direction = scan.points / np.linalg.norm(scan.points, axis=1, keepdims=True)
    assert np.allclose(scan.ranges, 10.0 / direction[:, 2], rtol=1e-5)


def _ground_range_error(interpolate: bool) -> npt.NDArray[np.float32]:
    """Absolute range error of ground returns between 20 and 50 m against the analytic range.

    The ground is ``1.7`` m below the sensor, so a beam whose unit direction has downward
    component ``d_y`` meets it at range ``1.7 / d_y``.
    """
    scan = sample_scan(_ground(1.7), FOCAL, PATTERN, interpolate=interpolate)
    down = scan.points[:, 1] / scan.ranges
    expected = 1.7 / down
    band = (scan.ranges > 20.0) & (scan.ranges < 50.0)
    assert band.sum() > 10
    return np.abs(scan.ranges[band] - expected[band])


def test_ground_plane_range_is_exact_with_interpolation() -> None:
    """Interpolating inverse depth reproduces a plane's range to 1 cm between 20 and 50 m."""
    assert _ground_range_error(True).max() < 0.01


def test_nearest_pixel_is_less_accurate_on_a_slant() -> None:
    """The reason for interpolating: nearest-pixel ground ranges are off by decimetres."""
    assert _ground_range_error(False).max() > 0.05


def test_no_surface_gives_no_return_and_range_limit_applies() -> None:
    """Zero depth is sky (no return); a surface beyond the maximum range is not returned."""
    assert len(sample_scan(np.zeros((HEIGHT, WIDTH), np.float32), FOCAL, PATTERN).points) == 0
    limited = BeamPattern(channels=8, azimuth_step_deg=1.0, max_range_m=5.0)
    assert len(sample_scan(_wall(10.0), FOCAL, limited).points) == 0


def test_no_blending_across_an_object_edge() -> None:
    """A near box in front of a far wall: no point lies between the two depths."""
    depth = _wall(40.0)
    depth[300:800, 800:1100] = 8.0
    scan = sample_scan(depth, FOCAL, PATTERN)
    z = scan.points[:, 2]
    assert np.all(np.isclose(z, 8.0, atol=1e-3) | np.isclose(z, 40.0, atol=1e-3))
    assert np.isclose(z, 8.0, atol=1e-3).sum() > 0


def test_reprojection_returns_the_source_pixel() -> None:
    """Projecting each point through the pinhole lands in the pixel it was read from."""
    scan = sample_scan(_wall(15.0), FOCAL, PATTERN)
    u = WIDTH / 2.0 + FOCAL * scan.points[:, 0] / scan.points[:, 2]
    v = HEIGHT / 2.0 + FOCAL * scan.points[:, 1] / scan.points[:, 2]
    assert np.all(np.floor(u).astype(int) == scan.pixel_x)
    assert np.all(np.floor(v).astype(int) == scan.pixel_y)


def test_velodyne_frame_and_kitti_bin(tmp_path: Path) -> None:
    """Camera right/down/forward becomes Velodyne forward/left/up, and the .bin round-trips."""
    camera = np.array([[1.0, 2.0, 30.0]], dtype=np.float32)  # right 1, down 2, forward 30
    velodyne = to_velodyne_frame(camera)
    assert velodyne.tolist() == [[30.0, -1.0, -2.0]]  # forward, left (-right), up (-down)
    path = tmp_path / "000000.bin"
    write_kitti_bin(path, velodyne)
    data = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    assert data.shape == (1, 4) and data[0, 3] == 0.0
    assert data[0, :3].tolist() == [30.0, -1.0, -2.0]


def test_beam_pattern_counts() -> None:
    """Channel and azimuth counts follow the pattern; the vertical field is centred."""
    pattern = BeamPattern(channels=64, horizontal_fov_deg=100.0, azimuth_step_deg=0.1)
    assert len(pattern.elevations_deg()) == 64
    assert len(pattern.azimuths_deg()) == 1001
    assert pattern.elevations_deg()[0] == pytest.approx(-pattern.vertical_fov_deg / 2.0)
