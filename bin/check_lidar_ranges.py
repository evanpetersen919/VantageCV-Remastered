"""Stage 1 range check of the depth-derived LiDAR (docs/lidar/preregistration.md, checks 3.1, 3.2).

    PYTHONPATH=. python bin/check_lidar_ranges.py --out results/lidar_range_check.json

Needs the game running. For each wall distance a scene of a level ground plane and one vertical
wall at that distance (both of known position) is loaded, the engine's depth is captured from a
level camera 1.71 m above the ground (``CaptureObjectMasks``, the capture the datasets use), a
scan is sampled with ``src/sensors/depth_lidar.py`` and every return's range is compared with the
analytic range of the surface its beam meets: ``min(D / d_z, h / d_y)`` for a wall at ``D`` and a
ground ``h`` below the sensor, ``d`` the beam's unit direction.

Reported per range band: the median and 99th percentile of the absolute range error, for the
registered sampler (interpolated) and for nearest-pixel lookup beside it, and the check 3.2
identities. The pass rule is the registered one (median <= 0.02 m, p99 <= 0.10 m up to 50 m).
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from src.orchestration.live_render import LiveRenderer, vertical_fov_deg
from src.orchestration.scenario_serializer import _mesh_to_json
from src.orchestration.semantic_labels import capture_semantic
from src.procedural.mesh_factory import Mesh, flat_quad_mesh
from src.sensors.depth_lidar import BeamPattern, Scan, sample_scan
from src.ue5.backend import UE5Backend

SENSOR_HEIGHT_M = 1.71
WALL_DISTANCES_M = (5.0, 10.0, 20.0, 30.0, 50.0)
BANDS_M = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 50.0), (50.0, 80.0))
MEDIAN_LIMIT_M, P99_LIMIT_M, CHECKED_UP_TO_M = 0.02, 0.10, 50.0
WALL_HALF_WIDTH_M, WALL_HEIGHT_M = 60.0, 30.0


def wall_mesh(distance: float) -> Mesh:
    """A vertical wall facing the camera at ``distance`` ahead, drawn from both sides."""
    y, z = WALL_HALF_WIDTH_M, WALL_HEIGHT_M
    vertices = np.array(
        [[distance, -y, 0.0], [distance, y, 0.0], [distance, y, z], [distance, -y, z]],
        dtype=np.float64,
    )
    return Mesh(
        vertices=vertices,
        triangles=np.array([0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2], dtype=np.int64),
        uvs=vertices[:, 1:] / 4.0,
        material="asphalt",
    )


def expected_range(scan: Scan, distance: float) -> npt.NDArray[np.float64]:
    """Analytic range of the surface each beam meets (wall at ``distance`` or the ground)."""
    direction = scan.points.astype(np.float64) / scan.ranges.astype(np.float64)[:, None]
    wall = distance / direction[:, 2]
    with np.errstate(divide="ignore"):
        ground = np.where(direction[:, 1] > 1e-9, SENSOR_HEIGHT_M / direction[:, 1], np.inf)
    return np.minimum(wall, ground)


def band_statistics(ranges: npt.NDArray[Any], errors: npt.NDArray[Any]) -> List[Dict[str, Any]]:
    """Median and 99th percentile of ``errors`` per range band."""
    rows = []
    for low, high in BANDS_M:
        pick = (ranges >= low) & (ranges < high)
        if not pick.any():
            rows.append({"band_m": [low, high], "points": 0})
            continue
        rows.append(
            {
                "band_m": [low, high],
                "points": int(pick.sum()),
                "median_abs_error_m": float(np.median(errors[pick])),
                "p99_abs_error_m": float(np.percentile(errors[pick], 99)),
            }
        )
    return rows


def envelope_share(scan: Scan, depth_m: npt.NDArray[np.floating[Any]]) -> float:
    """Share of points whose camera-frame z lies within the min and max of the 3x3 pixels."""
    height, width = depth_m.shape
    inside = 0
    for x, y, z in zip(scan.pixel_x, scan.pixel_y, scan.points[:, 2]):
        window = depth_m[max(y - 1, 0) : min(y + 2, height), max(x - 1, 0) : min(x + 2, width)]
        window = window[window > 0]
        inside += int(window.size > 0 and window.min() - 1e-3 <= z <= window.max() + 1e-3)
    return inside / max(len(scan.points), 1)


async def measure(renderer: LiveRenderer, distance: float, work: Path) -> Tuple[Any, ...]:
    """Load the scene for ``distance``, capture its depth; returns (depth_m, focal_px)."""
    ground = flat_quad_mesh(-5.0, -80.0, 160.0, 80.0, 0.0, 4.0, "asphalt")
    await renderer.load(
        {"meshes": [_mesh_to_json(ground), _mesh_to_json(wall_mesh(distance))], "assets": []}
    )
    position = np.array([0.0, 0.0, SENSOR_HEIGHT_M])
    look_at = np.array([100.0, 0.0, SENSOR_HEIGHT_M])
    width, height = await renderer.capture(position, look_at, work / f"known_{distance:g}.png")
    groups: List[Tuple[int, Dict[str, List[int]]]] = [
        (1, {"assets": [], "meshes": [0, 1], "glows": []})
    ]
    maps = await capture_semantic(renderer.backend, groups, (width, height), work / "depth_tmp")
    focal = (height / 2.0) / np.tan(np.radians(vertical_fov_deg()) / 2.0)
    return maps.depth_m, focal


async def run(uri: str, out: Path) -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """Run the check at every wall distance; returns the result dictionary."""
    pattern = BeamPattern(channels=64, azimuth_step_deg=0.1, max_range_m=80.0)
    results: Dict[str, Any] = {"sensor_height_m": SENSOR_HEIGHT_M, "pattern": pattern.__dict__}
    errors = {"interpolated": [], "nearest": []}  # type: Dict[str, List[Any]]
    ranges: List[Any] = []
    identities: List[Dict[str, float]] = []
    async with UE5Backend(uri, timeout_seconds=300.0) as backend:
        renderer = LiveRenderer(backend)
        with tempfile.TemporaryDirectory() as work:
            for distance in WALL_DISTANCES_M:
                depth_m, focal = await measure(renderer, distance, Path(work))
                for mode, interpolate in (("interpolated", True), ("nearest", False)):
                    scan = sample_scan(depth_m, focal, pattern, interpolate=interpolate)
                    errors[mode].append(np.abs(scan.ranges - expected_range(scan, distance)))
                    if mode == "interpolated":
                        ranges.append(scan.ranges)
                        height, width = depth_m.shape
                        u = width / 2.0 + focal * scan.points[:, 0] / scan.points[:, 2]
                        v = height / 2.0 + focal * scan.points[:, 1] / scan.points[:, 2]
                        identities.append(
                            {
                                "wall_m": distance,
                                "points": int(len(scan.points)),
                                "pixel_mismatch": int(
                                    (np.floor(u) != scan.pixel_x).sum()
                                    + (np.floor(v) != scan.pixel_y).sum()
                                ),
                                "max_distance_outside_pixel_px": float(
                                    max(
                                        np.maximum(
                                            np.maximum(scan.pixel_x - u, u - (scan.pixel_x + 1)), 0
                                        ).max(),
                                        np.maximum(
                                            np.maximum(scan.pixel_y - v, v - (scan.pixel_y + 1)), 0
                                        ).max(),
                                    )
                                ),
                                "share_depth_in_pixel_envelope": envelope_share(scan, depth_m),
                            }
                        )
    all_ranges = np.concatenate(ranges)
    for mode, mode_errors in errors.items():
        err = np.concatenate(mode_errors)
        results[mode] = band_statistics(all_ranges, err)
        near = all_ranges < CHECKED_UP_TO_M
        results[f"{mode}_up_to_{CHECKED_UP_TO_M:g}m"] = {
            "points": int(near.sum()),
            "median_abs_error_m": float(np.median(err[near])),
            "p99_abs_error_m": float(np.percentile(err[near], 99)),
        }
    registered = results[f"interpolated_up_to_{CHECKED_UP_TO_M:g}m"]
    results["self_consistency"] = identities
    results["passed_3_1"] = bool(
        registered["median_abs_error_m"] <= MEDIAN_LIMIT_M
        and registered["p99_abs_error_m"] <= P99_LIMIT_M
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def main() -> None:
    """Parse arguments, run, print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--ue5-uri", default="ws://localhost:8765")
    parser.add_argument("--out", type=Path, default=Path("results/lidar_range_check.json"))
    args = parser.parse_args()
    results = asyncio.run(run(args.ue5_uri, args.out))
    print(json.dumps({k: v for k, v in results.items() if k != "pattern"}, indent=1))


if __name__ == "__main__":
    main()
