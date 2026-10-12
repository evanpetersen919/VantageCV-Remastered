"""Derive the KITTI beam table, per-ring return fill and range noise from real scans.

    PYTHONPATH=. python bin/derive_beam_pattern.py --kitti F:/datasets/kitti \\
        --dataset live_dataset/kitti_v19 --out configs/kitti_beam_pattern.json

Real side (every ``--every``-th KITTI training scan, forward sector only, ``|azimuth| <= 40.8``):

1. ring elevations: a histogram of return elevations in the Velodyne frame, peaks found with
   ``scipy.signal.find_peaks``. The Velodyne HDL-64E has two blocks of 32 lasers; the lower block's
   peaks are read as they are, the upper block's span is resampled to 32 even steps (its peaks are
   sparse, so a few are spurious or missing);
2. per-ring fill: points per ring divided by the azimuth slots of the sector (0.18 degrees);
3. ground range noise: robust (MAD) scatter of ground points about a fitted plane at 8 to 25 m.

Synthetic side (the same measurements on scans sampled from ``--dataset``'s depth maps with the
derived table): per-ring fill, and the ground scatter. ``keep`` is real fill over synthetic fill per
ring, capped at 1 (where the synthetic ring is almost empty, 1: there is nothing to drop), and
``noise_sigma_m`` is the extra scatter that makes the synthetic ground as rough as the real one,
``sqrt(real^2 - synthetic^2)``. Everything is written to the JSON with its inputs.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.signal import find_peaks

from src.export.box3d_formats import projection_matrix
from src.orchestration.live_render import camera_from_image_entry
from src.sensors.depth_lidar import KITTI_PATTERN, BeamPattern, sample_scan, to_velodyne_frame

HALF_FIELD_DEG = 40.8
SLOTS_PER_RING = int(round(2 * HALF_FIELD_DEG / 0.18)) + 1
MIN_RANGE_M = 3.0
RING_TOLERANCE_DEG = 0.15


def _sector(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Forward-sector points beyond ``MIN_RANGE_M``: columns x, y, z, elevation, range."""
    xyz = points[:, :3].astype(np.float64)
    horizontal = np.hypot(xyz[:, 0], xyz[:, 1])
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    keep = (np.abs(azimuth) <= HALF_FIELD_DEG) & (xyz[:, 0] > 0) & (horizontal > MIN_RANGE_M)
    xyz = xyz[keep]
    elevation = np.degrees(np.arctan2(xyz[:, 2], horizontal[keep]))
    return np.column_stack([xyz, elevation, horizontal[keep]])


def ring_table(elevations: npt.NDArray[np.float64]) -> Tuple[List[float], List[float]]:
    """The 64-entry table (bottom to top) and the raw peak positions it was built from."""
    bins = np.arange(-25.5, 3.5, 0.01)
    counts, _ = np.histogram(elevations, bins)
    smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    peaks, _ = find_peaks(smooth, prominence=smooth.max() * 0.01, distance=8)
    centres = ((bins[:-1] + bins[1:]) / 2.0)[peaks]
    if centres.size < 64:
        raise ValueError(f"only {centres.size} elevation peaks found, need at least 64")
    lower = centres[:32]
    upper = np.linspace(centres[32], centres[-1], 32)
    return list(np.concatenate([lower, upper]).astype(float)), list(centres.astype(float))


def ground_sigma(points_velodyne: npt.NDArray[np.floating[Any]]) -> float:
    """Robust scatter (m) of ground points 8 to 25 m ahead about a fitted plane; NaN if too few."""
    x, y, z = (points_velodyne[:, i].astype(np.float64) for i in range(3))
    keep = (x > 8) & (x < 25) & (np.abs(y) < 4) & (z > -2.4) & (z < -1.2)
    if keep.sum() < 200:
        return float("nan")
    design = np.column_stack([x[keep], y[keep], np.ones(keep.sum())])
    target = z[keep]
    mask = np.ones(target.shape, dtype=bool)
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[mask], target[mask], rcond=None)
        residual = target - design @ coefficients
        mask = np.abs(residual) < 0.15
    spread = residual[mask]
    return float(1.4826 * np.median(np.abs(spread - np.median(spread))))


def _assign_rings(elevation: npt.NDArray[np.float64], table: Sequence[float]) -> npt.NDArray[Any]:
    """Nearest ring of each elevation, or -1 when none is within the tolerance."""
    rings = np.asarray(table)
    nearest = np.abs(elevation[:, None] - rings[None, :]).argmin(axis=1)
    close = np.abs(elevation - rings[nearest]) <= RING_TOLERANCE_DEG
    return np.where(close, nearest, -1)


def real_measurements(
    kitti: Path, every: int, limit: int
) -> Tuple[List[float], List[float], List[float], int]:
    """The ring table, its raw peaks, the fill per ring and the number of scans used."""
    files = sorted((kitti / "training" / "velodyne").glob("*.bin"))[::every][:limit]
    sectors = [_sector(np.fromfile(f, dtype=np.float32).reshape(-1, 4)) for f in files]
    table, peaks = ring_table(np.concatenate([s[:, 3] for s in sectors]))
    fill = np.zeros(64)
    for sector in sectors:
        rings = _assign_rings(sector[:, 3], table)
        fill += np.bincount(rings[rings >= 0], minlength=64)[:64]
    return table, peaks, list(fill / (len(files) * SLOTS_PER_RING)), len(files)


def real_ground_sigma(kitti: Path, every: int, limit: int) -> float:
    """Median over scans of the ground scatter of real scans."""
    files = sorted((kitti / "training" / "velodyne").glob("*.bin"))[::every][:limit]
    values = [ground_sigma(np.fromfile(f, dtype=np.float32).reshape(-1, 4)) for f in files]
    return float(np.nanmedian(values))


def synthetic_measurements(
    dataset: Path, pattern: BeamPattern, count: int
) -> Tuple[List[float], float, int]:
    """(fill per ring, median ground scatter, frames used) of scans sampled with ``pattern``."""
    data = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    images = [i for i in data["images"] if "depth_file" in i][:count]
    fill = np.zeros(pattern.channels)
    sigmas = []
    for image in images:
        depth = np.asarray(Image.open(dataset / image["depth_file"])).astype(np.float64)
        depth = (depth / float(image["depth_scale"])).astype(np.float32)
        camera = camera_from_image_entry(image)
        scan = sample_scan(depth, camera.intrinsics.focal_length_x, pattern)
        fill += np.bincount(scan.channel, minlength=pattern.channels)
        back = (
            np.linalg.inv(camera.intrinsics.get_intrinsic_matrix())
            @ projection_matrix(camera)[:, :3]
        )
        sigmas.append(ground_sigma(to_velodyne_frame(scan.points.astype(np.float64) @ back)))
    slots = len(pattern.azimuths_deg())
    return list(fill / (len(images) * slots)), float(np.nanmedian(sigmas)), len(images)


def main() -> None:
    """Parse arguments, measure, write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--kitti", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("configs/kitti_beam_pattern.json"))
    parser.add_argument("--every", type=int, default=12, help="use every N-th KITTI scan")
    parser.add_argument("--real-scans", type=int, default=300)
    parser.add_argument("--synthetic-frames", type=int, default=250)
    args = parser.parse_args()
    table, peaks, real_fill, used = real_measurements(args.kitti, args.every, args.real_scans)
    sigma_real = real_ground_sigma(args.kitti, args.every, args.real_scans)
    pattern = BeamPattern(
        channels=64,
        horizontal_fov_deg=KITTI_PATTERN.horizontal_fov_deg,
        azimuth_step_deg=KITTI_PATTERN.azimuth_step_deg,
        max_range_m=KITTI_PATTERN.max_range_m,
        elevation_table_deg=tuple(table),
    )
    synth_fill, sigma_synth, frames = synthetic_measurements(
        args.dataset, pattern, args.synthetic_frames
    )
    keep = [1.0 if s < 0.02 else float(min(1.0, r / s)) for r, s in zip(real_fill, synth_fill)]
    result: Dict[str, Any] = {
        "elevation_table_deg": [round(v, 3) for v in table],
        "raw_peaks_deg": [round(v, 3) for v in peaks],
        "real_fill": [round(v, 4) for v in real_fill],
        "synthetic_fill": [round(v, 4) for v in synth_fill],
        "keep": [round(v, 4) for v in keep],
        "ground_sigma_real_m": round(sigma_real, 4),
        "ground_sigma_synthetic_m": round(sigma_synth, 4),
        "noise_sigma_m": round(float(np.sqrt(max(0.0, sigma_real**2 - sigma_synth**2))), 4),
        "real_scans": used,
        "synthetic_frames": frames,
        "slots_per_ring": SLOTS_PER_RING,
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if "peaks" not in k}, indent=1))


if __name__ == "__main__":
    main()
