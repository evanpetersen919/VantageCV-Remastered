"""Sensor effects measured on real scans and applied to a simulated one.

The plan is docs/lidar/improvement_plan.md.

The depth-buffer sampler of ``depth_lidar`` returns a point for every beam that meets a surface
inside the range limit. A real spinning LiDAR loses returns (dark or wet ground, very oblique
surfaces, the near field) and its ranges scatter. Two effects, both with constants derived from
KITTI scans by ``bin/derive_beam_pattern.py`` and stored in ``configs/kitti_beam_pattern.json``:

* ``apply_dropout``: each return is kept with a probability that depends on its ring.
* ``add_range_noise``: the range of each return is perturbed along its beam by a Gaussian.

Both take a ``numpy`` generator so a frame's scan is reproducible from its seed.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from src.sensors.depth_lidar import BeamPattern, Scan, sample_scan


def apply_dropout(scan: Scan, keep_by_channel: Sequence[float], rng: np.random.Generator) -> Scan:
    """``scan`` with each return kept with the probability of its channel (ring)."""
    keep = np.asarray(keep_by_channel, dtype=np.float64)
    if scan.channel.size and int(scan.channel.max()) >= keep.size:
        raise ValueError("keep_by_channel has fewer entries than the scan has channels")
    probability = keep[scan.channel]
    mask = rng.random(scan.channel.shape[0]) < probability
    return Scan(
        points=scan.points[mask],
        ranges=scan.ranges[mask],
        channel=scan.channel[mask],
        pixel_x=scan.pixel_x[mask],
        pixel_y=scan.pixel_y[mask],
    )


def add_range_noise(scan: Scan, sigma_m: float, rng: np.random.Generator) -> Scan:
    """``scan`` with every range perturbed along its beam by a zero-mean Gaussian of ``sigma_m``."""
    if sigma_m <= 0.0 or scan.ranges.size == 0:
        return scan
    noisy: npt.NDArray[np.float32] = np.maximum(
        scan.ranges + rng.normal(0.0, sigma_m, scan.ranges.shape).astype(np.float32), 0.1
    )
    scale = (noisy / scan.ranges)[:, None]
    return replace(scan, points=(scan.points * scale).astype(np.float32), ranges=noisy)


@dataclass(frozen=True)
class Realism:
    """Which measured effects to apply when sampling a scan (each off by default).

    ``elevation_table_deg``: the sensor's ring elevations; ``keep_by_channel``: per-ring keep
    probability; ``noise_sigma_m``: extra range scatter; ``min_points``: a labelled object with
    fewer returns in its box is left unlabelled (the object stays in the scan, as in KITTI).
    """

    elevation_table_deg: Optional[Tuple[float, ...]] = None
    keep_by_channel: Optional[Tuple[float, ...]] = None
    noise_sigma_m: float = 0.0
    min_points: int = 0


def load_realism(
    path: Path, table: bool = False, dropout: bool = False, noise: bool = False, min_points: int = 0
) -> Realism:
    """The selected effects with the constants of ``configs/kitti_beam_pattern.json``."""
    constants = json.loads(path.read_text(encoding="utf-8"))
    return Realism(
        elevation_table_deg=tuple(constants["elevation_table_deg"]) if table else None,
        keep_by_channel=tuple(constants["keep"]) if dropout else None,
        noise_sigma_m=float(constants["noise_sigma_m"]) if noise else 0.0,
        min_points=min_points,
    )


def realistic_scan(  # pylint: disable=too-many-arguments
    depth_m: npt.NDArray[np.floating[Any]],
    focal_px: float,
    pattern: BeamPattern,
    realism: Realism,
    seed: int,
) -> Scan:
    """``sample_scan`` of ``depth_m`` with the selected effects; ``seed`` fixes the draws."""
    if realism.elevation_table_deg is not None:
        pattern = replace(pattern, elevation_table_deg=realism.elevation_table_deg)
    scan = sample_scan(depth_m, focal_px, pattern)
    rng = np.random.default_rng(seed)
    if realism.keep_by_channel is not None:
        scan = apply_dropout(scan, realism.keep_by_channel, rng)
    return add_range_noise(scan, realism.noise_sigma_m, rng)
