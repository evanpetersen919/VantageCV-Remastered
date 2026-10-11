"""Class- and range-conditioned reflectance for the depth-derived LiDAR scans (follow-up study, T5).

The engine's depth carries no intensity, so every synthetic point had reflectance 0 while real KITTI
points carry values from about 0.1 to 0.5 that depend on the surface and fall with range
(``bin/audit_lidar_domain_gap.py``). Here each point draws a value from the histogram measured on
KITTI's training split for its kind (car, pedestrian, cyclist, background) and range band
(``configs/kitti_derived_constants.json``). It is a statistical stand-in for the sensor, not a
physical reflectance model: a point's value depends on its class and range, not on its material or
incidence angle.
"""

from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from src.ground_truth import semantic_classes as sem

BAND_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 80.0])
BAND_NAMES = ("0-10", "10-20", "20-30", "30-40", "40-80")
KIND_BY_CLASS = {
    sem.CAR: "Car",
    sem.TRUCK: "Car",
    sem.BUS: "Car",
    sem.PERSON: "Pedestrian",
    sem.RIDER: "Cyclist",
    sem.BICYCLE: "Cyclist",
}


def sample_intensity(  # pylint: disable=too-many-locals
    class_ids: npt.NDArray[np.uint8],
    ranges: npt.NDArray[np.floating[Any]],
    constants: Dict[str, Any],
    seed: int,
    min_points: int = 200,
) -> npt.NDArray[np.float32]:
    """One reflectance in [0, 1] per point from KITTI's class and range conditioned histograms.

    A (kind, band) with fewer than ``min_points`` measured points in the constants falls back to the
    background histogram of the same band. Deterministic for a given ``seed``.
    """
    histograms = constants["intensity"]["histograms"]
    edges = np.asarray(constants["intensity"]["bin_edges"], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    band = np.clip(np.searchsorted(BAND_EDGES, ranges, side="right") - 1, 0, len(BAND_NAMES) - 1)
    kinds = np.full(len(class_ids), "background", dtype=object)
    for class_id, kind in KIND_BY_CLASS.items():
        kinds[class_ids == class_id] = kind
    out = np.zeros(len(class_ids), dtype=np.float32)
    for kind in set(kinds.tolist()):
        for b, name in enumerate(BAND_NAMES):
            pick = np.nonzero((kinds == kind) & (band == b))[0]
            if pick.size == 0:
                continue
            entry: Dict[str, Any] = histograms[kind][name]
            if entry["points"] < min_points:
                entry = histograms["background"][name]
            probability = np.asarray(entry["probability"], dtype=np.float64)
            probability = probability / probability.sum()
            chosen = rng.choice(len(probability), size=pick.size, p=probability)
            low, high = edges[chosen], edges[chosen + 1]
            out[pick] = (low + (high - low) * rng.random(pick.size)).astype(np.float32)
    return out
