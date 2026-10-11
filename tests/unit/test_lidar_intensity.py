"""Tests for the class- and range-conditioned reflectance sampler (``lidar_intensity`` module)."""

from typing import Any, Dict

import numpy as np

from src.ground_truth import semantic_classes as sem
from src.sensors.lidar_intensity import sample_intensity


def _histogram(peak: int, points: int = 1000) -> Dict[str, Any]:
    probability = [0.0] * 20
    probability[peak] = 1.0
    return {"points": points, "probability": probability}


def _constants(car_peak: int, background_peak: int) -> Dict[str, Any]:
    bands = ("0-10", "10-20", "20-30", "30-40", "40-80")
    return {
        "intensity": {
            "bin_edges": np.linspace(0.0, 1.0, 21).tolist(),
            "histograms": {
                "Car": {b: _histogram(car_peak) for b in bands},
                "Pedestrian": {b: _histogram(8) for b in bands},
                "Cyclist": {b: _histogram(6, points=10) for b in bands},  # too few: falls back
                "background": {b: _histogram(background_peak) for b in bands},
            },
        }
    }


def test_values_come_from_the_class_histogram() -> None:
    """Car points fall in the car histogram's bin, background points in the background's."""
    ids = np.array([sem.CAR] * 50 + [sem.ROAD] * 50, dtype=np.uint8)
    values = sample_intensity(ids, np.full(100, 15.0), _constants(3, 12), seed=0)
    assert np.all((values[:50] >= 0.15) & (values[:50] < 0.20))
    assert np.all((values[50:] >= 0.60) & (values[50:] < 0.65))


def test_sparse_class_band_falls_back_to_background() -> None:
    """A kind with too few measured points uses the background histogram of the band."""
    ids = np.array([sem.RIDER] * 20, dtype=np.uint8)
    values = sample_intensity(ids, np.full(20, 5.0), _constants(3, 12), seed=1)
    assert np.all((values >= 0.60) & (values < 0.65))


def test_deterministic_and_in_range() -> None:
    """The same seed gives the same values, all inside [0, 1]."""
    ids = np.array([sem.PERSON, sem.CAR, sem.ROAD] * 30, dtype=np.uint8)
    ranges = np.linspace(1.0, 79.0, 90)
    first = sample_intensity(ids, ranges, _constants(3, 12), seed=7)
    second = sample_intensity(ids, ranges, _constants(3, 12), seed=7)
    assert np.array_equal(first, second)
    assert first.min() >= 0.0 and first.max() <= 1.0
