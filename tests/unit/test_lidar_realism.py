"""Tests for the measured sensor effects (``src/sensors/lidar_realism.py``) and the beam table."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.sensors.depth_lidar import BeamPattern, Scan, sample_scan
from src.sensors.lidar_realism import (
    Realism,
    add_range_noise,
    apply_dropout,
    load_realism,
    realistic_scan,
)


def _scan(count: int = 2000, channels: int = 4) -> Scan:
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(count, 3)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    ranges = rng.uniform(5.0, 60.0, count).astype(np.float32)
    return Scan(
        points=direction * ranges[:, None],
        ranges=ranges,
        channel=(np.arange(count) % channels).astype(np.int32),
        pixel_x=np.zeros(count, dtype=np.int32),
        pixel_y=np.zeros(count, dtype=np.int32),
    )


def test_elevation_table_replaces_uniform_spacing() -> None:
    """An explicit table is returned as given; a wrong length is rejected."""
    table = (-10.0, -9.0, 0.0, 2.0)
    assert BeamPattern(channels=4, elevation_table_deg=table).elevations_deg().tolist() == [
        -10.0,
        -9.0,
        0.0,
        2.0,
    ]
    with pytest.raises(ValueError):
        BeamPattern(channels=5, elevation_table_deg=table).elevations_deg()


def test_dropout_keeps_the_requested_share_per_channel() -> None:
    """Channel 0 is dropped entirely, channel 1 kept entirely, channel 2 about half."""
    scan = _scan()
    out = apply_dropout(scan, [0.0, 1.0, 0.5, 1.0], np.random.default_rng(1))
    counts = np.bincount(out.channel, minlength=4)
    assert counts[0] == 0
    assert counts[1] == 500
    assert 200 < counts[2] < 300
    assert out.points.shape[0] == out.ranges.shape[0] == out.pixel_x.shape[0]


def test_dropout_is_reproducible() -> None:
    """The same generator seed gives the same survivors."""
    scan = _scan()
    first = apply_dropout(scan, [0.5] * 4, np.random.default_rng(7))
    second = apply_dropout(scan, [0.5] * 4, np.random.default_rng(7))
    assert np.array_equal(first.points, second.points)


def test_noise_moves_points_along_their_beam() -> None:
    """Directions are unchanged, ranges scatter by about sigma, and sigma 0 is the identity."""
    scan = _scan()
    noisy = add_range_noise(scan, 0.03, np.random.default_rng(2))
    before = scan.points / scan.ranges[:, None]
    after = noisy.points / noisy.ranges[:, None]
    assert np.allclose(before, after, atol=1e-4)
    assert abs(float(np.std(noisy.ranges - scan.ranges)) - 0.03) < 0.005
    assert add_range_noise(scan, 0.0, np.random.default_rng(2)) is scan


def test_realistic_scan_with_nothing_selected_matches_the_plain_sampler() -> None:
    """All effects off reproduces ``sample_scan`` exactly."""
    depth = np.full((120, 200), 20.0, dtype=np.float32)
    pattern = BeamPattern(channels=8, vertical_fov_deg=10.0, horizontal_fov_deg=30.0)
    plain = sample_scan(depth, 150.0, pattern)
    same = realistic_scan(depth, 150.0, pattern, Realism(), seed=3)
    assert np.array_equal(plain.points, same.points)


def test_load_realism_selects_effects(tmp_path: Path) -> None:
    """Only the requested effects are filled in from the constants file."""
    path = tmp_path / "pattern.json"
    path.write_text(
        json.dumps({"elevation_table_deg": [0.0, 1.0], "keep": [0.5, 1.0], "noise_sigma_m": 0.02}),
        encoding="utf-8",
    )
    only_table = load_realism(path, table=True)
    assert only_table.elevation_table_deg == (0.0, 1.0)
    assert only_table.keep_by_channel is None and only_table.noise_sigma_m == 0.0
    full = load_realism(path, table=True, dropout=True, noise=True, min_points=5)
    assert full.keep_by_channel == (0.5, 1.0)
    assert full.noise_sigma_m == 0.02 and full.min_points == 5
