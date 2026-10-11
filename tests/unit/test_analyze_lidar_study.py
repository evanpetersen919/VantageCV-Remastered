"""Tests for the LiDAR study's decision rule (``bin/analyze_lidar_study.py``)."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyze_lidar_study",
    Path(__file__).resolve().parents[2] / "bin" / "analyze_lidar_study.py",
)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def _write(  # pylint: disable=too-many-arguments
    root: Path, arm: str, seed: int, car: float, ped: float, cyc: float
) -> None:
    classes = {
        name: {"3d": {"easy": 0.0, "moderate": value, "hard": 0.0}}
        for name, value in (("Car", car), ("Pedestrian", ped), ("Cyclist", cyc))
    }
    payload = {"M": (car + ped + cyc) / 3.0, "classes": classes}
    (root / f"lidar_{arm}_s{seed}_kitti_val.json").write_text(json.dumps(payload), "utf-8")


def _fill(root: Path, e_gain: float, noise: float = 0.0, car_e: float = 50.0) -> None:
    for seed in study.SEEDS:
        jitter = noise * (seed - 2)
        for arm in ("A", "B"):
            _write(root, arm, seed, 50.0, 30.0 + jitter, 25.0 + jitter)
        _write(root, "E", seed, car_e, 30.0 + e_gain, 25.0 + e_gain)
    for arm in study.SWEEP_ARMS:
        for seed in study.SWEEP_SEEDS:
            _write(root, arm, seed, 50.0, 30.0, 25.0)


def test_consistent_gain_helps_pending_bootstrap(tmp_path: Path) -> None:
    """A clear gain with small noise passes the interval and guard."""
    _fill(tmp_path, e_gain=4.0, noise=0.2)
    result = study.analyse(tmp_path)
    assert result["verdict"].startswith("helps") and "bootstrap" in result["verdict"]


def test_no_gain_is_inconclusive(tmp_path: Path) -> None:
    """Zero difference with noise leaves the interval spanning 0."""
    _fill(tmp_path, e_gain=0.0, noise=0.5)
    assert study.analyse(tmp_path)["verdict"].startswith("inconclusive")


def test_loss_is_hurts(tmp_path: Path) -> None:
    """A consistent drop puts the whole interval below 0."""
    _fill(tmp_path, e_gain=-4.0, noise=0.2)
    assert study.analyse(tmp_path)["verdict"].startswith("hurts")


def test_car_guard_failure_blocks_helps(tmp_path: Path) -> None:
    """A pedestrian and cyclist gain with car AP 2 below A's is not called a gain."""
    _fill(tmp_path, e_gain=4.0, noise=0.2, car_e=48.0)
    result = study.analyse(tmp_path)
    assert not result["guard_holds"]
    assert result["verdict"].startswith("inconclusive")


def test_missing_sweep_arms_are_skipped(tmp_path: Path) -> None:
    """The registered arms alone are enough to analyse; the sweep is optional."""
    _fill(tmp_path, e_gain=4.0, noise=0.2)
    for path in tmp_path.glob("lidar_*250_*.json"):
        path.unlink()
    for path in tmp_path.glob("lidar_*500_*.json"):
        path.unlink()
    result = study.analyse(tmp_path)
    assert result["sweep"] == {}
    assert result["verdict"].startswith("helps")
