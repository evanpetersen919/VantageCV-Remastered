"""Tests for the cyclist study's decision rule (``bin/analyze_rider_study.py``)."""

import importlib.util
import json
from pathlib import Path
from typing import Dict

SPEC = importlib.util.spec_from_file_location(
    "analyze_rider_study",
    Path(__file__).resolve().parents[2] / "bin" / "analyze_rider_study.py",
)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def _write(root: Path, arm: str, seed: int, ap: Dict[str, float], bench: str = "bdd100k") -> None:
    per_class = {
        name: ap.get(name, 0.3) / 100.0
        for name in ("rider", "bike", "motor", "person", "car", "bus", "truck")
    }
    path = root / f"hpc_rider_{arm}_s{seed}_{bench}_riders.json"
    path.write_text(json.dumps({"overall": {"per_class_ap": per_class}}), encoding="utf-8")


def _fill(root: Path, e_gain: float, noise: float = 0.0, car_e: float = 40.0) -> None:
    for seed in study.SEEDS:
        jitter = noise * (seed - 2)
        base = {"rider": 10.0 + jitter, "bike": 10.0 + jitter, "car": 40.0, "person": 20.0}
        for arm in ("A", "B", "C"):
            _write(root, arm, seed, base)
        _write(
            root, "E", seed, {**base, "rider": 10.0 + e_gain, "bike": 10.0 + e_gain, "car": car_e}
        )
    for arm in study.SWEEP_ARMS:
        for seed in study.SWEEP_SEEDS:
            _write(root, arm, seed, {"rider": 10.0, "bike": 10.0})


def test_clear_gain_is_helps_pending_bootstrap(tmp_path: Path) -> None:
    """A consistent +3 AP gain with small seed noise passes criteria 1, 2 and 4."""
    _fill(tmp_path, e_gain=3.0, noise=0.1)
    result = study.analyse(tmp_path)
    assert result["d_interval"]["low"] > 0 and result["guards_hold"]
    assert result["verdict"].startswith("helps")
    assert "bootstrap" in result["verdict"]


def test_no_gain_is_inconclusive(tmp_path: Path) -> None:
    """Zero mean difference with noise leaves the interval spanning 0."""
    _fill(tmp_path, e_gain=0.0, noise=0.5)
    result = study.analyse(tmp_path)
    assert result["verdict"].startswith("inconclusive")


def test_loss_is_hurts(tmp_path: Path) -> None:
    """A consistent -3 AP difference puts the whole interval below 0."""
    _fill(tmp_path, e_gain=-3.0, noise=0.1)
    result = study.analyse(tmp_path)
    assert result["d_interval"]["high"] < 0
    assert result["verdict"].startswith("hurts")


def test_guard_failure_blocks_helps(tmp_path: Path) -> None:
    """A gain on rider and bike with car AP 2 below A's is not called a gain."""
    _fill(tmp_path, e_gain=3.0, noise=0.1, car_e=38.0)
    result = study.analyse(tmp_path)
    assert not result["guards_hold"]
    assert result["verdict"].startswith("inconclusive")


def test_t_interval_matches_hand_calculation() -> None:
    """For d = 1, 2, 3, 4, 5: mean 3, sd 1.5811, half-width 2.776 * 1.5811 / sqrt(5)."""
    interval = study.paired_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(interval["mean"] - 3.0) < 1e-9
    assert abs((interval["high"] - interval["low"]) / 2.0 - 1.9632) < 1e-3
