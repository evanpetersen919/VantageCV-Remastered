"""Tests for the cyclist study's bootstrap (``bin/bootstrap_rider_study.py``)."""

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.evaluation.match_table import average_precision, save_table

SPEC = importlib.util.spec_from_file_location(
    "bootstrap_rider_study",
    Path(__file__).resolve().parents[2] / "bin" / "bootstrap_rider_study.py",
)
assert SPEC is not None and SPEC.loader is not None
boot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boot)

IMAGES = 60


def _class(rng: np.random.Generator, quality: float) -> Dict[str, Any]:
    n = 400
    score = np.sort(rng.random(n))[::-1]
    return {
        "score": score,
        "image": rng.integers(0, IMAGES, n),
        "matched": rng.random((10, n)) < quality,
        "ignored": np.zeros((10, n), dtype=bool),
        "gt_count": rng.integers(0, 3, IMAGES),
    }


def _write_models(root: Path, e_quality: float, b_quality: float) -> None:
    rng = np.random.default_rng(0)
    for arm, quality in (("B", b_quality), ("E", e_quality)):
        for seed in boot.SEEDS:
            classes = {10: _class(rng, quality), 2: _class(rng, quality)}
            table = {"images": list(range(IMAGES)), "classes": classes}
            save_table(root / f"hpc_rider_{arm}_s{seed}_bdd100k_matches.npz", table)
            published = {
                "overall": {
                    "per_class_ap": {
                        "rider": average_precision(classes[10]),
                        "bike": average_precision(classes[2]),
                    }
                }
            }
            path = root / f"hpc_rider_{arm}_s{seed}_bdd100k_riders.json"
            path.write_text(json.dumps(published), encoding="utf-8")


def test_validation_accepts_tables_that_reproduce_the_scores(tmp_path: Path) -> None:
    """Tables built from the same data as the published AP agree to rounding."""
    _write_models(tmp_path, 0.3, 0.3)
    assert boot.validate(tmp_path, tmp_path)["max_abs_gap"] < 1e-9


def test_validation_catches_a_mismatch(tmp_path: Path) -> None:
    """A published AP that differs from its table by more than the tolerance is reported."""
    _write_models(tmp_path, 0.3, 0.3)
    path = tmp_path / "hpc_rider_E_s0_bdd100k_riders.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["overall"]["per_class_ap"]["rider"] += 0.05
    path.write_text(json.dumps(data), encoding="utf-8")
    assert boot.validate(tmp_path, tmp_path)["max_abs_gap"] > boot.TOLERANCE


def test_better_arm_has_positive_difference_under_resampling(tmp_path: Path) -> None:
    """E with a higher match rate than B gives a positive full-sample and resampled difference."""
    _write_models(tmp_path, 0.6, 0.2)
    boot.validate(tmp_path, tmp_path)  # loads the tables
    assert boot._difference(np.ones(IMAGES)) > 0  # pylint: disable=protected-access
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(20):
        weights = np.bincount(rng.integers(0, IMAGES, IMAGES), minlength=IMAGES).astype(float)
        draws.append(boot._difference(weights))  # pylint: disable=protected-access
    assert np.mean(draws) == pytest.approx(
        boot._difference(np.ones(IMAGES)), abs=0.05  # pylint: disable=protected-access
    )
