"""Tests for the LiDAR study's arm selection (``bin/prepare_lidar_arms.py``)."""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_lidar_arms",
    Path(__file__).resolve().parents[2] / "bin" / "prepare_lidar_arms.py",
)
assert SPEC is not None and SPEC.loader is not None
arms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(arms)

REAL = [f"{i:06d}" for i in range(2000)]
SYNTHETIC = [f"{100000 + i}" for i in range(300)]


def _select(**kwargs: Any) -> Dict[str, List[str]]:
    options: Dict[str, Any] = {
        "base_fraction": 0.1,
        "extra": 150,
        "hold": 100,
        "sweep": [50],
        "seed": 0,
    }
    options.update(kwargs)
    result: Dict[str, List[str]] = arms.select_arms(REAL, SYNTHETIC, **options)
    return result


def test_base_is_a_tenth_and_shared_by_every_arm() -> None:
    """A is 200 frames; B and E start with exactly those frames."""
    selected = _select()
    assert len(selected["A"]) == 200
    assert selected["B"][:200] == selected["A"] and selected["E"][:200] == selected["A"]


def test_extra_counts_and_sources() -> None:
    """B adds real frames, E adds synthetic ones, each S of them; the sweep sizes are named."""
    selected = _select()
    assert len(selected["B"]) == 350 and len(selected["E"]) == 350
    assert all(i in set(REAL) for i in selected["B"][200:])
    assert all(i in set(SYNTHETIC) for i in selected["E"][200:])
    assert len(selected["B50"]) == 250 and len(selected["E50"]) == 250


def test_hold_is_disjoint_from_everything_trained_on() -> None:
    """The checkpoint-selection slice shares no frame with the base or the extras."""
    selected = _select()
    held = set(selected["hold"])
    assert len(held) == 100
    assert not held & set(selected["B"]) and not held & set(selected["E"])


def test_selection_is_deterministic_and_seed_dependent() -> None:
    """The same seed gives the same lists; another seed gives another base."""
    assert _select() == _select()
    assert _select(seed=1)["A"] != _select()["A"]


def test_too_small_a_pool_raises() -> None:
    """Asking for more extra frames than exist is an error, not a silent shortfall."""
    with pytest.raises(ValueError):
        _select(extra=5000)
    with pytest.raises(ValueError):
        arms.select_arms(REAL, SYNTHETIC[:10], 0.1, 150, 100, [], 0)
