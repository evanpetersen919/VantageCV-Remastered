"""Tests for the LiDAR follow-up analysis (``bin/analyze_lidar_followup.py``)."""

import importlib.util
import json
from pathlib import Path
from typing import Dict

SPEC = importlib.util.spec_from_file_location(
    "analyze_lidar_followup",
    Path(__file__).resolve().parents[2] / "bin" / "analyze_lidar_followup.py",
)
assert SPEC is not None and SPEC.loader is not None
followup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(followup)


def _write(root: Path, arm: str, seed: int, level: Dict[str, float]) -> None:
    classes = {n: {"3d": {"easy": 0.0, "moderate": v, "hard": 0.0}} for n, v in level.items()}
    payload = {"M": sum(level.values()) / len(level), "classes": classes}
    (root / f"lidar_{arm}_s{seed}_kitti_val.json").write_text(json.dumps(payload), "utf-8")


def _fill(root: Path, e4_shift: float, car_shift: float = 0.0) -> None:
    """Every arm at the same level except E4, shifted; seed jitter is shared."""
    names = {arm for _, pairs in followup.CONTRASTS for pair in pairs for arm in pair}
    for seed in followup.SEEDS:
        base = {"Car": 50.0 + 0.1 * seed, "Pedestrian": 30.0 + 0.2 * seed, "Cyclist": 25.0}
        for arm in names:
            level = dict(base)
            if arm == "E4":
                level = {k: v + e4_shift for k, v in level.items()}
                level["Car"] += car_shift - e4_shift
            _write(root, "E" if arm == "E0" else arm, seed, level)


def test_loss_is_not_supported(tmp_path: Path) -> None:
    """A uniform drop for E4 gives a negative interval and no support."""
    _fill(tmp_path, e4_shift=-3.0)
    result = followup.analyse(tmp_path)
    assert result["confirmatory"]["interval"]["high"] < 0
    assert not result["confirmatory"]["supported"]


def test_gain_with_guard_is_supported(tmp_path: Path) -> None:
    """A gain with car AP level with A's passes the registered rule."""
    _fill(tmp_path, e4_shift=3.0, car_shift=0.0)
    result = followup.analyse(tmp_path)
    assert result["confirmatory"]["supported"]
    assert result["confirmatory"]["guard_holds"]


def test_car_guard_blocks_support(tmp_path: Path) -> None:
    """A gain on the other classes with car AP 2 below A's is not supported."""
    _fill(tmp_path, e4_shift=3.0, car_shift=-2.0)
    result = followup.analyse(tmp_path)
    assert not result["confirmatory"]["guard_holds"]
    assert not result["confirmatory"]["supported"]


def test_report_lists_every_group(tmp_path: Path) -> None:
    """The text report names each contrast group and the confirmatory line."""
    _fill(tmp_path, e4_shift=0.0)
    text = "\n".join(followup.render(followup.analyse(tmp_path)))
    for title, _ in followup.CONTRASTS:
        assert title in text
    assert "CONFIRMATORY E4 - A" in text


def _fill_improvement(root: Path, s4_shift: float) -> None:
    names = {a for _, p in followup.IMPROVEMENT_CONTRASTS for pair in p for a in pair}
    for seed in followup.SEEDS:
        base = {"Car": 50.0, "Pedestrian": 30.0 + 0.2 * seed, "Cyclist": 25.0}
        for arm in names:
            shift = s4_shift if arm == "S4" else 0.0
            _write(root, arm, seed, {k: v + shift for k, v in base.items()})


def test_improvement_confirmatory_is_s4_against_astep(tmp_path: Path) -> None:
    """The improvement study reads S4 - Astep; a uniform gain is supported."""
    _fill_improvement(tmp_path, 2.0)
    result = followup.analyse(tmp_path, "improvement")
    assert result["confirmatory"]["test"] == "S4 - Astep"
    assert result["confirmatory"]["supported"]


def test_improvement_loss_is_not_supported(tmp_path: Path) -> None:
    """A uniform drop for S4 puts the interval below 0."""
    _fill_improvement(tmp_path, -3.0)
    result = followup.analyse(tmp_path, "improvement")
    assert result["confirmatory"]["interval"]["high"] < 0
    assert not result["confirmatory"]["supported"]


def test_insertion_confirmatory_is_o3_against_astep_occ(tmp_path: Path) -> None:
    """The insertion study reads O3 - Astep_occ; a uniform drop for O3 is not supported."""
    names = {a for _, p in followup.INSERTION_CONTRASTS for pair in p for a in pair}
    for seed in followup.SEEDS:
        base = {"Car": 50.0, "Pedestrian": 30.0 + 0.2 * seed, "Cyclist": 25.0}
        for arm in names:
            shift = -3.0 if arm == "O3" else 0.0
            _write(tmp_path, arm, seed, {k: v + shift for k, v in base.items()})
    result = followup.analyse(tmp_path, "insertion")
    assert result["confirmatory"]["test"] == "O3 - Astep_occ"
    assert result["confirmatory"]["interval"]["high"] < 0
    assert not result["confirmatory"]["supported"]
