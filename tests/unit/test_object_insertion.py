"""Tests for pasted-object shadows and database merging (stage 2 of the LiDAR improvement study)."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.export.object_database import DB_FOLDER, DB_INFOS, merge_object_databases
from src.sensors.shadow import visible_mask


def _box(centre_x: float, centre_y: float = 0.0, half: float = 0.5) -> np.ndarray:
    """The 8 corners of an axis-aligned cube (LiDAR frame: x forward, y left, z up)."""
    offsets = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return np.array([centre_x, centre_y, 0.0]) + half * offsets


def test_points_behind_the_box_are_removed_and_others_kept() -> None:
    """A point on the same beam but farther away is hidden; nearer and off-beam points stay."""
    corners = _box(10.0)[None, :, :]
    points = np.array(
        [
            [20.0, 0.0, 0.0],  # behind the box, same beam
            [5.0, 0.0, 0.0],  # in front of the box
            [20.0, 6.0, 0.0],  # far to the side
            [20.0, 0.0, 6.0],  # far above
        ]
    )
    keep = visible_mask(points, corners)
    assert keep.tolist() == [False, True, True, True]


def test_no_boxes_or_no_points_keeps_everything() -> None:
    """Empty inputs return an all-true (or empty) mask."""
    points = np.random.default_rng(0).normal(size=(5, 3)) + np.array([10.0, 0.0, 0.0])
    assert visible_mask(points, np.zeros((0, 8, 3))).all()
    assert visible_mask(np.zeros((0, 3)), _box(10.0)[None]).shape == (0,)


def test_box_behind_the_sensor_is_skipped() -> None:
    """A box with a corner at x <= 0 casts no shadow."""
    points = np.array([[20.0, 0.0, 0.0]])
    assert visible_mask(points, _box(0.2, half=0.5)[None]).all()


def _write_db(root: Path, ident: str, name: str, centre: tuple) -> Dict[str, Any]:
    folder = root / DB_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    relative = f"{DB_FOLDER}/{ident}_{name}_0.bin"
    points = np.zeros((6, 4), dtype=np.float32)
    points[:, 0] = np.linspace(-0.2, 0.2, 6)
    points.tofile(root / relative)
    return {"name": name, "path": relative, "box3d_lidar": np.array([*centre, 0, 1, 1, 1, 0.0])}


def _write_infos(root: Path, items: Dict[str, list]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / DB_INFOS).open("wb") as handle:
        pickle.dump(items, handle)


def test_merge_adds_files_and_infos(tmp_path: Path) -> None:
    """Objects of the extra root appear in the target's pickle and folder, originals untouched."""
    target, extra = tmp_path / "target", tmp_path / "extra"
    _write_infos(target, {"Pedestrian": [_write_db(target, "000001", "Pedestrian", (10, 0, 0))]})
    _write_infos(
        extra,
        {
            "Pedestrian": [_write_db(extra, "900001", "Pedestrian", (20, 1, 0))],
            "Cyclist": [_write_db(extra, "900002", "Cyclist", (15, -1, 0))],
        },
    )
    added = merge_object_databases(target, extra)
    assert added == {"Pedestrian": 1, "Cyclist": 1}
    with (target / DB_INFOS).open("rb") as handle:
        merged = pickle.load(handle)
    assert len(merged["Pedestrian"]) == 2 and len(merged["Cyclist"]) == 1
    assert (target / DB_FOLDER / "900002_Cyclist_0.bin").exists()
    assert (target / DB_FOLDER / "000001_Pedestrian_0.bin").exists()


def test_merge_redraws_intensity_when_constants_are_given(tmp_path: Path) -> None:
    """With constants, an added object's fourth column is drawn from the histogram."""
    constants = json.loads(Path("configs/kitti_derived_constants.json").read_text("utf-8"))
    target, extra = tmp_path / "target", tmp_path / "extra"
    _write_infos(target, {})
    _write_infos(extra, {"Pedestrian": [_write_db(extra, "900001", "Pedestrian", (12, 0, 0))]})
    merge_object_databases(target, extra, constants, seed=3)
    points = np.fromfile(target / DB_FOLDER / "900001_Pedestrian_0.bin", dtype=np.float32)
    values = points.reshape(-1, 4)[:, 3]
    assert (values >= 0.0).all() and (values <= 1.0).all()
    assert values.max() > 0.0
