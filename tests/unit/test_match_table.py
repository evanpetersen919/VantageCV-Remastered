"""Tests for the resampling-ready AP (``src/evaluation/match_table.py``) against pycocotools."""

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.evaluation import match_table as mt


def _random_set(seed: int, images: int = 40) -> Dict[str, Any]:
    """Ground truth with two classes, some crowd regions, and noisy detections of it."""
    rng = np.random.default_rng(seed)
    gt: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    for image in range(images):
        for _ in range(int(rng.integers(0, 6))):
            category = int(rng.integers(1, 3))
            x, y = rng.uniform(0, 500, 2)
            w, h = rng.uniform(10, 120, 2)
            crowd = int(rng.random() < 0.1)
            gt.append(
                {
                    "id": len(gt) + 1,
                    "image_id": image,
                    "category_id": category,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": crowd,
                }
            )
            if rng.random() < 0.8:
                jitter = rng.normal(0, 4, 4)
                detections.append(
                    {
                        "image_id": image,
                        "category_id": category,
                        "bbox": [
                            x + jitter[0],
                            y + jitter[1],
                            max(w + jitter[2], 3),
                            max(h + jitter[3], 3),
                        ],
                        "score": float(rng.uniform(0.05, 1.0)),
                    }
                )
        for _ in range(int(rng.integers(0, 4))):  # false positives
            x, y = rng.uniform(0, 500, 2)
            detections.append(
                {
                    "image_id": image,
                    "category_id": int(rng.integers(1, 3)),
                    "bbox": [x, y, 40.0, 40.0],
                    "score": float(rng.uniform(0.01, 0.9)),
                }
            )
    return {
        "images": [{"id": i, "width": 640, "height": 640} for i in range(images)],
        "annotations": gt,
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        "detections": detections,
    }


def _coco_ap(data: Dict[str, Any], category: int) -> float:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {k: data[k] for k in ("images", "annotations", "categories")}
        gt.createIndex()
        dt = gt.loadRes(data["detections"])
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.catIds = [category]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, 0, 0, 2]  # all areas, maxDets 100
    return float(np.mean(precision[precision > -1]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pycocotools(seed: int) -> None:
    """Per-class AP@[.5:.95] equals COCOeval's to 1e-9, with crowd regions and false positives."""
    data = _random_set(seed)
    table = mt.build_match_table(data, data["detections"])
    for category in (1, 2):
        ours = mt.average_precision(table["classes"][category])
        assert ours == pytest.approx(_coco_ap(data, category), abs=1e-9)


def test_integer_weights_equal_duplicating_images() -> None:
    """A bootstrap weight of 2 on an image is the same as listing the image twice."""
    data = _random_set(5, images=12)
    table = mt.build_match_table(data, data["detections"])
    weights = np.ones(12)
    weights[3], weights[7] = 2.0, 0.0
    weighted = mt.average_precision(table["classes"][1], weights)
    # duplicate image 3, drop image 7, by rebuilding the data set
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    mapping = [i for i in range(12) if i != 7] + [3]
    for new_id, old in enumerate(mapping):
        images.append({"id": new_id, "width": 640, "height": 640})
        for a in data["annotations"]:
            if a["image_id"] == old:
                annotations.append({**a, "image_id": new_id, "id": len(annotations) + 1})
        for d in data["detections"]:
            if d["image_id"] == old:
                detections.append({**d, "image_id": new_id})
    rebuilt = {"images": images, "annotations": annotations, "categories": data["categories"]}
    direct = mt.average_precision(mt.build_match_table(rebuilt, detections)["classes"][1])
    assert weighted == pytest.approx(direct, abs=1e-9)


def test_class_without_ground_truth_is_nan() -> None:
    """Zero-weighting every image with ground truth leaves nothing to average: nan."""
    data = _random_set(1, images=6)
    table = mt.build_match_table(data, data["detections"])
    assert np.isnan(mt.average_precision(table["classes"][1], np.zeros(6)))


def test_table_round_trips_through_npz(tmp_path: Path) -> None:
    """Saving and loading a table gives the same AP."""
    data = _random_set(2)
    table = mt.build_match_table(data, data["detections"])
    path = tmp_path / "table.npz"
    mt.save_table(path, table)
    loaded = mt.load_table(path)
    assert mt.average_precision(loaded["classes"][2]) == pytest.approx(
        mt.average_precision(table["classes"][2]), abs=1e-12
    )
