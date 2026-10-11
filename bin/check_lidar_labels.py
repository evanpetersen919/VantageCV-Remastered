"""Stage 1 label-consistency check of the depth-derived LiDAR (preregistration check 3.3).

    PYTHONPATH=. python bin/check_lidar_labels.py DATASET_DIR --out results/lidar_label_check.json

``DATASET_DIR`` is a rendered dataset with ``--exact-labels --semantic-maps`` (so it has per-object
visible masks, 3D boxes and depth maps). For every frame a scan is sampled from its depth map
(``src/sensors/depth_lidar.py``, the 64-channel 0.1 degree pattern, 80 m limit), each point takes
the instance whose visible mask covers the pixel it was read from, and points are expressed in
the label frame (the level camera frame of the 3D boxes, ``src/export/box3d_formats.py``).

For every ``car`` (category 3) with at least ``MIN_POINTS`` labelled points the share of its points
inside its own 3D box, grown by ``MARGIN_M``, is measured. The registered rule is that the pooled
share is at least 95%. Also reported: the per-car share distribution, how many visible cars have
no point or fewer than ``MIN_POINTS`` (their size and range), and the same for pedestrians
(category 1), where the box is looser than the body, as information only.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from PIL import Image
from pycocotools import mask as mask_utils

from src.export.box3d_formats import projection_matrix
from src.orchestration.live_render import camera_from_image_entry
from src.sensors.depth_lidar import BeamPattern, sample_scan

CAR, PERSON = 3, 1
MIN_POINTS = 20
MARGIN_M = 0.1
PASS_SHARE = 0.95
PATTERN = BeamPattern(channels=64, azimuth_step_deg=0.1, max_range_m=80.0)


def points_in_box(
    points: npt.NDArray[np.float64], box: Dict[str, Any], margin: float
) -> npt.NDArray[np.bool_]:
    """Which label-frame ``points`` lie inside the KITTI-convention ``box`` grown by ``margin``.

    ``box`` holds ``location`` (bottom centre), ``dimensions_hwl`` and ``rotation_y``; the length
    axis is the box's x before rotation about the down axis and the width axis its z.
    """
    height, width, length = (float(v) for v in box["dimensions_hwl"])
    centre = np.array(box["location"], dtype=np.float64) - np.array([0.0, height / 2.0, 0.0])
    ry = float(box["rotation_y"])
    cos, sin = np.cos(ry), np.sin(ry)
    rotation = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    local = (points - centre) @ rotation  # rows: R^T (p - c)
    half = np.array([length / 2.0, height / 2.0, width / 2.0]) + margin
    return np.asarray((np.abs(local) <= half).all(axis=1))


def _decode(annotation: Dict[str, Any]) -> npt.NDArray[np.bool_]:
    rle = dict(annotation["mask_rle"])
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return np.asarray(mask_utils.decode(rle)).astype(bool)


def check(dataset: Path, max_frames: int = 0) -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """Run the check over a dataset directory; returns the result dictionary."""
    data = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in data["annotations"]:
        if annotation["category_id"] in (CAR, PERSON) and "mask_rle" in annotation:
            by_image.setdefault(annotation["image_id"], []).append(annotation)
    shares: Dict[int, List[float]] = {CAR: [], PERSON: []}
    inside_total = {CAR: 0, PERSON: 0}
    labelled_total = {CAR: 0, PERSON: 0}
    sparse: List[Dict[str, float]] = []
    frames = 0
    for image in data["images"]:
        annotations = by_image.get(image["id"], [])
        if not annotations or "depth_file" not in image:
            continue
        if max_frames and frames >= max_frames:
            break
        frames += 1
        depth = np.asarray(Image.open(dataset / image["depth_file"])).astype(np.float64)
        depth /= float(image["depth_scale"])
        camera = camera_from_image_entry(image)
        focal = camera.intrinsics.focal_length_x
        scan = sample_scan(depth.astype(np.float32), focal, PATTERN)
        back = (
            np.linalg.inv(camera.intrinsics.get_intrinsic_matrix())
            @ projection_matrix(camera)[:, :3]
        )
        assert np.allclose(back @ back.T, np.eye(3), atol=1e-6), "level rotation is not orthonormal"
        label_points = scan.points.astype(np.float64) @ back  # (back^T p) as row vectors
        for annotation in annotations:
            box = annotation.get("box3d")
            if box is None:
                continue
            mask = _decode(annotation)
            member = mask[scan.pixel_y, scan.pixel_x]
            count = int(member.sum())
            category = annotation["category_id"]
            if count < MIN_POINTS:
                if category == CAR:
                    distance = float(np.linalg.norm(np.array(box["location"])))
                    sparse.append({"points": count, "distance_m": distance})
                continue
            inside = points_in_box(label_points[member], box, MARGIN_M)
            shares[category].append(float(inside.mean()))
            inside_total[category] += int(inside.sum())
            labelled_total[category] += count
    result: Dict[str, Any] = {
        "frames": frames,
        "dataset": str(dataset),
        "margin_m": MARGIN_M,
        "min_points": MIN_POINTS,
        "pattern": {"channels": PATTERN.channels, "azimuth_step_deg": PATTERN.azimuth_step_deg},
    }
    for name, category in (("car", CAR), ("person", PERSON)):
        values = np.array(shares[category])
        result[name] = {
            "objects_with_enough_points": int(len(values)),
            "pooled_share_inside": inside_total[category] / max(labelled_total[category], 1),
            "per_object_share_p5_p50": [float(np.percentile(values, q)) for q in (5, 50)]
            if len(values)
            else [],
            "labelled_points": labelled_total[category],
        }
    distances = [s["distance_m"] for s in sparse]
    result["cars_with_too_few_points"] = {
        "count": len(sparse),
        "median_distance_m": float(np.median(distances)) if distances else None,
    }
    result["passed_3_3"] = bool(
        result["car"]["objects_with_enough_points"] > 0
        and result["car"]["pooled_share_inside"] >= PASS_SHARE
    )
    return result


def main() -> None:
    """Parse arguments, run, print and write the result."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/lidar_label_check.json"))
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    result = check(args.dataset, args.max_frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
