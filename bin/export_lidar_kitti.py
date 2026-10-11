"""Write KITTI-format LiDAR scans, labels and calibration from a rendered dataset (stage 2 data).

    PYTHONPATH=. python bin/export_lidar_kitti.py --dataset live_dataset/kitti_v18 \\
        --out F:/datasets/kitti_synthetic --first-id 100000

``--dataset`` is rendered with ``--profile kitti --exact-labels --semantic-maps --classes riders``
(so it has a depth map and 3D boxes per frame). For every frame with a depth map this writes, under
``<out>/training``: ``velodyne/<id>.bin`` (the KITTI-pattern scan of ``src/sensors/depth_lidar.py``,
in the Velodyne frame, intensity 0), ``label_2/<id>.txt``, ``calib/<id>.txt`` and a blank
``image_2/<id>.png`` of KITTI's size (OpenPCDet reads the image size; no camera data is used).
Ids start at ``--first-id`` so they never collide with KITTI's own 000000 to 007480. A summary of
points per scan and per labelled car by range is written to ``<out>/synthetic_summary.json``.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.export.box3d_formats import projection_matrix
from src.export.kitti_lidar import (
    KITTI_CX,
    KITTI_CY,
    KITTI_FX,
    KITTI_HEIGHT,
    KITTI_WIDTH,
    calibration,
    frame_labels,
    points_in_box,
)
from src.orchestration.live_render import camera_from_image_entry
from src.sensors.depth_lidar import KITTI_PATTERN, sample_scan, to_velodyne_frame, write_kitti_bin

RANGE_BANDS_M = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 80.0))


def _car_points_by_range(
    label_points: npt.NDArray[np.float64],
    annotations: List[Dict[str, Any]],
    bands: Dict[str, List[int]],
) -> None:
    """Add the point count of every car of the frame to ``bands`` (by distance of its box)."""
    for annotation in annotations:
        box = annotation.get("box3d")
        fully_visible = annotation.get("visibility_fraction", 0.0) > 0.9
        if annotation["category_id"] != 3 or box is None or not fully_visible:
            continue
        distance = float(np.hypot(box["location"][0], box["location"][2]))
        count = int(points_in_box(label_points, box, 0.1).sum())
        for low, high in RANGE_BANDS_M:
            if low <= distance < high:
                bands[f"{low:g}-{high:g}"].append(count)


def export(  # pylint: disable=too-many-locals
    dataset: Path, out: Path, first_id: int
) -> Dict[str, Any]:
    """Write every frame of ``dataset``; returns the summary dictionary."""
    data = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in data["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    root = out / "training"
    for folder in ("velodyne", "label_2", "calib", "image_2"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    blank = Image.new("RGB", (KITTI_WIDTH, KITTI_HEIGHT))
    bands: Dict[str, List[int]] = {f"{low:g}-{high:g}": [] for low, high in RANGE_BANDS_M}
    ids: List[str] = []
    points_per_scan: List[int] = []
    points_in_image: List[int] = []
    labels_per_type: Dict[str, int] = {}
    for image in data["images"]:
        if "depth_file" not in image:
            continue
        sample = f"{first_id + len(ids):06d}"
        depth = np.asarray(Image.open(dataset / image["depth_file"])).astype(np.float64)
        depth = (depth / float(image["depth_scale"])).astype(np.float32)
        camera = camera_from_image_entry(image)
        scan = sample_scan(depth, camera.intrinsics.focal_length_x, KITTI_PATTERN)
        back = (
            np.linalg.inv(camera.intrinsics.get_intrinsic_matrix())
            @ projection_matrix(camera)[:, :3]
        )
        label_points = scan.points.astype(np.float64) @ back
        write_kitti_bin(root / "velodyne" / f"{sample}.bin", to_velodyne_frame(label_points))
        lines = frame_labels(by_image.get(image["id"], []))
        for line in lines:
            kind = line.split()[0]
            labels_per_type[kind] = labels_per_type.get(kind, 0) + 1
        (root / "label_2" / f"{sample}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        (root / "calib" / f"{sample}.txt").write_text(calibration(), encoding="utf-8")
        blank.save(root / "image_2" / f"{sample}.png")
        _car_points_by_range(label_points, by_image.get(image["id"], []), bands)
        points_per_scan.append(len(scan.points))
        depth_z = np.maximum(label_points[:, 2], 1e-6)
        u = KITTI_FX * label_points[:, 0] / depth_z + KITTI_CX
        v = KITTI_FX * label_points[:, 1] / depth_z + KITTI_CY
        points_in_image.append(
            int(((u >= 0) & (u < KITTI_WIDTH) & (v >= 0) & (v < KITTI_HEIGHT)).sum())
        )
        ids.append(sample)
    (out / "ImageSets").mkdir(parents=True, exist_ok=True)
    (out / "ImageSets" / "synthetic.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    summary = {
        "frames": len(ids),
        "first_id": ids[0] if ids else None,
        "points_per_scan_mean": float(np.mean(points_per_scan)) if points_per_scan else 0.0,
        "points_in_kitti_image_mean": float(np.mean(points_in_image)) if points_in_image else 0.0,
        "labels_per_type": labels_per_type,
        "car_points_by_range_m": {
            name: {
                "cars": len(counts),
                "median_points": float(np.median(counts)) if counts else None,
            }
            for name, counts in bands.items()
        },
    }
    (out / "synthetic_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    """Parse arguments, export, print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-id", type=int, default=100000)
    args = parser.parse_args()
    print(json.dumps(export(args.dataset, args.out, args.first_id), indent=1))


if __name__ == "__main__":
    main()
