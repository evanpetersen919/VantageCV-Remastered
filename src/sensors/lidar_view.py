"""Load a rendered frame's depth, class map and boxes as a coloured LiDAR scan (for the viewers).

Shared by ``bin/visualize_lidar_scan.py`` and ``bin/render_lidar_flythrough.py``. The scan is
sampled from the frame's depth map with ``src/sensors/depth_lidar.py`` and every point takes the
Cityscapes colour of the semantic class of the pixel it was read from.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.export.box3d_formats import projection_matrix
from src.ground_truth.semantic_classes import PALETTE
from src.orchestration.live_render import camera_from_image_entry
from src.sensors.depth_lidar import KITTI_PATTERN, BeamPattern, sample_scan

FULL_PATTERN = BeamPattern(channels=64, azimuth_step_deg=0.1, max_range_m=80.0)
BOX_COLOURS = {3: "tab:red", 8: "tab:orange", 6: "tab:brown", 1: "tab:cyan", 10: "tab:green"}
BEV_AHEAD_M, BEV_SIDE_M = 60.0, 30.0
# edges of a box from the corner order of ``box_corners`` (bottom face 0-3, top face 4-7)
BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def class_colours(classes: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """Cityscapes colours (0 to 1) of an array of class ids."""
    table = np.zeros((256, 3), dtype=np.float64)
    for class_id, rgb in PALETTE.items():
        table[class_id] = np.array(rgb) / 255.0
    return np.asarray(table[classes])


def labelled_boxes(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The annotations that have a 3D box of a class drawn in ``BOX_COLOURS``."""
    return [
        a for a in annotations if a.get("box3d") is not None and a["category_id"] in BOX_COLOURS
    ]


@dataclass
class Frame:
    """One frame's scan and what it is drawn with."""

    image: Dict[str, Any]
    annotations: List[Dict[str, Any]]
    label: npt.NDArray[np.float64]  # points in the level camera frame: x right, y down, z forward
    colours: npt.NDArray[np.float64]
    pixel_x: npt.NDArray[np.int32]
    pixel_y: npt.NDArray[np.int32]


def load_frame(
    dataset: Path, frame: int, pattern_name: str, pattern: Optional[BeamPattern] = None
) -> Frame:
    """Sample the scan of the ``frame``-th frame that has a depth map and colour it by class."""
    data = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    image = [i for i in data["images"] if "depth_file" in i][frame]
    annotations = [a for a in data["annotations"] if a["image_id"] == image["id"]]
    depth = np.asarray(Image.open(dataset / image["depth_file"])).astype(np.float64)
    depth = (depth / float(image["depth_scale"])).astype(np.float32)
    classes = np.asarray(Image.open(dataset / image["semantic_file"]))
    camera = camera_from_image_entry(image)
    pattern = pattern or (KITTI_PATTERN if pattern_name == "kitti" else FULL_PATTERN)
    scan = sample_scan(depth, camera.intrinsics.focal_length_x, pattern)
    intrinsic = camera.intrinsics.get_intrinsic_matrix()
    back = np.linalg.inv(intrinsic) @ projection_matrix(camera)[:, :3]
    label = scan.points.astype(np.float64) @ back
    colours = class_colours(classes[scan.pixel_y, scan.pixel_x])
    return Frame(image, annotations, label, colours, scan.pixel_x, scan.pixel_y)
