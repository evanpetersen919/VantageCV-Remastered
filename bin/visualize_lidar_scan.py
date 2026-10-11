"""Draw a synthetic LiDAR scan from a rendered dataset: three views of the same points.

    PYTHONPATH=. python bin/visualize_lidar_scan.py --dataset live_dataset/kitti_v18 \\
        --frame 12 --pattern kitti --out demo/lidar_viz

The scan is sampled from the frame's depth map (``src/sensors/depth_lidar.py``) and each point is
coloured by the semantic class of the pixel it was read from (Cityscapes colours: road purple,
sidewalk pink, building grey, vegetation green, car blue, person red, pole and sign yellow-grey).

* left: bird's-eye view (x forward up the page, y left), within 60 m ahead and 30 m to each side,
  with the frame's 3D boxes drawn as footprints;
* top right: the points over the camera frame;
* bottom right: the points alone on black, as a LiDAR-only perspective.

``--pattern kitti`` is the registered stage 2 sensor (81.6 degrees across, +2 to -24.8 degrees);
``--pattern full`` is a debug view over the whole 106 x 74 degree camera frame (64 channels,
0.1 degree steps), which no real sensor has but shows the entire scene.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import numpy.typing as npt  # noqa: E402  pylint: disable=wrong-import-position
from matplotlib.gridspec import GridSpec  # noqa: E402  pylint: disable=wrong-import-position
from PIL import Image  # noqa: E402  pylint: disable=wrong-import-position

from src.export.box3d_formats import (  # noqa: E402  pylint: disable=wrong-import-position
    projection_matrix,
)
from src.export.kitti_lidar import box_corners  # noqa: E402  pylint: disable=wrong-import-position
from src.ground_truth.semantic_classes import (  # noqa: E402  pylint: disable=wrong-import-position
    PALETTE,
)
from src.orchestration.live_render import (  # noqa: E402  pylint: disable=wrong-import-position
    camera_from_image_entry,
)
from src.sensors.depth_lidar import (  # noqa: E402  pylint: disable=wrong-import-position
    KITTI_PATTERN,
    BeamPattern,
    sample_scan,
)

FULL_PATTERN = BeamPattern(channels=64, azimuth_step_deg=0.1, max_range_m=80.0)
BOX_COLOURS = {3: "tab:red", 8: "tab:orange", 6: "tab:brown", 1: "tab:cyan", 10: "tab:green"}
BEV_AHEAD_M, BEV_SIDE_M = 60.0, 30.0


def _class_colours(classes: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    table = np.zeros((256, 3), dtype=np.float64)
    for class_id, rgb in PALETTE.items():
        table[class_id] = np.array(rgb) / 255.0
    return np.asarray(table[classes])


def _boxes(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        a for a in annotations if a.get("box3d") is not None and a["category_id"] in BOX_COLOURS
    ]


def draw(  # pylint: disable=too-many-locals,too-many-statements
    dataset: Path, frame: int, pattern_name: str, out: Path
) -> Path:
    """Write ``<out>/<pattern>_<frame>.png`` for the ``frame``-th frame that has a depth map."""
    data = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    image = [i for i in data["images"] if "depth_file" in i][frame]
    annotations = [a for a in data["annotations"] if a["image_id"] == image["id"]]
    depth = np.asarray(Image.open(dataset / image["depth_file"])).astype(np.float64)
    depth = (depth / float(image["depth_scale"])).astype(np.float32)
    classes = np.asarray(Image.open(dataset / image["semantic_file"]))
    camera = camera_from_image_entry(image)
    pattern = KITTI_PATTERN if pattern_name == "kitti" else FULL_PATTERN
    scan = sample_scan(depth, camera.intrinsics.focal_length_x, pattern)
    intrinsic = camera.intrinsics.get_intrinsic_matrix()
    back = np.linalg.inv(intrinsic) @ projection_matrix(camera)[:, :3]
    label = scan.points.astype(np.float64) @ back  # level camera frame: x right, y down, z fwd
    colours = _class_colours(classes[scan.pixel_y, scan.pixel_x])

    figure = plt.figure(figsize=(21, 11), facecolor="white")
    grid = GridSpec(2, 2, width_ratios=[1.0, 1.7], figure=figure)
    bev = figure.add_subplot(grid[:, 0])
    over = figure.add_subplot(grid[0, 1])
    solo = figure.add_subplot(grid[1, 1])

    bev.set_facecolor("black")
    bev.scatter(-label[:, 0], label[:, 2], c=colours, s=2.0, linewidths=0)
    for annotation in _boxes(annotations):
        corners = box_corners(annotation["box3d"])[[0, 1, 2, 3, 0]]
        bev.plot(
            -corners[:, 0], corners[:, 2], color=BOX_COLOURS[annotation["category_id"]], lw=1.4
        )
    bev.set_xlim(BEV_SIDE_M, -BEV_SIDE_M)  # y is left: positive on the left of the page
    bev.set_ylim(0, BEV_AHEAD_M)
    bev.set_aspect("equal")
    bev.set_xlabel("left (m)")
    bev.set_ylabel("forward (m)")
    bev.set_title(f"bird's-eye view, {pattern_name} pattern, {len(label):,} points")

    picture = Image.open(dataset / image["file_name"]).convert("RGB")
    over.imshow(picture)
    over.scatter(scan.pixel_x + 0.5, scan.pixel_y + 0.5, c=colours, s=3.0, linewidths=0)
    over.set_title("points over the camera frame (colour = class)")
    over.axis("off")

    solo.set_facecolor("black")
    solo.scatter(scan.pixel_x, scan.pixel_y, c=colours, s=3.0, linewidths=0)
    solo.set_xlim(0, picture.width)
    solo.set_ylim(picture.height, 0)
    solo.set_aspect("equal")
    solo.set_xticks([])
    solo.set_yticks([])
    solo.set_title("points alone")

    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{pattern_name}_{frame:03d}.png"
    figure.tight_layout()
    figure.savefig(target, dpi=110)
    plt.close(figure)
    return target


def main() -> None:
    """Parse arguments and draw."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frame", type=int, nargs="+", default=[0])
    parser.add_argument("--pattern", choices=["kitti", "full"], default="kitti")
    parser.add_argument("--out", type=Path, default=Path("demo/lidar_viz"))
    args = parser.parse_args()
    for frame in args.frame:
        print(draw(args.dataset, frame, args.pattern, args.out))


if __name__ == "__main__":
    main()
