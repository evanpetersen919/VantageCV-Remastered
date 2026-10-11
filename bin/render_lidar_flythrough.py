"""Render a camera flythrough of one synthetic LiDAR scan as a GIF (visualisation, not study data).

    PYTHONPATH=. python bin/render_lidar_flythrough.py --dataset demo/kitti_probe --frame 6 \\
        --out demo/lidar_viz/flythrough_006.gif

The scan is sampled from the frame's depth map (``src/sensors/depth_lidar.py``) over the whole
camera frame with ``--channels`` beams (default 192, denser than the 64 of the study sensor so
the picture is smooth; the study's scans are unaffected) and drawn by a small point renderer
(``src/sensors/lidar_render.py``): a pinhole camera moves along a smooth path through the scene
(behind the sensor, down the street between the cars, past them looking back, then up and over for
a high side view), points are splatted with distance fade and a soft glow, and the labelled 3D
boxes are wireframes. Points are coloured by Cityscapes class. Written with Pillow only (no video
codec is installed), so a GIF.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.colors import to_rgb
from PIL import Image
from scipy.interpolate import CubicSpline

from src.export.kitti_lidar import box_corners
from src.sensors.depth_lidar import BeamPattern
from src.sensors.lidar_render import draw_boxes, splat, tone_map
from src.sensors.lidar_view import BOX_COLOURS, labelled_boxes, load_frame

# camera keyframes (scene axes: x right, y up, z forward; the sensor is at the origin, 1.71 m up)
KEY_POSITIONS = [(-4.0, 3.2, -7.0), (-1.2, 1.6, 6.0), (4.5, 3.0, 24.0), (10.0, 9.0, 0.0)]
KEY_TARGETS = [(0.0, 0.5, 20.0), (1.0, 0.8, 28.0), (0.0, 0.5, 8.0), (0.0, 0.0, 14.0)]
HFOV_DEG = 68.0
SENSOR_HEIGHT_M = 1.71
BOX_RGB = {k: tuple(int(c * 255) for c in to_rgb(v)) for k, v in BOX_COLOURS.items()}


def camera_path(frames: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Camera positions and look-at targets, smooth through the keyframes, (frames, 3) each."""
    knots = np.linspace(0.0, 1.0, len(KEY_POSITIONS))
    steps = np.linspace(0.0, 1.0, frames)
    position = CubicSpline(knots, np.array(KEY_POSITIONS), bc_type="natural")(steps)
    target = CubicSpline(knots, np.array(KEY_TARGETS), bc_type="natural")(steps)
    return np.asarray(position), np.asarray(target)


def main() -> None:  # pylint: disable=too-many-locals
    """Parse arguments, render the frames, write the GIF."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("demo/lidar_viz/flythrough.gif"))
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--size", type=int, nargs=2, default=[960, 540])
    parser.add_argument("--still", type=int, default=None, help="write only this frame as a PNG")
    args = parser.parse_args()

    pattern = BeamPattern(channels=args.channels, azimuth_step_deg=0.1, max_range_m=80.0)
    loaded = load_frame(args.dataset, args.frame, "full", pattern)
    # scene axes: x right, y up (the sensor is SENSOR_HEIGHT_M above the ground), z forward
    points = np.stack(
        [loaded.label[:, 0], -loaded.label[:, 1] + SENSOR_HEIGHT_M, loaded.label[:, 2]], axis=1
    )
    boxes = []
    for annotation in labelled_boxes(loaded.annotations):
        corners = box_corners(annotation["box3d"])
        scene = np.stack([corners[:, 0], -corners[:, 1] + SENSOR_HEIGHT_M, corners[:, 2]], axis=1)
        boxes.append((scene, annotation["category_id"]))
    positions, targets = camera_path(args.frames)
    size = (int(args.size[0]), int(args.size[1]))

    def render(index: int) -> Image.Image:
        picture = tone_map(
            splat(points, loaded.colours, positions[index], targets[index], size, HFOV_DEG)
        )
        image = Image.fromarray((picture * 255).astype(np.uint8))
        return draw_boxes(image, boxes, BOX_RGB, positions[index], targets[index], HFOV_DEG)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.still is not None:
        still = args.out.with_suffix(".png")
        render(args.still).save(still)
        print(still)
        return
    pictures: List[Image.Image] = [render(k) for k in range(args.frames)]
    pictures[0].save(
        args.out, save_all=True, append_images=pictures[1:], duration=55, loop=0, optimize=True
    )
    print(args.out, f"{args.out.stat().st_size / 1e6:.1f} MB", len(loaded.label), "points")


if __name__ == "__main__":
    main()
