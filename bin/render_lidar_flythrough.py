"""Render a camera flythrough of one synthetic LiDAR scan as a GIF (visualisation, not study data).

    PYTHONPATH=. python bin/render_lidar_flythrough.py --dataset demo/kitti_probe --frame 6 \\
        --out demo/lidar_viz/flythrough_006.gif

The scan is sampled from the frame's depth map (``src/sensors/depth_lidar.py``) over the whole
camera frame with ``--channels`` beams (default 192, denser than the 64 of the study sensor so
the picture is smooth; the study's scans are unaffected) and drawn by a small point renderer: a
pinhole camera moves along a smooth path through the scene (behind the sensor, down the street
between the cars, past them looking back, then up and over for a high side view), points are
splatted with distance fade and a soft glow, and the labelled 3D boxes are wireframes. Points are
coloured by Cityscapes class. Written with Pillow only (no video codec is installed), so a GIF.
"""

import argparse
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

from src.export.kitti_lidar import box_corners
from src.sensors.depth_lidar import BeamPattern
from src.sensors.lidar_view import BOX_COLOURS, BOX_EDGES, labelled_boxes, load_frame

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


def _basis(
    position: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward])


def project(
    points: npt.NDArray[np.float64],
    position: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    size: Tuple[int, int],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pixel coordinates (x, y) and camera depth of ``points`` seen from the camera."""
    width, height = size
    focal = (width / 2.0) / np.tan(np.radians(HFOV_DEG) / 2.0)
    cam = (points - position) @ _basis(position, target).T
    depth = cam[:, 2]
    safe = np.maximum(depth, 1e-3)
    x = width / 2.0 + focal * cam[:, 0] / safe
    y = height / 2.0 - focal * cam[:, 1] / safe
    return np.stack([x, y], axis=1), depth


def render_frame(  # pylint: disable=too-many-locals,too-many-arguments
    points: npt.NDArray[np.float64],
    colours: npt.NDArray[np.float64],
    boxes: List[Tuple[npt.NDArray[np.float64], Any]],
    position: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    size: Tuple[int, int],
) -> Image.Image:
    """One frame: splatted points with fade and glow, then box wireframes."""
    width, height = size
    pixels, depth = project(points, position, target, size)
    visible = (depth > 0.8) & (pixels[:, 0] > 1) & (pixels[:, 0] < width - 2)
    visible &= (pixels[:, 1] > 1) & (pixels[:, 1] < height - 2)
    px = pixels[visible]
    z = depth[visible]
    fade = (0.30 + 0.70 * np.exp(-z / 40.0))[:, None]
    tint = np.clip(colours[visible] * 1.25 + 0.08, 0.0, 1.0) * fade
    buffer = np.zeros((height, width, 3), dtype=np.float32)
    x0 = np.rint(px[:, 0]).astype(np.int64)
    y0 = np.rint(px[:, 1]).astype(np.int64)
    near = z < 22.0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            weight = 1.0 if (dx == 0 and dy == 0) else 0.55
            use = near | ((dx == 0) & (dy == 0))
            xs = np.clip(x0[use] + dx, 0, width - 1)
            ys = np.clip(y0[use] + dy, 0, height - 1)
            np.add.at(buffer, (ys, xs), (tint[use] * weight).astype(np.float32))
    glow = gaussian_filter(buffer, sigma=(5.0, 5.0, 0))
    image = 1.0 - np.exp(-(buffer * 0.8 + glow * 1.6))  # soft tone-map: dense areas do not clip
    picture = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(picture)
    for corners, category in boxes:
        edge_px, edge_z = project(corners, position, target, size)
        if (edge_z < 3.0).any():
            continue
        colour = BOX_RGB[category]
        for first, second in BOX_EDGES:
            draw.line([tuple(edge_px[first]), tuple(edge_px[second])], fill=colour, width=2)
    return picture


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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.still is not None:
        index = args.still
        picture = render_frame(
            points, loaded.colours, boxes, positions[index], targets[index], size
        )
        still = args.out.with_suffix(".png")
        picture.save(still)
        print(still)
        return
    pictures = [
        render_frame(points, loaded.colours, boxes, positions[k], targets[k], size)
        for k in range(args.frames)
    ]
    pictures[0].save(
        args.out, save_all=True, append_images=pictures[1:], duration=55, loop=0, optimize=True
    )
    print(args.out, f"{args.out.stat().st_size / 1e6:.1f} MB", len(loaded.label), "points")


if __name__ == "__main__":
    main()
