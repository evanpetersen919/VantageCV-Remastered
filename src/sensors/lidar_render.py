"""A small point renderer for LiDAR visualisations: pinhole camera, distance fade, soft glow.

Used by ``bin/render_lidar_flythrough.py`` and ``bin/render_lidar_explainer.py``. Scene axes are
x right, y up, z forward, with the ground at y = 0 and the sensor at (0, sensor height, 0).
Nothing here is used to make study data; it only draws scans.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from src.sensors.lidar_view import BOX_EDGES

Array = npt.NDArray[np.float64]


def _basis(position: Array, target: Array) -> Array:
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.asarray(np.stack([right, up, forward]))


def project(
    points: Array, position: Array, target: Array, size: Tuple[int, int], hfov_deg: float
) -> Tuple[Array, Array]:
    """Pixel coordinates (x, y) and camera depth of ``points`` seen from the camera."""
    width, height = size
    focal = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    cam = (points - position) @ _basis(position, target).T
    depth = cam[:, 2]
    safe = np.maximum(depth, 1e-3)
    x = width / 2.0 + focal * cam[:, 0] / safe
    y = height / 2.0 - focal * cam[:, 1] / safe
    return np.stack([x, y], axis=1), depth


def splat(  # pylint: disable=too-many-locals,too-many-arguments
    points: Array,
    colours: Array,
    position: Array,
    target: Array,
    size: Tuple[int, int],
    hfov_deg: float,
    brightness: Optional[Array] = None,
    near_radius_m: float = 22.0,
) -> npt.NDArray[np.float32]:
    """Float RGB buffer (height, width, 3) of the points, before tone-mapping.

    ``brightness`` (one value per point, default 1) scales each point, which lets a sweep front
    glow brighter than the points it has already laid down.
    """
    width, height = size
    pixels, depth = project(points, position, target, size, hfov_deg)
    visible = (depth > 0.8) & (pixels[:, 0] > 1) & (pixels[:, 0] < width - 2)
    visible &= (pixels[:, 1] > 1) & (pixels[:, 1] < height - 2)
    px, z = pixels[visible], depth[visible]
    fade = (0.30 + 0.70 * np.exp(-z / 40.0))[:, None]
    gain = np.ones(len(points)) if brightness is None else brightness
    tint = np.clip(colours[visible] * 1.25 + 0.08, 0.0, 1.0) * fade * gain[visible][:, None]
    buffer = np.zeros((height, width, 3), dtype=np.float32)
    x0 = np.rint(px[:, 0]).astype(np.int64)
    y0 = np.rint(px[:, 1]).astype(np.int64)
    near = z < near_radius_m
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            weight = 1.0 if (dx == 0 and dy == 0) else 0.55
            use = near | ((dx == 0) & (dy == 0))
            xs = np.clip(x0[use] + dx, 0, width - 1)
            ys = np.clip(y0[use] + dy, 0, height - 1)
            np.add.at(buffer, (ys, xs), (tint[use] * weight).astype(np.float32))
    return buffer


def tone_map(
    buffer: npt.NDArray[np.float32], glow_sigma: float = 4.0, glow_gain: float = 1.6
) -> npt.NDArray[np.float32]:
    """Soft tone-map with a glow, so dense areas do not clip."""
    glow = gaussian_filter(buffer, sigma=(glow_sigma, glow_sigma, 0))
    return np.asarray(1.0 - np.exp(-(buffer * 0.8 + glow * glow_gain)), dtype=np.float32)


def draw_boxes(  # pylint: disable=too-many-arguments,too-many-locals
    picture: Image.Image,
    boxes: List[Tuple[Array, Any]],
    colours: dict,  # type: ignore[type-arg]
    position: Array,
    target: Array,
    hfov_deg: float,
    opacity: float = 1.0,
    min_depth_m: float = 3.0,
) -> Image.Image:
    """``picture`` with the 3D box wireframes drawn on (``opacity`` below 1 blends them in)."""
    layer = Image.new("RGB", picture.size, (0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for corners, category in boxes:
        pixels, depth = project(corners, position, target, picture.size, hfov_deg)
        if (depth < min_depth_m).any():
            continue
        for first, second in BOX_EDGES:
            draw.line(
                [tuple(pixels[first]), tuple(pixels[second])],
                fill=colours[category],
                width=2,
            )
    layer_array = np.asarray(layer, dtype=np.float32) * opacity
    base = np.asarray(picture, dtype=np.float32)
    return Image.fromarray(np.clip(np.maximum(base, layer_array), 0, 255).astype(np.uint8))
