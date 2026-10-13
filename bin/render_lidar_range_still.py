"""Render the README's LiDAR still: one scan coloured by distance, seen from behind and above.

    PYTHONPATH=. python bin/render_lidar_range_still.py --dataset live_dataset/kitti_v19 \\
        --frame 12 --out docs/images/lidar_range_view.jpg

Every point of the scan is coloured by its range from the sensor (near is bright yellow-white, far
fades through orange and magenta to deep purple), over faint range arcs at 10, 20, 40, 60 and 80 m
on the ground. It deliberately shows what ``bin/render_lidar_explainer.py`` does not: the whole scan
as a range picture, with no class colours and no boxes.

Text and overlays follow the project's other README images (the lower-left title with the accent
bar and soft shadow, the outlined pill chip), laid out at 1920 x 1080 and not scaled. It is a
visualisation; it makes no study data.
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Any, List

import numpy as np
import numpy.typing as npt
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter

from src.sensors.depth_lidar import KITTI_PATTERN
from src.sensors.lidar_render import project, splat, tone_map
from src.sensors.lidar_view import load_frame

SIZE = (1920, 1080)
SENSOR_HEIGHT_M = 1.71
NEAR_M, FAR_M = 4.0, 80.0
ARCS_M = (10, 20, 40, 60)
POSITION = np.array([0.0, 6.2, -11.0])
TARGET = np.array([0.0, 1.2, 24.0])
HFOV_DEG = 56.0
HALF_FIELD_DEG = 40.8


def _explainer() -> Any:
    """The explainer script as a module, for its fonts and overlay layers."""
    path = Path(__file__).resolve().parent / "render_lidar_explainer.py"
    spec = importlib.util.spec_from_file_location("render_lidar_explainer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def range_colours(ranges: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Near is bright, far is deep purple; the far end stays visible against black."""
    span = np.log(FAR_M) - np.log(NEAR_M)
    value = 1.0 - np.clip((np.log(np.maximum(ranges, NEAR_M)) - np.log(NEAR_M)) / span, 0.0, 1.0)
    return np.asarray(colormaps["inferno"](0.30 + 0.68 * value)[:, :3])


def key_layer(module: Any) -> Image.Image:
    """The distance key at the top right: a gradient strip between two end labels."""
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    face = module._font(module.BOLD, 26)  # pylint: disable=protected-access
    left, top, width, height = SIZE[0] - 72 - 520, 64, 520, 54
    draw.rounded_rectangle(
        [left, top, left + width, top + height],
        radius=27,
        fill=(10, 10, 10, 190),
        outline=(255, 255, 255, 120),
        width=2,
    )
    strip_left, strip_right = left + 118, left + width - 112
    for x in range(strip_left, strip_right):
        t = (x - strip_left) / float(strip_right - strip_left - 1)
        rgb = tuple(int(c * 255) for c in colormaps["inferno"](0.98 - 0.78 * t)[:3])
        draw.line([(x, top + 18), (x, top + height - 18)], fill=rgb + (255,))
    draw.text((left + 26, top + 8), f"{int(NEAR_M)} m", font=face, fill=(255, 255, 255, 255))
    draw.text((strip_right + 14, top + 8), f"{int(FAR_M)} m", font=face, fill=(255, 255, 255, 255))
    return layer


def arcs_layer(module: Any) -> Image.Image:
    """Faint range arcs on the ground with their distances, projected from the same camera."""
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    face = module._font(module.REGULAR, 26)  # pylint: disable=protected-access
    angles = np.radians(np.linspace(-HALF_FIELD_DEG, HALF_FIELD_DEG, 120))
    for metres in ARCS_M:
        ground = np.stack(
            [metres * np.sin(angles), np.zeros_like(angles), metres * np.cos(angles)], 1
        )
        pixels, depth = project(ground, POSITION, TARGET, SIZE, HFOV_DEG)
        if (depth <= 0.8).any():
            continue
        draw.line([tuple(p) for p in pixels], fill=(255, 255, 255, 62), width=2)
        x, y = pixels[-1]
        draw.text((x + 12, y - 16), f"{metres} m", font=face, fill=(255, 255, 255, 150))
    return layer


def _grade(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """The project's mild S-curve and vignette."""
    width, height = SIZE
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    r2 = ((xs - width / 2) / (width / 2)) ** 2 + ((ys - height / 2) / (height / 2)) ** 2
    vignette = 1.0 - 0.22 * np.clip(r2 / 2.0, 0.0, 1.0)
    x = frame / 255.0
    x = x + 0.10 * (x - 0.5) * (1.0 - np.abs(2.0 * x - 1.0))
    return np.asarray(np.clip(x * vignette[..., None], 0.0, 1.0) * 255.0, dtype=np.float32)


def background() -> npt.NDArray[np.float32]:
    """A very dark blue-black gradient, a little lighter towards the horizon."""
    height, width = SIZE[1], SIZE[0]
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top = np.array([6, 8, 16], dtype=np.float32)
    bottom = np.array([14, 18, 34], dtype=np.float32)
    column = top[None, :] * (1 - rows) + bottom[None, :] * rows
    return np.repeat(column[:, None, :], width, axis=1) / 255.0


def build(dataset: Path, frame: int) -> Image.Image:  # pylint: disable=too-many-locals
    """The finished 1920 x 1080 picture."""
    module = _explainer()
    loaded = load_frame(dataset, frame, "kitti", KITTI_PATTERN)
    label = loaded.label
    points = np.stack([label[:, 0], -label[:, 1] + SENSOR_HEIGHT_M, label[:, 2]], axis=1)
    colours = range_colours(np.linalg.norm(label, axis=1))
    road = loaded.class_ids == 7  # Cityscapes road: dense near-field rings must not clip
    gain = np.where(road, 0.55, 1.0)
    gain = np.where(label[:, 2] < 9.0, gain * 0.35, gain)
    buffer = splat(points, colours, POSITION, TARGET, SIZE, HFOV_DEG, gain, near_radius_m=45.0)
    scan = tone_map(buffer, glow_sigma=5.0, glow_gain=1.5)
    base = np.clip(background() + scan * 1.05, 0.0, 1.0)
    picture = Image.fromarray((base * 255).astype(np.uint8)).convert("RGBA")
    layers: List[Image.Image] = [
        arcs_layer(module),
        key_layer(module),
        module.chip_layer("RANGE VIEW"),
        module.caption_layer(
            "THE SCAN, SEEN IN RANGE",
            "64 beams, 81.6 degrees across: each point coloured by its distance",
        ),
    ]
    for layer in layers:
        picture.alpha_composite(layer)
    graded = _grade(np.asarray(picture.convert("RGB"), dtype=np.float32))
    return Image.fromarray(graded.astype(np.uint8)).filter(ImageFilter.UnsharpMask(1.2, 40, 2))


def main() -> None:
    """Parse arguments and write the picture."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("docs/images/lidar_range_view.jpg"))
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.dataset, args.frame).save(args.out, quality=args.quality)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
