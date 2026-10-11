"""Render the "how the LiDAR works" animation (a GIF for the README) from one rendered frame.

    PYTHONPATH=. python bin/render_lidar_explainer.py --dataset live_dataset/kitti_v18 \\
        --frame 716 --out docs/images/lidar_how_it_works.gif

Five short scenes from one frame of a rendered dataset, using the study's own sensor pattern
(``KITTI_PATTERN``: 64 beams, 81.6 degrees across):

1. the engine renders the scene;
2. ... and its depth, the distance to the nearest surface at every pixel;
3. the 64-beam pattern sweeps across the image and reads that depth along its fixed angles;
4. the camera pulls back from the sensor into 3D: every sample is a point with a class colour;
5. the labelled 3D boxes appear and the view sways: the scan and its labels, as exported.

It is a visualisation, drawn with ``src/sensors/lidar_render.py``; it makes no study data.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageFont

from src.export.kitti_lidar import box_corners
from src.sensors.depth_lidar import KITTI_PATTERN
from src.sensors.lidar_render import draw_boxes, splat, tone_map
from src.sensors.lidar_view import BOX_COLOURS, labelled_boxes, load_frame

SIZE = (800, 450)
SENSOR_HEIGHT_M = 1.71
FIRST_PERSON_HFOV = 106.3
CAPTIONS = {
    1: "The engine renders the scene",
    2: "...and the depth of every pixel: the distance to the nearest surface",
    3: "A 64-beam sensor samples that depth along fixed angles, sweeping across",
    4: "Every sample becomes a 3D point carrying its object's class",
    5: "Output: a KITTI-format point cloud with exact 3D boxes",
}
BOX_RGB = {k: tuple(int(c * 255) for c in to_rgb(v)) for k, v in BOX_COLOURS.items()}


def _font(size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _caption(picture: Image.Image, step: int, alpha: float = 1.0) -> Image.Image:
    """The step number and caption on a dark strip along the bottom."""
    out = picture.copy().convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    height = 46
    draw.rectangle(
        [0, out.height - height, out.width, out.height], fill=(0, 0, 0, int(170 * alpha))
    )
    draw.text(
        (16, out.height - height + 8),
        str(step),
        font=_font(26),
        fill=(255, 190, 60, int(255 * alpha)),
    )
    draw.text(
        (44, out.height - height + 12),
        CAPTIONS[step],
        font=_font(19),
        fill=(255, 255, 255, int(255 * alpha)),
    )
    return Image.alpha_composite(out, layer).convert("RGB")


def _ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blend(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    return Image.blend(a, b, min(max(t, 0.0), 1.0))


def _depth_picture(dataset: Path, image: Dict[str, Any], size: Tuple[int, int]) -> Image.Image:
    depth = np.asarray(Image.open(dataset / image["depth_file"])).astype(np.float64)
    depth /= float(image["depth_scale"])
    valid = depth > 0
    value = np.zeros_like(depth)
    value[valid] = 1.0 - np.clip(
        (np.log(depth[valid]) - np.log(3.0)) / (np.log(80.0) - np.log(3.0)), 0, 1
    )
    rgb = (colormaps["turbo"](value)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = (10, 10, 24)
    return Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)


def build(  # pylint: disable=too-many-locals,too-many-statements
    dataset: Path, frame: int, fps_scale: float = 1.0
) -> List[Image.Image]:
    """All animation frames."""
    data = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    image = [i for i in data["images"] if "depth_file" in i][frame]
    loaded = load_frame(dataset, frame, "kitti", KITTI_PATTERN)
    rgb = (
        Image.open(dataset / image["file_name"])
        .convert("RGB")
        .resize(SIZE, Image.Resampling.LANCZOS)
    )
    depth_picture = _depth_picture(dataset, image, SIZE)
    scale = SIZE[0] / float(image["width"])

    label = loaded.label
    points = np.stack([label[:, 0], -label[:, 1] + SENSOR_HEIGHT_M, label[:, 2]], axis=1)
    colours = loaded.colours
    road = np.all(np.abs(colours - np.array([128, 64, 128]) / 255.0) < 0.02, axis=1)
    gain_points = np.where(road, 0.40, 0.95)  # dense near-field ground rings must not clip
    gain_points = np.where(label[:, 2] < 8.0, gain_points * 0.55, gain_points)  # very near surfaces
    azimuth = np.degrees(np.arctan2(label[:, 0], label[:, 2]))
    pixel = np.stack([loaded.pixel_x * scale, loaded.pixel_y * scale], axis=1)
    boxes = []
    for annotation in labelled_boxes(loaded.annotations):
        c = box_corners(annotation["box3d"])
        scene = np.stack([c[:, 0], -c[:, 1] + SENSOR_HEIGHT_M, c[:, 2]], axis=1)
        boxes.append((scene, annotation["category_id"]))

    frames: List[Image.Image] = []

    def count(n: int) -> int:
        return max(2, int(round(n * fps_scale)))

    black = Image.new("RGB", SIZE, (0, 0, 0))
    # 1: the rendered frame fades in
    for k in range(count(16)):
        frames.append(_caption(_blend(black, rgb, k / 6.0), 1, min(1.0, k / 5.0)))
    # 2: cross-fade to the depth map
    for k in range(count(18)):
        frames.append(_caption(_blend(rgb, depth_picture, k / 8.0), 2, 1.0))

    # 3: the sweep over the (dimmed) frame
    dim = Image.eval(rgb, lambda v: int(v * 0.40))
    low, high = azimuth.min() - 0.5, azimuth.max() + 0.5
    sweep_frames = count(36)
    base = np.asarray(_blend(depth_picture, dim, 1.0), dtype=np.float32) / 255.0
    for k in range(sweep_frames):
        front = low + (high - low) * _ease(k / (sweep_frames - 1.0))
        revealed = azimuth <= front
        near_front = revealed & (azimuth > front - 2.0)
        layer = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.float32)
        px = np.rint(pixel[revealed]).astype(np.int64)
        gain = np.where(near_front[revealed], 2.4, 1.0)
        tint = (
            np.clip(colours[revealed] * 1.3 + 0.1, 0, 1) * (gain * gain_points[revealed])[:, None]
        )
        for dx in (0, 1):
            for dy in (0, 1):
                xs = np.clip(px[:, 0] + dx, 0, SIZE[0] - 1)
                ys = np.clip(px[:, 1] + dy, 0, SIZE[1] - 1)
                np.maximum.at(layer, (ys, xs), tint.astype(np.float32))
        lit = np.clip(layer, 0, 1)
        glow = tone_map(layer * 0.5, 3.0)
        picture = np.clip(
            base * (1.0 - 0.5 * lit.max(axis=2, keepdims=True)) + lit * 0.9 + glow * 0.35, 0, 1
        )
        out = Image.fromarray((picture * 255).astype(np.uint8))
        x_front = SIZE[0] / 2.0 + (
            np.tan(np.radians(front)) * (SIZE[0] / 2.0) / np.tan(np.radians(FIRST_PERSON_HFOV / 2))
        )
        ImageDraw.Draw(out).line(
            [(x_front, 0), (x_front, SIZE[1] - 46)], fill=(120, 230, 255), width=2
        )
        frames.append(_caption(out, 3, 1.0))

    # 4: pull back from the sensor into 3D
    start_pos, start_tgt = np.array([0.0, SENSOR_HEIGHT_M, 0.0]), np.array(
        [0.0, SENSOR_HEIGHT_M, 10.0]
    )
    end_pos, end_tgt = np.array([-5.5, 5.2, -14.0]), np.array([0.5, 1.0, 20.0])
    pull = count(40)
    for k in range(pull):
        t = _ease(k / (pull - 1.0))
        position = start_pos + (end_pos - start_pos) * t
        target = start_tgt + (end_tgt - start_tgt) * t
        hfov = FIRST_PERSON_HFOV + (60.0 - FIRST_PERSON_HFOV) * t
        picture = tone_map(splat(points, colours, position, target, SIZE, hfov, gain_points))
        out = Image.fromarray((picture * 255).astype(np.uint8))
        if t < 0.25:  # the camera frame dissolves as the view leaves the sensor
            out = _blend(
                Image.fromarray((np.clip(base * 0.9 + picture * 1.2, 0, 1) * 255).astype(np.uint8)),
                out,
                t / 0.25,
            )
        out = draw_boxes(
            out, boxes, BOX_RGB, position, target, hfov, opacity=_ease((t - 0.6) / 0.4)
        )
        frames.append(_caption(out, 4, 1.0))

    # 5: sway with the labelled boxes
    sway = count(44)
    for k in range(sway):
        angle = np.radians(26.0 * np.sin(2.0 * np.pi * k / sway))
        offset = end_pos - end_tgt
        rotated = np.array(
            [
                offset[0] * np.cos(angle) + offset[2] * np.sin(angle),
                offset[1] * (1.0 - 0.18 * np.cos(2.0 * np.pi * k / sway)),
                -offset[0] * np.sin(angle) + offset[2] * np.cos(angle),
            ]
        )
        position = end_tgt + rotated
        picture = tone_map(splat(points, colours, position, end_tgt, SIZE, 60.0, gain_points))
        out = Image.fromarray((picture * 255).astype(np.uint8))
        out = draw_boxes(out, boxes, BOX_RGB, position, end_tgt, 60.0)
        frames.append(_caption(out, 5, 1.0))
    # loop: fade back to black
    last = frames[-1]
    for k in range(count(8)):
        frames.append(_blend(last, black, (k + 1) / count(8)))
    return frames


def _palette(pictures: List[Image.Image], colours: int) -> Image.Image:
    """One adaptive palette for ``pictures`` (sampled evenly)."""
    sample = pictures[:: max(1, len(pictures) // 16)]
    montage = Image.new("RGB", (SIZE[0], SIZE[1] * len(sample)))
    for k, picture in enumerate(sample):
        montage.paste(picture, (0, k * SIZE[1]))
    return montage.quantize(
        colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )


def save_gif(frames: List[Image.Image], path: Path, duration_ms: int, colours: int) -> None:
    """Write ``frames`` as one looping GIF.

    The two photographic scenes (the rendered frame and its depth map) get their own rich palette;
    the dark point-cloud scenes share another, so each segment is free of flicker and posterising.
    """
    split = int(len(frames) * 0.22)
    segments = [frames[:split], frames[split:]]
    converted: List[Image.Image] = []
    for segment in segments:
        palette = _palette(segment, colours)
        converted += [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in segment]
    path.parent.mkdir(parents=True, exist_ok=True)
    converted[0].save(
        path,
        save_all=True,
        append_images=converted[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    """Parse arguments, render, write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=716)
    parser.add_argument("--out", type=Path, default=Path("docs/images/lidar_how_it_works.gif"))
    parser.add_argument("--duration-ms", type=int, default=70)
    parser.add_argument("--colours", type=int, default=256)
    parser.add_argument("--still", type=int, default=None, help="write only this frame as PNG")
    args = parser.parse_args()
    frames = build(args.dataset, args.frame)
    if args.still is not None:
        still: Optional[Path] = args.out.with_suffix(".png")
        assert still is not None
        frames[args.still].save(still)
        print(still, len(frames), "frames")
        return
    save_gif(frames, args.out, args.duration_ms, args.colours)
    print(args.out, f"{args.out.stat().st_size / 1e6:.1f} MB", len(frames), "frames")


if __name__ == "__main__":
    main()
