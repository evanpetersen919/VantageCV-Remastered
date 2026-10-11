"""Render the "how the LiDAR works" animation (a GIF for the README) from one rendered frame.

    PYTHONPATH=. python bin/render_lidar_explainer.py --dataset live_dataset/kitti_v18 \\
        --frame 716 --out docs/images/lidar_how_it_works.gif

Five short scenes from one frame of a rendered dataset, using the study's own sensor pattern
(``KITTI_PATTERN``: 64 beams, 81.6 degrees across):

1. the engine renders the scene;
2. ... and its depth, the distance to the nearest surface at every pixel;
3. the 64-beam pattern sweeps across the image and reads that depth along its fixed angles;
4. the camera pulls back from the sensor into 3D: every sample is a point, coloured by class;
5. the labelled 3D boxes appear and the view sways: the scan and its labels, as exported.

Text and overlays follow the project's other README animations (``demo/compose_demo.py``): the
lower-left title with the accent bar and soft shadow, the outlined pill chip at the top left, the
class key at the top right, the same S-curve grade and vignette, and the same label colours
(person yellow, car cyan, truck orange, bus pink). The overlays are laid out at 1920 x 1080 as
there and scaled to the 720 x 405 of ``vehicle_annotations.webp``.

It is a visualisation, drawn with ``src/sensors/lidar_render.py``; it makes no study data.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.export.kitti_lidar import box_corners
from src.ground_truth import semantic_classes as sem
from src.sensors.depth_lidar import KITTI_PATTERN
from src.sensors.lidar_render import draw_boxes, splat, tone_map
from src.sensors.lidar_view import labelled_boxes, load_frame

SIZE = (720, 405)  # as vehicle_annotations.webp
FULL = (1920, 1080)  # the layout size of the project's overlays
SENSOR_HEIGHT_M = 1.71
FIRST_PERSON_HFOV = 106.3
BOLD, REGULAR = "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"
ACCENT = (27, 175, 122)
LEGEND = [
    ("person", (255, 214, 64)),
    ("car", (70, 220, 255)),
    ("truck", (255, 140, 66)),
    ("bus", (255, 100, 150)),
]
CYCLIST_RGB = (140, 255, 120)
TITLE = "LIDAR FROM THE ENGINE'S DEPTH"
STEPS = {
    1: ("RENDERED FRAME", "The engine renders the scene"),
    2: ("METRIC DEPTH", "The distance to the nearest surface, at every pixel"),
    3: ("64-BEAM SWEEP", "A 64-beam sensor samples that depth along fixed angles"),
    4: ("3D POINT CLOUD", "Every sample becomes a 3D point carrying its object's class"),
    5: ("3D BOXES", "The scan and its exact 3D boxes, in KITTI format"),
}
# label colours of the legend, by Cityscapes class id and by riders-profile category id
POINT_RGB = {
    sem.PERSON: (255, 214, 64),
    sem.RIDER: CYCLIST_RGB,
    sem.BICYCLE: CYCLIST_RGB,
    sem.CAR: (70, 220, 255),
    sem.TRUCK: (255, 140, 66),
    sem.BUS: (255, 100, 150),
}
BACKGROUND_RGB = (150, 158, 176)
BOX_RGB = {1: LEGEND[0][1], 3: LEGEND[1][1], 8: LEGEND[2][1], 6: LEGEND[3][1], 10: CYCLIST_RGB}


def _font(path: str, size: int) -> Any:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _shadowed(
    layer: Image.Image, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, face: Any
) -> None:
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((xy[0] + 2, xy[1] + 3), text, font=face, fill=(0, 0, 0, 200))
    layer.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(5)))
    draw.text(xy, text, font=face, fill=(255, 255, 255, 255))


def caption_layer(title: str, subtitle: str) -> Image.Image:
    """The lower-left caption on a dark gradient (as ``demo/compose_demo.caption_layer``)."""
    width, height = FULL
    layer = Image.new("RGBA", FULL, (0, 0, 0, 0))
    ramp = np.zeros((height, width, 4), dtype=np.uint8)
    rows = np.clip((np.arange(height) - (height - 380)) / 380.0, 0.0, 1.0)
    ramp[..., 3] = (rows[:, None] ** 1.4 * 175).astype(np.uint8)
    layer.alpha_composite(Image.fromarray(ramp, "RGBA"))
    draw = ImageDraw.Draw(layer)
    draw.rectangle([84, height - 190, 92, height - 108], fill=ACCENT + (255,))
    _shadowed(layer, draw, (112, height - 202), title, _font(BOLD, 58))
    _shadowed(layer, draw, (114, height - 128), subtitle, _font(REGULAR, 30))
    return layer


def chip_layer(text: str, colour: Tuple[int, int, int] = ACCENT) -> Image.Image:
    """The outlined pill label at the top left (as ``demo/compose_demo.chip_layer``)."""
    layer = Image.new("RGBA", FULL, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    face = _font(BOLD, 30)
    width = int(draw.textlength(text, font=face)) + 56
    draw.rounded_rectangle(
        [72, 64, 72 + width, 124],
        radius=30,
        fill=(10, 10, 10, 190),
        outline=colour + (255,),
        width=3,
    )
    draw.text((100, 71), text, font=face, fill=(255, 255, 255, 255))
    return layer


def legend_layer() -> Image.Image:
    """The class colour key at the top right (as ``demo/compose_demo.legend_layer``)."""
    layer = Image.new("RGBA", FULL, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    face = _font(BOLD, 26)
    x = FULL[0] - 72
    for name, colour in reversed(LEGEND):
        width = int(draw.textlength(name, font=face)) + 60
        x -= width
        draw.rounded_rectangle(
            [x, 64, x + width - 10, 118],
            radius=27,
            fill=(10, 10, 10, 190),
            outline=colour + (255,),
            width=3,
        )
        draw.text((x + 26, 71), name, font=face, fill=colour + (255,))
    return layer


def _scaled(layer: Image.Image) -> Image.Image:
    return layer.resize(SIZE, Image.Resampling.LANCZOS)


def _grade(
    frame: npt.NDArray[np.float32], vignette: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """The project's mild S-curve and vignette, applied to every frame alike."""
    x = frame / 255.0
    x = x + 0.10 * (x - 0.5) * (1.0 - np.abs(2.0 * x - 1.0))
    return np.asarray(np.clip(x * vignette[..., None], 0.0, 1.0) * 255.0, dtype=np.float32)


def _vignette() -> npt.NDArray[np.float32]:
    width, height = SIZE
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    r2 = ((xs - width / 2) / (width / 2)) ** 2 + ((ys - height / 2) / (height / 2)) ** 2
    return np.asarray(1.0 - 0.20 * np.clip(r2 / 2.0, 0.0, 1.0), dtype=np.float32)


def _alpha_scaled(layer: Image.Image, alpha: float) -> Image.Image:
    out = layer.copy()
    band = np.asarray(out.getchannel("A"), dtype=np.float32) * min(max(alpha, 0.0), 1.0)
    out.putalpha(Image.fromarray(band.astype(np.uint8)))
    return out


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
    span = np.log(80.0) - np.log(3.0)
    value[valid] = 1.0 - np.clip((np.log(depth[valid]) - np.log(3.0)) / span, 0, 1)
    rgb = (colormaps["turbo"](value)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = (10, 10, 24)
    return Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)


def _point_colours(class_ids: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """Legend colours for the labelled classes, muted tones for the background classes."""
    rgb = np.tile(np.array(BACKGROUND_RGB, dtype=np.float64), (len(class_ids), 1))
    rgb[class_ids == sem.ROAD] = (84, 94, 124)
    rgb[class_ids == sem.VEGETATION] = (110, 160, 120)
    rgb[class_ids == sem.TERRAIN] = (110, 160, 120)
    for class_id, colour in POINT_RGB.items():
        rgb[class_ids == class_id] = colour
    return np.asarray(rgb / 255.0)


def build(  # pylint: disable=too-many-locals,too-many-statements
    dataset: Path, frame: int, fps_scale: float = 1.0
) -> List[Image.Image]:
    """All animation frames at ``SIZE``, with the project's overlays and grade."""
    data = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    image = [i for i in data["images"] if "depth_file" in i][frame]
    loaded = load_frame(dataset, frame, "kitti", KITTI_PATTERN)
    source = Image.open(dataset / image["file_name"]).convert("RGB")
    rgb = source.resize(SIZE, Image.Resampling.LANCZOS)
    depth_picture = _depth_picture(dataset, image, SIZE)
    scale = SIZE[0] / float(image["width"])

    label = loaded.label
    points = np.stack([label[:, 0], -label[:, 1] + SENSOR_HEIGHT_M, label[:, 2]], axis=1)
    colours = _point_colours(loaded.class_ids)
    road = loaded.class_ids == sem.ROAD
    gain_points = np.where(road, 0.30, 0.95)  # dense near-field ground rings must not clip
    gain_points = np.where(label[:, 2] < 8.0, gain_points * 0.35, gain_points)
    azimuth = np.degrees(np.arctan2(label[:, 0], label[:, 2]))
    pixel = np.stack([loaded.pixel_x * scale, loaded.pixel_y * scale], axis=1)
    boxes = []
    for annotation in labelled_boxes(loaded.annotations):
        if annotation["category_id"] not in BOX_RGB:
            continue
        c = box_corners(annotation["box3d"])
        scene = np.stack([c[:, 0], -c[:, 1] + SENSOR_HEIGHT_M, c[:, 2]], axis=1)
        boxes.append((scene, annotation["category_id"]))

    vignette = _vignette()
    legend = _scaled(legend_layer())
    chips = {k: _scaled(chip_layer(v[0])) for k, v in STEPS.items()}
    captions = {k: _scaled(caption_layer(TITLE, v[1])) for k, v in STEPS.items()}

    def finish(picture: Image.Image, step: int, local: int, total: int) -> Image.Image:
        """Overlays (fading in and out within the scene), then the grade."""
        fade = min(1.0, local / 6.0) * min(1.0, (total - 1 - local) / 4.0 + 0.4)
        base = picture.convert("RGBA")
        for layer in (captions[step], chips[step], legend):
            base.alpha_composite(_alpha_scaled(layer, fade))
        graded = _grade(np.asarray(base.convert("RGB"), dtype=np.float32), vignette)
        return Image.fromarray(graded.astype(np.uint8))

    def count(n: int) -> int:
        return max(2, int(round(n * fps_scale)))

    frames: List[Image.Image] = []
    black = Image.new("RGB", SIZE, (0, 0, 0))
    # 1: the rendered frame fades in
    n = count(18)
    for k in range(n):
        frames.append(finish(_blend(black, rgb, k / 6.0), 1, k, n))
    # 2: cross-fade to the depth map
    n = count(18)
    for k in range(n):
        frames.append(finish(_blend(rgb, depth_picture, k / 7.0), 2, k, n))

    # 3: the sweep over the (dimmed) frame
    dim = Image.eval(rgb, lambda v: int(v * 0.40))
    low, high = azimuth.min() - 0.5, azimuth.max() + 0.5
    n = count(36)
    base_array = np.asarray(dim, dtype=np.float32) / 255.0
    half_width = SIZE[0] / 2.0 / np.tan(np.radians(FIRST_PERSON_HFOV / 2))
    for k in range(n):
        front = low + (high - low) * _ease(k / (n - 1.0))
        revealed = azimuth <= front
        near_front = revealed & (azimuth > front - 2.0)
        layer = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.float32)
        px = np.rint(pixel[revealed]).astype(np.int64)
        gain = np.where(near_front[revealed], 2.2, 1.0) * gain_points[revealed]
        tint = np.clip(colours[revealed] * 1.15 + 0.05, 0, 1) * gain[:, None]
        for dx in (0, 1):
            for dy in (0, 1):
                xs = np.clip(px[:, 0] + dx, 0, SIZE[0] - 1)
                ys = np.clip(px[:, 1] + dy, 0, SIZE[1] - 1)
                np.maximum.at(layer, (ys, xs), tint.astype(np.float32))
        lit = np.clip(layer, 0, 1)
        glow = tone_map(layer * 0.5, 3.0)
        mixed = base_array * (1.0 - 0.5 * lit.max(axis=2, keepdims=True)) + lit * 0.9 + glow * 0.35
        out = Image.fromarray((np.clip(mixed, 0, 1) * 255).astype(np.uint8))
        x_front = SIZE[0] / 2.0 + np.tan(np.radians(front)) * half_width
        ImageDraw.Draw(out).line([(x_front, 0), (x_front, SIZE[1])], fill=(120, 230, 255), width=2)
        frames.append(finish(out, 3, k, n))

    # 4: pull back from the sensor into 3D
    start_pos = np.array([0.0, SENSOR_HEIGHT_M, 0.0])
    start_tgt = np.array([0.0, SENSOR_HEIGHT_M, 10.0])
    end_pos, end_tgt = np.array([-5.5, 5.2, -14.0]), np.array([0.5, 1.0, 20.0])
    n = count(40)
    for k in range(n):
        t = _ease(k / (n - 1.0))
        position = start_pos + (end_pos - start_pos) * t
        target = start_tgt + (end_tgt - start_tgt) * t
        hfov = FIRST_PERSON_HFOV + (60.0 - FIRST_PERSON_HFOV) * t
        picture = tone_map(
            splat(points, colours, position, target, SIZE, hfov, gain_points), glow_gain=0.7
        )
        out = Image.fromarray((picture * 255).astype(np.uint8))
        if t < 0.25:  # the camera frame dissolves as the view leaves the sensor
            ghost = Image.fromarray((np.clip(base_array * 0.6, 0, 1) * 255).astype(np.uint8))
            out = _blend(ghost, out, t / 0.25)
        out = draw_boxes(
            out, boxes, BOX_RGB, position, target, hfov, opacity=_ease((t - 0.6) / 0.4)
        )
        frames.append(finish(out, 4, k, n))

    # 5: sway with the labelled boxes
    n = count(44)
    for k in range(n):
        angle = np.radians(26.0 * np.sin(2.0 * np.pi * k / n))
        offset = end_pos - end_tgt
        rotated = np.array(
            [
                offset[0] * np.cos(angle) + offset[2] * np.sin(angle),
                offset[1] * (1.0 - 0.18 * np.cos(2.0 * np.pi * k / n)),
                -offset[0] * np.sin(angle) + offset[2] * np.cos(angle),
            ]
        )
        position = end_tgt + rotated
        picture = tone_map(
            splat(points, colours, position, end_tgt, SIZE, 60.0, gain_points), glow_gain=0.7
        )
        out = Image.fromarray((picture * 255).astype(np.uint8))
        out = draw_boxes(out, boxes, BOX_RGB, position, end_tgt, 60.0)
        frames.append(finish(out, 5, k, n))
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
    converted: List[Image.Image] = []
    for segment in (frames[:split], frames[split:]):
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


def save_webp(frames: List[Image.Image], path: Path, duration_ms: int, quality: int) -> None:
    """Write ``frames`` as one looping animated WebP (the format of the other README animations)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=quality,
        method=4,
    )


def main() -> None:
    """Parse arguments, render, write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=716)
    parser.add_argument("--out", type=Path, default=Path("docs/images/lidar_how_it_works.gif"))
    parser.add_argument("--duration-ms", type=int, default=70)
    parser.add_argument("--colours", type=int, default=256)
    parser.add_argument("--webp-quality", type=int, default=82)
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
    webp = args.out.with_suffix(".webp")
    save_webp(frames, webp, args.duration_ms, args.webp_quality)
    print(webp, f"{webp.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
