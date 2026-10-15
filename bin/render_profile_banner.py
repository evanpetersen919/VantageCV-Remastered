"""Render an ultrawide banner for a GitHub profile: one frame, shown three ways.

    PYTHONPATH=. python bin/render_profile_banner.py --dataset live_dataset/kitti_v19 \\
        --frame 12 --out docs/images/profile_banner.jpg

One frame of a rendered dataset (a street-level KITTI-profile render), cropped to an ultrawide band
and split into three bands:

* left: the render with its exact 2D boxes;
* middle: the engine's class map of the same image, with the same boxes, so the render flows into
  its labels;
* right: the 64-beam LiDAR scan of the same frame as a point cloud coloured by range, seen from
  behind and above (``bin/render_lidar_range_still.py``), with the exact 3D boxes of the objects
  that have a visible cluster of points drawn over it.

The title is overlaid lower left, in the style of the project's other images. Every pixel is the
project's own render, label or simulated scan; it makes no study data.
"""

# pylint: disable=too-many-locals,duplicate-code
# Image composition reads best as one straight run of named steps; the script loader mirrors
# the one in the other bin/ tools by design.

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.export.kitti_lidar import box_corners, points_in_box
from src.ground_truth.semantic_classes import PALETTE
from src.sensors.depth_lidar import KITTI_PATTERN
from src.sensors.lidar_render import draw_boxes, project
from src.sensors.lidar_view import labelled_boxes, load_frame

WIDTH, HEIGHT = 1920, 540  # the ultrawide band
TOP = 330  # first row kept of the 1080-row render and class map
LIDAR_TOP = 450  # first row kept of the 1080-row LiDAR picture
LIDAR_GAIN = 1.55
SENSOR_HEIGHT_M = 1.71
MIN_POINTS_IN_BOX = 150  # only boxes with a visible cluster are drawn; the rest is clutter
MAX_RANGE_M = 45.0
BOLD, REGULAR = "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"
ACCENT = (27, 175, 122)
SPLITS = (0.31, 0.62)  # where the class-map and the LiDAR bands start (share of the width)
SOFT = 60  # width in pixels of the blend between bands
BOX_RGB = {
    1: (255, 214, 64),
    2: (140, 255, 120),
    3: (70, 220, 255),
    6: (255, 100, 150),
    8: (255, 140, 66),
    10: (140, 255, 120),
}


def _sibling(name: str) -> Any:
    """A sibling script of ``bin/`` imported by file name."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _font(path: str, size: int) -> Any:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _band_masks() -> List[npt.NDArray[np.float32]]:
    """Three soft horizontal masks (left, middle, right) that sum to one at every column."""
    xs = np.arange(WIDTH, dtype=np.float32)
    first, second = SPLITS[0] * WIDTH, SPLITS[1] * WIDTH
    to_middle = np.clip((xs - (first - SOFT / 2)) / SOFT, 0.0, 1.0)
    to_right = np.clip((xs - (second - SOFT / 2)) / SOFT, 0.0, 1.0)
    masks = (1.0 - to_middle, to_middle * (1.0 - to_right), to_right)
    return [np.broadcast_to(m, (HEIGHT, WIDTH))[..., None] for m in masks]


def render_views(dataset: Path, frame: int) -> Dict[str, npt.NDArray[np.float32]]:
    """The three pictures of the ``frame``-th frame that has a depth map, cropped to the band."""
    data = json.loads((dataset / "annotations.json").read_text("utf-8"))
    image = [i for i in data["images"] if "depth_file" in i][frame]
    annotations: List[Dict[str, Any]] = [
        a for a in data["annotations"] if a["image_id"] == image["id"]
    ]
    crop = slice(TOP, TOP + HEIGHT)

    rgb = np.asarray(Image.open(dataset / image["file_name"]).convert("RGB"))
    semantic = np.asarray(Image.open(dataset / image["semantic_file"]))
    class_map = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for class_id, colour in PALETTE.items():
        class_map[semantic == class_id] = colour

    views = {}
    for key, source in (("render", rgb), ("classes", class_map)):
        picture = Image.fromarray(source[crop].copy())
        draw = ImageDraw.Draw(picture)
        for annotation in annotations if key == "render" else []:  # 2D boxes on the render only
            x, y, w, h = annotation["bbox"]
            colour = BOX_RGB.get(annotation["category_id"], (255, 255, 255))
            draw.rectangle([x, y - TOP, x + w, y + h - TOP], outline=colour, width=3)
        views[key] = np.asarray(picture, dtype=np.float32)

    # the same frame as a point cloud, seen from behind and above
    range_still = _sibling("render_lidar_range_still")
    still = range_still.build(dataset, frame, overlays=False)
    boxes = []
    scan_points = load_frame(dataset, frame, "kitti", KITTI_PATTERN).label
    for annotation in labelled_boxes(annotations):  # the exact 3D boxes, drawn over the points
        box = annotation["box3d"]
        if box["location"][2] > MAX_RANGE_M:
            continue
        if points_in_box(scan_points, box, 0.1).sum() < MIN_POINTS_IN_BOX:
            continue
        corners = box_corners(box)
        scene = np.stack([corners[:, 0], -corners[:, 1] + SENSOR_HEIGHT_M, corners[:, 2]], axis=1)
        pixels, _ = project(
            scene, range_still.POSITION, range_still.TARGET, still.size, range_still.HFOV_DEG
        )
        inside = (
            (pixels[:, 0] > SPLITS[1] * WIDTH + 12)
            & (pixels[:, 0] < WIDTH - 12)
            & (pixels[:, 1] > LIDAR_TOP + 12)
            & (pixels[:, 1] < LIDAR_TOP + HEIGHT - 12)
        )
        if inside.all():  # a box cut by the edge of the band reads as a mistake
            boxes.append((scene, annotation["category_id"]))
    still = draw_boxes(
        still, boxes, BOX_RGB, range_still.POSITION, range_still.TARGET, range_still.HFOV_DEG
    )
    band = np.asarray(still, dtype=np.float32)[LIDAR_TOP : LIDAR_TOP + HEIGHT]
    views["lidar"] = np.clip(
        band * LIDAR_GAIN, 0.0, 255.0
    )  # a point cloud needs a lift on a banner
    return views


def compose(
    views: Dict[str, npt.NDArray[np.float32]]
) -> Image.Image:  # pylint: disable=too-many-locals
    """Blend the three views into one picture and add the title, labels and dividers."""
    left, middle, right = _band_masks()
    picture = views["render"] * left + views["classes"] * middle + views["lidar"] * right
    image = Image.fromarray(np.clip(picture, 0, 255).astype(np.uint8)).convert("RGBA")

    # a dark ramp up from the bottom so the title reads on any scene
    ramp = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rows = np.clip((np.arange(HEIGHT) - (HEIGHT - 240)) / 240.0, 0.0, 1.0)
    ramp[..., 3] = (rows[:, None] ** 1.5 * 205).astype(np.uint8)
    image.alpha_composite(Image.fromarray(ramp, "RGBA"))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for share in SPLITS:  # thin dividers between the bands
        x = int(share * WIDTH)
        draw.line([(x, 0), (x, HEIGHT)], fill=ACCENT + (190,), width=2)
    label_face = _font(BOLD, 24)
    labels = ("RENDER + 2D BOXES", "EXACT CLASS MAP", "64-BEAM LIDAR")
    for share, text in zip((0.0,) + SPLITS, labels):
        x = int(share * WIDTH) + 28
        width = int(draw.textlength(text, font=label_face)) + 36
        draw.rounded_rectangle(
            [x, 24, x + width, 70],
            radius=23,
            fill=(10, 10, 10, 190),
            outline=ACCENT + (255,),
            width=2,
        )
        draw.text((x + 18, 31), text, font=label_face, fill=(255, 255, 255, 255))
    image.alpha_composite(overlay)

    title_face, sub_face = _font(BOLD, 66), _font(REGULAR, 30)
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text(
        (82, HEIGHT - 170), "VANTAGECV REMASTERED", font=title_face, fill=(0, 0, 0, 230)
    )
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(7)))
    canvas = ImageDraw.Draw(image)
    canvas.rectangle([46, HEIGHT - 160, 55, HEIGHT - 58], fill=ACCENT + (255,))
    canvas.text(
        (78, HEIGHT - 178), "VANTAGECV REMASTERED", font=title_face, fill=(255, 255, 255, 255)
    )
    canvas.text(
        (80, HEIGHT - 96),
        "Procedural driving data with pixel-exact labels and simulated LiDAR",
        font=sub_face,
        fill=(214, 220, 232, 255),
    )
    return image.convert("RGB")


def main() -> None:
    """Parse arguments, render, write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", type=Path, default=Path("live_dataset/kitti_v19"))
    parser.add_argument("--frame", type=int, default=12, help="index among frames with a depth map")
    parser.add_argument("--out", type=Path, default=Path("docs/images/profile_banner.jpg"))
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()
    picture = compose(render_views(args.dataset, args.frame))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    picture.save(args.out, quality=args.quality)
    print(f"wrote {args.out} {picture.size} ({args.out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
