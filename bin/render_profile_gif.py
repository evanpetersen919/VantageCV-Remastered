"""Render the short showcase loop for a GitHub profile README (a GIF and a WebP).

    PYTHONPATH=. python bin/render_profile_gif.py \\
        --package F:/datasets/vantagecv_release/vantagecv_v2_riders \\
        --out docs/images/profile_showcase.gif

Seven short scenes, about ten seconds, 720 x 405 (the size of the project's other animations):

1. a generated city from above (the frames of ``docs/images/city_generation.webp``);
2. one street frame with a cyclist, as rendered;
3. the same frame with its exact instance masks;
4. its full-scene class map;
5. its metric depth;
6. a LiDAR scan coloured by range (``bin/render_lidar_range_still.py``, without its own text);
7. a closing card with the repository and the dataset.

Overlays follow ``bin/render_lidar_explainer.py`` (lower-left title with the accent bar, outlined
chip at the top left, the same grade and vignette). Every frame is the project's own render or
label; it makes no study data.
"""

# pylint: disable=protected-access,too-many-locals,duplicate-code
# The overlay helpers are the explainer script's own (private by name, shared by design).

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

TITLE = "VANTAGECV REMASTERED"
REPO = "github.com/evanpetersen919/VantageCV-Remastered"
KAGGLE = "kaggle.com/datasets/evanpetersen919/vantagecv-synthetic-driving-riders"
FRAME_NAME = "live_0324_3_rider.png"  # a day frame with riders, cars and pedestrians
HOLD = 4  # frames a still scene is held, before blending
BLEND = 3  # frames of cross-fade between scenes


def _sibling(name: str) -> Any:
    """A sibling script of ``bin/`` imported by file name."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fit(picture: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return picture.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def street_pictures(package: Path, size: Tuple[int, int]) -> List[Image.Image]:
    """Render, instance masks, class map and depth of one packaged frame, at ``size``."""
    coco = json.loads((package / "annotations" / "instances_val.json").read_text("utf-8"))
    image = next(i for i in coco["images"] if Path(i["file_name"]).name == FRAME_NAME)
    classes = json.loads((package / "classes.json").read_text("utf-8"))
    palette = {c["id"]: tuple(c["color_rgb"]) for c in classes["semantic_classes"]}

    rgb_full = Image.open(package / image["file_name"]).convert("RGB")
    rgb = np.asarray(rgb_full)

    overlay = rgb.astype(np.float64).copy()
    rng = np.random.default_rng(3)
    for annotation in coco["annotations"]:
        if annotation["image_id"] != image["id"]:
            continue
        mask = mask_utils.decode(
            {
                "size": annotation["mask_rle"]["size"],
                "counts": annotation["mask_rle"]["counts"].encode(),
            }
        ).astype(bool)
        tint = rng.integers(80, 255, 3)
        overlay[mask] = 0.35 * overlay[mask] + 0.65 * tint

    semantic = np.asarray(Image.open(package / image["semantic_file"]))
    class_map = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for class_id, colour in palette.items():
        class_map[semantic == class_id] = colour

    depth = np.asarray(Image.open(package / image["depth_file"])).astype(np.float64)
    depth /= float(image["depth_scale"])
    valid = depth > 0
    value = np.zeros_like(depth)
    span = np.log(80.0) - np.log(3.0)
    value[valid] = 1.0 - np.clip((np.log(depth[valid]) - np.log(3.0)) / span, 0.0, 1.0)
    depth_rgb = (colormaps["turbo"](value)[..., :3] * 255).astype(np.uint8)
    depth_rgb[~valid] = (10, 10, 24)

    return [
        _fit(rgb_full, size),
        _fit(Image.fromarray(overlay.astype(np.uint8)), size),
        _fit(Image.fromarray(class_map), size),
        _fit(Image.fromarray(depth_rgb), size),
    ]


def closing_card(module: Any, size: Tuple[int, int]) -> Image.Image:
    """The last scene: the project name, three facts, the repository and the dataset."""
    width, height = size
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top, bottom = np.array([8, 10, 18]), np.array([16, 22, 40])
    gradient = (top * (1 - rows) + bottom * rows)[:, None, :]
    card = Image.fromarray(np.repeat(gradient, width, axis=1).astype(np.uint8))
    draw = ImageDraw.Draw(card)
    bold, regular = module.BOLD, module.REGULAR
    draw.rectangle([44, 70, 49, 124], fill=module.ACCENT)
    draw.text((66, 62), TITLE, font=module._font(bold, 40), fill=(255, 255, 255))
    draw.text(
        (68, 112),
        "Synthetic driving data with pixel-exact labels",
        font=module._font(regular, 21),
        fill=(205, 210, 220),
    )
    facts = [
        "Procedural cities, traffic, pedestrians and cyclists in Unreal Engine 5",
        "64-beam LiDAR matched to a real automotive sensor",
        "Public Kaggle dataset: 1,399 frames, masks, depth, class maps",
        "1,200+ tests and a documented experiment log",
    ]
    for i, fact in enumerate(facts):
        y = 188 + i * 34
        draw.ellipse([68, y + 9, 78, y + 19], fill=module.ACCENT)
        draw.text((92, y), fact, font=module._font(regular, 19), fill=(235, 238, 245))
    draw.text((68, 338), REPO, font=module._font(bold, 20), fill=(255, 255, 255))
    draw.text((68, 366), KAGGLE, font=module._font(regular, 15), fill=(160, 168, 185))
    return card


def build(  # pylint: disable=too-many-locals
    package: Path, images: Path, lidar_dataset: Path, lidar_frame: int
) -> List[List[Image.Image]]:
    """The loop as one group of frames per scene (each starts with its cross-fade)."""
    module = _sibling("render_lidar_explainer")
    size = module.SIZE
    street = street_pictures(package, size)
    lidar = _fit(
        _sibling("render_lidar_range_still").build(lidar_dataset, lidar_frame, overlays=False),
        size,
    )

    city_source = Image.open(images / "city_generation.webp")
    city: List[Image.Image] = []
    for index in range(14, min(44, getattr(city_source, "n_frames", 1)), 2):
        city_source.seek(index)
        city.append(_fit(city_source.copy(), size))

    vignette = module._vignette()

    def overlaid(picture: Image.Image, chip: str, subtitle: str) -> Image.Image:
        base = picture.convert("RGBA")
        base.alpha_composite(module._scaled(module.caption_layer(TITLE, subtitle)))
        base.alpha_composite(module._scaled(module.chip_layer(chip)))
        graded = module._grade(np.asarray(base.convert("RGB"), dtype=np.float32), vignette)
        return Image.fromarray(graded.astype(np.uint8))

    scenes: List[List[Image.Image]] = [
        [
            overlaid(
                p, "PROCEDURAL CITIES", "A new city from every seed, rendered in Unreal Engine 5"
            )
            for p in city
        ],
        [overlaid(street[0], "RENDERED FRAME", "Cars, pedestrians and cyclists, by day and night")]
        * (HOLD + 3),
        [overlaid(street[1], "EXACT MASKS", "Labels read from the engine, not a model")]
        * (HOLD + 3),
        [overlaid(street[2], "CLASS MAP", "Every pixel classified")] * (HOLD + 2),
        [overlaid(street[3], "METRIC DEPTH", "Distance to the nearest surface at every pixel")]
        * (HOLD + 2),
        [overlaid(lidar, "64-BEAM LIDAR", "Matched to a real automotive sensor")] * (HOLD + 6),
        [closing_card(module, size)] * (HOLD + 8),
    ]
    groups: List[List[Image.Image]] = []
    previous = scenes[0][0]
    for index, scene in enumerate(scenes):
        group: List[Image.Image] = []
        if index > 0:  # cross-fade from the previous scene's last frame
            for step in range(1, BLEND + 1):
                group.append(Image.blend(previous, scene[0], step / (BLEND + 1)))
        group.extend(scene)
        groups.append(group)
        previous = scene[-1]
    black = Image.new("RGB", size, (0, 0, 0))
    groups[-1].extend(  # fade out, so the loop restarts cleanly
        Image.blend(previous, black, step / (BLEND + 1)) for step in range(1, BLEND + 1)
    )
    return groups


def save_gif(groups: List[List[Image.Image]], path: Path, duration_ms: int, colours: int) -> None:
    """One looping GIF with a palette of its own for each scene (no cross-scene posterising)."""
    converted: List[Image.Image] = []
    for group in groups:
        sample = group[:: max(1, len(group) // 6)]
        montage = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for k, picture in enumerate(sample):
            montage.paste(picture, (0, k * sample[0].height))
        palette = montage.quantize(
            colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        converted += [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in group]
    converted[0].save(
        path,
        save_all=True,
        append_images=converted[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    """Parse arguments, render, write the GIF and the WebP."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--package", type=Path, required=True, help="the packaged dataset folder")
    parser.add_argument("--images", type=Path, default=Path("docs/images"))
    parser.add_argument("--lidar-dataset", type=Path, default=Path("live_dataset/kitti_v19"))
    parser.add_argument("--lidar-frame", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("docs/images/profile_showcase.gif"))
    parser.add_argument("--duration-ms", type=int, default=100)
    parser.add_argument("--colours", type=int, default=160)
    args = parser.parse_args()
    module = _sibling("render_lidar_explainer")
    groups = build(args.package, args.images, args.lidar_dataset, args.lidar_frame)
    frames = [frame for group in groups for frame in group]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(groups, args.out, args.duration_ms, args.colours)
    module.save_webp(frames, args.out.with_suffix(".webp"), args.duration_ms, 80)
    seconds = len(frames) * args.duration_ms / 1000.0
    for path in (args.out, args.out.with_suffix(".webp")):
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{len(frames)} frames, {seconds:.1f} s")


if __name__ == "__main__":
    main()
