"""Build the arms of the LiDAR study (docs/lidar/preregistration.md, section 4), OpenPCDet layout.

    PYTHONPATH=. python bin/prepare_lidar_arms.py --kitti F:/datasets/kitti \\
        --synthetic F:/datasets/kitti_synthetic --out F:/datasets/lidar_study

Real data: the standard 3,712-frame KITTI training split (``configs/kitti_splits/train.txt``) is
drawn once, seeded, into

* ``hold``: a held-out slice used only to pick a checkpoint (never trained on);
* the real **base** (10% of the split): arm ``A``;
* an **extra pool**: everything else in the split.

Arms (the same base in each; only what is added differs):

* ``A``      the real base;
* ``B``      A + S real frames from the extra pool (random, seeded);
* ``E``      A + S synthetic frames (the first S of a seeded shuffle of the synthetic set);
* ``B<n>``, ``E<n>`` the exploratory sweep sizes ``n``.

``val`` is the standard 3,769-frame validation list, used for the final scoring only. Everything
is written under ``<out>/kitti/``: ``training/{velodyne,label_2,calib,image_2}`` holding only the
frames some list uses (hard links, so no copy on the same drive) and ``ImageSets/<arm>.txt``.
``image_2`` holds blank 1242 x 375 images (the study is LiDAR-only; OpenPCDet reads the size).
Real frames have a few slightly different true image sizes; blank 1242 x 375 is used for every
frame and arm alike, so every arm crops the same way.
"""

# pylint: disable=duplicate-code
# The argument parsing and summary tail mirror the camera studies' arm builders by design.

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

SPLIT_DIR = Path("configs/kitti_splits")
FOLDERS = ("velodyne", "label_2", "calib")
EXTENSIONS = {"velodyne": ".bin", "label_2": ".txt", "calib": ".txt"}


def select_arms(  # pylint: disable=too-many-arguments,too-many-locals
    train_ids: Sequence[str],
    synthetic_ids: Sequence[str],
    base_fraction: float,
    extra: int,
    hold: int,
    sweep: Sequence[int],
    seed: int,
) -> Dict[str, List[str]]:
    """The frame lists of every arm, ``hold`` and the real base, from one seeded shuffle."""
    order = [
        train_ids[i] for i in np.random.Generator(np.random.PCG64(seed)).permutation(len(train_ids))
    ]
    held = order[:hold]
    base_count = int(round(base_fraction * len(train_ids)))
    base = order[hold : hold + base_count]
    pool = order[hold + base_count :]
    sizes = sorted({extra, *sweep})
    if len(pool) < max(sizes):
        raise ValueError(f"only {len(pool)} extra real frames, need {max(sizes)}")
    synthetic = [
        synthetic_ids[i]
        for i in np.random.Generator(np.random.PCG64(seed + 1)).permutation(len(synthetic_ids))
    ]
    if len(synthetic) < max(sizes):
        raise ValueError(f"only {len(synthetic)} synthetic frames, need {max(sizes)}")
    arms: Dict[str, List[str]] = {"A": list(base), "hold": list(held)}
    for size in sizes:
        suffix = "" if size == extra else str(size)
        arms[f"B{suffix}"] = list(base) + pool[:size]
        arms[f"E{suffix}"] = list(base) + synthetic[:size]
    return arms


def _link(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def build(  # pylint: disable=too-many-locals,too-many-arguments
    kitti: Path,
    synthetic: Path,
    out: Path,
    base_fraction: float,
    extra: int,
    hold: int,
    sweep: Sequence[int],
    seed: int,
) -> Dict[str, Any]:
    """Select, link and list every arm; returns the counts that go in ``split.json``."""
    train_ids = (SPLIT_DIR / "train.txt").read_text(encoding="utf-8").split()
    val_ids = (SPLIT_DIR / "val.txt").read_text(encoding="utf-8").split()
    synthetic_ids = (synthetic / "ImageSets" / "synthetic.txt").read_text(encoding="utf-8").split()
    arms = select_arms(train_ids, synthetic_ids, base_fraction, extra, hold, sweep, seed)
    arms["val"] = list(val_ids)
    real_used = {i for ids in arms.values() for i in ids if i in set(train_ids) | set(val_ids)}
    synthetic_used = {i for ids in arms.values() for i in ids if i in set(synthetic_ids)}
    root = out / "kitti"
    blank = root / "_blank.png"
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1242, 375)).save(blank)
    for frame, source_root in [(f, kitti) for f in sorted(real_used)] + [
        (f, synthetic) for f in sorted(synthetic_used)
    ]:
        for folder in FOLDERS:
            name = frame + EXTENSIONS[folder]
            _link(source_root / "training" / folder / name, root / "training" / folder / name)
        _link(blank, root / "training" / "image_2" / f"{frame}.png")
    (root / "ImageSets").mkdir(exist_ok=True)
    for name, ids in arms.items():
        (root / "ImageSets" / f"{name}.txt").write_text("\n".join(sorted(ids)) + "\n", "utf-8")
    summary = {
        "seed": seed,
        "base_fraction": base_fraction,
        "counts": {name: len(ids) for name, ids in arms.items()},
        "real_frames_linked": len(real_used),
        "synthetic_frames_linked": len(synthetic_used),
    }
    (out / "split.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    """Parse arguments, build, print the counts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--kitti", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-fraction", type=float, default=0.10)
    parser.add_argument("--extra", type=int, default=1000)
    parser.add_argument("--hold", type=int, default=200)
    parser.add_argument("--sweep", type=int, nargs="*", default=[250, 500])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.kitti,
                args.synthetic,
                args.out,
                args.base_fraction,
                args.extra,
                args.hold,
                args.sweep,
                args.seed,
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
