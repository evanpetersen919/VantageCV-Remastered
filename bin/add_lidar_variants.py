"""Add synthetic-scan variants to an existing LiDAR study bundle (see followup_preregistration.md).

    PYTHONPATH=. python bin/add_lidar_variants.py --bundle F:/datasets/lidar_study \\
        --variant E1=F:/datasets/kitti_syn_E1 --variant E2=F:/datasets/kitti_syn_E2

For each ``NAME=FOLDER`` (an output of ``bin/export_lidar_kitti.py``) this links the folder's
frames into ``<bundle>/kitti/training`` and writes ``ImageSets/NAME.txt``: the real base
(``A.txt``) plus the first S frames of a seeded shuffle of the folder's frame list. The shuffle
depends only on the list's length and the seed, so two variants exported from the same rendered
frames (E0, E1, E2) get the same underlying 1,000 frames. Writes ``followup_files.txt``
(paths relative to the bundle's parent) listing every new file, for ``tar -T``.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

FOLDERS = {"velodyne": ".bin", "label_2": ".txt", "calib": ".txt", "image_2": ".png"}


def shuffled_frames(ids: List[str], seed: int) -> List[str]:
    """``ids`` in the seeded order used for every synthetic arm (seed + 1, as the study builder)."""
    order = np.random.Generator(np.random.PCG64(seed + 1)).permutation(len(ids))
    return [ids[i] for i in order]


def add_variant(  # pylint: disable=too-many-arguments,too-many-locals
    bundle: Path, name: str, folder: Path, extra: int, seed: int, new_files: List[str]
) -> int:
    """Link one variant's frames into the bundle and write its list; returns the frames added."""
    kitti = bundle / "kitti"
    ids = (folder / "ImageSets" / "synthetic.txt").read_text(encoding="utf-8").split()
    if len(ids) < extra:
        raise ValueError(f"{name}: only {len(ids)} frames, need {extra}")
    chosen = shuffled_frames(ids, seed)[:extra]
    base = (kitti / "ImageSets" / "A.txt").read_text(encoding="utf-8").split()
    for frame in chosen:
        for sub, extension in FOLDERS.items():
            source = folder / "training" / sub / f"{frame}{extension}"
            target = kitti / "training" / sub / f"{frame}{extension}"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            new_files.append(target.relative_to(bundle.parent).as_posix())
    listing = kitti / "ImageSets" / f"{name}.txt"
    listing.write_text("\n".join(sorted(base + chosen)) + "\n", encoding="utf-8")
    new_files.append(listing.relative_to(bundle.parent).as_posix())
    return len(chosen)


def main() -> None:
    """Parse arguments, add the variants, write the file list."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--extra", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    new_files: List[str] = []
    added: Dict[str, int] = {}
    for item in args.variant:
        name, _, folder = item.partition("=")
        added[name] = add_variant(args.bundle, name, Path(folder), args.extra, args.seed, new_files)
    (args.bundle / "followup_files.txt").write_text("\n".join(new_files) + "\n", encoding="utf-8")
    print(json.dumps({"added": added, "files": len(new_files)}, indent=1))


if __name__ == "__main__":
    main()
