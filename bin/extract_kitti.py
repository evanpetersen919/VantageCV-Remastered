"""Extract the labelled training part of the KITTI 3D object detection zips.

    PYTHONPATH=. python bin/extract_kitti.py --root F:/datasets/kitti

Reads ``data_object_velodyne.zip``, ``data_object_calib.zip`` and ``data_object_label_2.zip`` in
``--root`` and extracts only their ``training/`` members (the ``testing/`` scans have no labels
and are not used, see docs/lidar/preregistration.md amendment 1). Files already present with the
right size are skipped, so an interrupted run can be repeated. Prints the count per folder and
fails if a count differs from the 7,481 training frames.
"""

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Dict

ARCHIVES = ("data_object_velodyne.zip", "data_object_calib.zip", "data_object_label_2.zip")
TRAINING_FRAMES = 7481
PREFIX = "training/"


def extract(root: Path) -> Dict[str, int]:
    """Extract ``training/`` of every archive under ``root``; returns files present per folder."""
    counts: Dict[str, int] = {}
    for name in ARCHIVES:
        with zipfile.ZipFile(root / name) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.startswith(PREFIX):
                    continue
                target = root / info.filename
                if target.exists() and target.stat().st_size == info.file_size:
                    pass
                else:
                    archive.extract(info, root)
                folder = Path(info.filename).parts[1]
                counts[folder] = counts.get(folder, 0) + 1
    return counts


def main() -> None:
    """Extract and check the counts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    counts = extract(args.root)
    print(counts)
    bad = {k: v for k, v in counts.items() if v != TRAINING_FRAMES}
    if bad:
        sys.exit(f"unexpected counts (want {TRAINING_FRAMES}): {bad}")


if __name__ == "__main__":
    main()
