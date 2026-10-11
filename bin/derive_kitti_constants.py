"""Derive the KITTI-side constants for the LiDAR follow-up study, from the training split only.

    PYTHONPATH=. python bin/derive_kitti_constants.py --kitti F:/datasets/kitti/training \\
        --split configs/kitti_splits/train.txt --out configs/kitti_derived_constants.json

Reads the labels (and, for intensity, a sample of the scans) of the standard 3,712-frame
training split, never the validation frames, and writes:

* ``car_length_p95_m``: the 95th percentile of KITTI ``Car`` length; a synthetic vehicle longer than
  this is what KITTI would call a ``Van`` (a class its Car evaluation ignores);
* ``range_cap_m``: per class, the 99th percentile of labelled distance (an object farther than this
  is not labelled in KITTI, so it is left unlabelled in the synthetic scans too);
* ``pedestrian_box_m``: KITTI's mean pedestrian width and length (its annotation convention is a
  larger box than the body);
* ``intensity``: per class and range band, a histogram of reflectance (0 to 1) of the points inside
  the labelled boxes of that class, and of the points inside no box ("background"), for the
  synthetic sampler to draw from.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

from src.export.kitti_lidar import points_in_box, read_calibration

CLASSES = ("Car", "Pedestrian", "Cyclist")
BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 80.0))
BINS = np.linspace(0.0, 1.0, 21)  # 20 histogram bins over reflectance 0 to 1


def derive(  # pylint: disable=too-many-locals,too-many-branches
    kitti: Path, ids: List[str], scan_frames: int
) -> Dict[str, Any]:
    """The constants as a JSON-serialisable dictionary."""
    labels: Dict[str, List[List[float]]] = {c: [] for c in CLASSES + ("Van",)}
    for frame in ids:
        for line in (kitti / "label_2" / f"{frame}.txt").read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) >= 15 and f[0] in labels:
                h, w, length = (float(v) for v in f[8:11])
                x, z = float(f[11]), float(f[13])
                labels[f[0]].append([h, w, length, float(np.hypot(x, z))])
    arrays = {c: np.array(v) for c, v in labels.items()}
    constants: Dict[str, Any] = {
        "source": "KITTI training split (configs/kitti_splits/train.txt), labels and scans",
        "frames": len(ids),
        "car_length_p95_m": float(np.percentile(arrays["Car"][:, 2], 95)),
        "van_length_mean_m": float(arrays["Van"][:, 2].mean()),
        "range_cap_m": {c: float(np.percentile(arrays[c][:, 3], 99)) for c in CLASSES},
        "pedestrian_box_m": {
            "width": float(arrays["Pedestrian"][:, 1].mean()),
            "length": float(arrays["Pedestrian"][:, 2].mean()),
        },
    }
    # reflectance histograms from a sample of scans
    pools: Dict[str, Dict[str, List[npt.NDArray[np.float32]]]] = {
        k: {f"{lo:g}-{hi:g}": [] for lo, hi in BANDS} for k in CLASSES + ("background",)
    }
    step = max(1, len(ids) // scan_frames)
    for frame in ids[::step][:scan_frames]:
        scan = np.fromfile(kitti / "velodyne" / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)
        rect, tr = read_calibration(kitti / "calib" / f"{frame}.txt")
        camera = (rect @ (tr[:, :3] @ scan[:, :3].astype(np.float64).T + tr[:, 3:4])).T
        distance = np.hypot(scan[:, 0], scan[:, 1])
        inside_any = np.zeros(len(scan), dtype=bool)
        for line in (kitti / "label_2" / f"{frame}.txt").read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) < 15 or f[0] not in CLASSES + ("Van", "Truck", "Misc"):
                continue
            box = {
                "dimensions_hwl": [float(v) for v in f[8:11]],
                "location": [float(v) for v in f[11:14]],
                "rotation_y": float(f[14]),
            }
            inside = points_in_box(camera, box, 0.1)
            inside_any |= inside
            if f[0] in CLASSES:
                for lo, hi in BANDS:
                    pick = inside & (distance >= lo) & (distance < hi)
                    if pick.any():
                        pools[f[0]][f"{lo:g}-{hi:g}"].append(scan[pick, 3])
        for lo, hi in BANDS:
            pick = ~inside_any & (distance >= lo) & (distance < hi) & (scan[:, 0] > 0)
            if pick.any():
                pools["background"][f"{lo:g}-{hi:g}"].append(scan[pick, 3][::20])
    histograms: Dict[str, Dict[str, Any]] = {}
    for kind, bands in pools.items():
        histograms[kind] = {}
        for band, chunks in bands.items():
            values = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=BINS)
            histograms[kind][band] = {
                "points": int(values.size),
                "probability": (counts / max(counts.sum(), 1)).tolist(),
            }
    constants["intensity"] = {"bin_edges": BINS.tolist(), "histograms": histograms}
    return constants


def main() -> None:
    """Parse arguments, derive, write and print the headline constants."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--kitti", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("configs/kitti_splits/train.txt"))
    parser.add_argument("--scan-frames", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("configs/kitti_derived_constants.json"))
    args = parser.parse_args()
    ids = args.split.read_text(encoding="utf-8").split()
    constants = derive(args.kitti, ids, args.scan_frames)
    args.out.write_text(json.dumps(constants, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in constants.items() if k != "intensity"}, indent=1))


if __name__ == "__main__":
    main()
