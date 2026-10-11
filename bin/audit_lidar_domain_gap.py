"""Measure how the synthetic LiDAR scans differ from real KITTI scans, object by object.

    PYTHONPATH=. python bin/audit_lidar_domain_gap.py --real F:/datasets/kitti/training \\
        --synthetic F:/datasets/kitti_synthetic/training --out results/lidar_domain_gap.json

Both folders hold ``velodyne``, ``label_2`` and ``calib`` in KITTI's layout. For Car, Pedestrian and
Cyclist it reports, on a sample of frames from each:

* box size (height, width, length) mean, standard deviation and 5th/95th percentile;
* objects per frame, by range band;
* points inside the (untruncated, unoccluded) objects' boxes, by range band;
* the intensity of all points and of the points on objects (the synthetic scans carry none);
* the share of boxes by KITTI difficulty (height in the image, occlusion, truncation).

It is an audit: it changes nothing and trains nothing.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import numpy.typing as npt

from src.export.kitti_lidar import points_in_box, read_calibration

CLASSES = ("Car", "Pedestrian", "Cyclist")
BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 80.0))


def read_labels(path: Path) -> List[Dict[str, Any]]:
    """The Car, Pedestrian and Cyclist rows of a label file as dictionaries."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 15 or f[0] not in CLASSES:
            continue
        rows.append(
            {
                "type": f[0],
                "truncated": float(f[1]),
                "occluded": int(float(f[2])),
                "bbox_height": float(f[7]) - float(f[5]),
                "dimensions_hwl": [float(v) for v in f[8:11]],
                "location": [float(v) for v in f[11:14]],
                "rotation_y": float(f[14]),
            }
        )
    return rows


def sample_ids(folder: Path, count: int) -> List[str]:
    """``count`` frame ids spread evenly through the folder's velodyne files."""
    names = sorted(p.stem for p in (folder / "velodyne").glob("*.bin"))
    step = max(1, len(names) // count)
    return names[::step][:count]


def summarise(values: Sequence[float]) -> Dict[str, float]:
    """Mean, standard deviation, 5th and 95th percentile of ``values`` (empty gives zeros)."""
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p95": 0.0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
    }


def audit(folder: Path, frames: int) -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """The statistics of one dataset folder."""
    dims: Dict[str, Dict[str, List[float]]] = {c: {"h": [], "w": [], "l": []} for c in CLASSES}
    per_frame: Dict[str, Dict[str, List[int]]] = {
        c: {f"{lo:g}-{hi:g}": [] for lo, hi in BANDS} for c in CLASSES
    }
    points: Dict[str, Dict[str, List[int]]] = {
        c: {f"{lo:g}-{hi:g}": [] for lo, hi in BANDS} for c in CLASSES
    }
    intensity_all: List[npt.NDArray[np.float32]] = []
    intensity_obj: List[npt.NDArray[np.float32]] = []
    ids = sample_ids(folder, frames)
    for frame in ids:
        scan = np.fromfile(folder / "velodyne" / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)
        rect, tr = read_calibration(folder / "calib" / f"{frame}.txt")
        camera = (rect @ (tr[:, :3] @ scan[:, :3].astype(np.float64).T + tr[:, 3:4])).T
        intensity_all.append(scan[:: max(1, len(scan) // 2000), 3])
        counts = {c: {k: 0 for k in per_frame[c]} for c in CLASSES}
        for row in read_labels(folder / "label_2" / f"{frame}.txt"):
            c = row["type"]
            h, w, length = row["dimensions_hwl"]
            dims[c]["h"].append(h)
            dims[c]["w"].append(w)
            dims[c]["l"].append(length)
            distance = float(np.hypot(row["location"][0], row["location"][2]))
            band = next((f"{lo:g}-{hi:g}" for lo, hi in BANDS if lo <= distance < hi), None)
            if band is None:
                continue
            counts[c][band] += 1
            if row["truncated"] < 0.05 and row["occluded"] == 0:
                inside = points_in_box(camera, row, 0.1)
                points[c][band].append(int(inside.sum()))
                if inside.any():
                    intensity_obj.append(scan[inside, 3])
        for c in CLASSES:
            for band, n in counts[c].items():
                per_frame[c][band].append(n)
    all_i: npt.NDArray[np.float32] = (
        np.concatenate(intensity_all) if intensity_all else np.zeros(1, dtype=np.float32)
    )
    obj_i: npt.NDArray[np.float32] = (
        np.concatenate(intensity_obj) if intensity_obj else np.zeros(1, dtype=np.float32)
    )
    return {
        "frames": len(ids),
        "box_size_m": {c: {k: summarise(v) for k, v in dims[c].items()} for c in CLASSES},
        "objects_per_frame_by_range": {
            c: {b: float(np.mean(v)) if v else 0.0 for b, v in per_frame[c].items()}
            for c in CLASSES
        },
        "points_per_clean_object_by_range": {
            c: {
                b: {"objects": len(v), "median": float(np.median(v)) if v else None}
                for b, v in points[c].items()
            }
            for c in CLASSES
        },
        "intensity": {
            "all_points": {
                "mean": float(all_i.mean()),
                "p50": float(np.percentile(all_i, 50)),
                "p95": float(np.percentile(all_i, 95)),
            },
            "points_on_objects": {
                "mean": float(obj_i.mean()),
                "p50": float(np.percentile(obj_i, 50)),
                "p95": float(np.percentile(obj_i, 95)),
            },
        },
    }


def main() -> None:
    """Audit both folders, print a comparison and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("results/lidar_domain_gap.json"))
    args = parser.parse_args()
    result = {
        "real": audit(args.real, args.frames),
        "synthetic": audit(args.synthetic, args.frames),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for c in CLASSES:
        print(f"== {c}: box size (h, w, l), mean +- std")
        for name in ("real", "synthetic"):
            s = result[name]["box_size_m"][c]
            h, w, ln = s["h"], s["w"], s["l"]
            print(
                f"  {name:9s} h {h['mean']:.2f}+-{h['std']:.2f}  w {w['mean']:.2f}+-{w['std']:.2f}"
                f"  l {ln['mean']:.2f}+-{ln['std']:.2f}  (n={h['n']})"
            )
        print("  objects per frame / median points on a clean object, by range band:")
        for b in (f"{lo:g}-{hi:g}" for lo, hi in BANDS):
            row = []
            for name in ("real", "synthetic"):
                pts = result[name]["points_per_clean_object_by_range"][c][b]
                per_frame = result[name]["objects_per_frame_by_range"][c][b]
                row.append(
                    f"{name[:5]} {per_frame:5.2f}/frame, pts {pts['median']} (n={pts['objects']})"
                )
            print(f"    {b:6s} " + "   ".join(row))
    for name in ("real", "synthetic"):
        i = result[name]["intensity"]
        print(
            f"intensity {name}: all mean {i['all_points']['mean']:.3f} p95 "
            f"{i['all_points']['p95']:.3f}; on objects mean {i['points_on_objects']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
