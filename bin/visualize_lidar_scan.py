"""Draw a synthetic LiDAR scan: a bird's-eye view with its labels, and over the camera frame.

    PYTHONPATH=. python bin/visualize_lidar_scan.py --dataset demo/kitti_probe \\
        --export demo/kitti_probe_export --frame 0 --out demo/lidar_viz

``--dataset`` is the rendered dataset (for the colour image and its projection), ``--export`` the
output of ``bin/export_lidar_kitti.py``; ``--frame`` is the position in the export's id list. Left:
points seen from above (x forward up the page, y left), coloured by height, with the ``label_2``
boxes in the colour of their type. Right: the same points projected over the camera image,
coloured by range.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
from PIL import Image  # noqa: E402  pylint: disable=wrong-import-position

from src.export.kitti_lidar import box_corners  # noqa: E402  pylint: disable=wrong-import-position

TYPE_COLOURS = {
    "Car": "tab:red",
    "Truck": "tab:orange",
    "Misc": "tab:brown",
    "Pedestrian": "tab:blue",
    "Cyclist": "tab:green",
}


def _labels(path: Path) -> List[Dict[str, Any]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 15:
            continue
        boxes.append(
            {
                "type": f[0],
                "dimensions_hwl": [float(v) for v in f[8:11]],
                "location": [float(v) for v in f[11:14]],
                "rotation_y": float(f[14]),
            }
        )
    return boxes


def draw(  # pylint: disable=too-many-locals
    dataset: Path, export: Path, frame: int, out: Path
) -> Path:
    """Write ``<out>/frame_<id>.png`` for the ``frame``-th scan of the export."""
    ids = (export / "ImageSets" / "synthetic.txt").read_text(encoding="utf-8").split()
    sample = ids[frame]
    root = export / "training"
    scan = np.fromfile(root / "velodyne" / f"{sample}.bin", dtype=np.float32).reshape(-1, 4)
    x_v, y_v, z_v = scan[:, 0], scan[:, 1], scan[:, 2]
    boxes = _labels(root / "label_2" / f"{sample}.txt")
    data = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    entries = [i for i in data["images"] if "depth_file" in i]
    image = entries[frame]

    figure, (bev, over) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    near = x_v < 70.0
    scatter = bev.scatter(
        y_v[near], x_v[near], c=z_v[near], s=1.0, cmap="viridis", vmin=-2.0, vmax=1.5
    )
    for box in boxes:
        corners = box_corners(box)[[0, 1, 2, 3, 0]]  # the bottom face, camera frame
        bev.plot(-corners[:, 0], corners[:, 2], color=TYPE_COLOURS.get(box["type"], "k"), lw=1.5)
    bev.set_xlim(-45, 45)
    bev.set_ylim(0, 70)
    bev.set_aspect("equal")
    bev.set_xlabel("y, left (m)")
    bev.set_ylabel("x, forward (m)")
    bev.set_title(f"scan {sample}: {len(scan):,} points; boxes = labels")
    figure.colorbar(scatter, ax=bev, label="height z (m)", shrink=0.7)

    picture = Image.open(dataset / image["file_name"]).convert("RGB")
    over.imshow(picture)
    projection = np.array(image["kitti_P2"], dtype=np.float64).reshape(3, 4)
    camera = np.stack([-y_v, -z_v, x_v, np.ones_like(x_v)], axis=0)  # label frame, 4 x N
    uvw = projection @ camera
    ok = uvw[2] > 0.5
    u, v = uvw[0, ok] / uvw[2, ok], uvw[1, ok] / uvw[2, ok]
    shown = (u >= 0) & (u < picture.width) & (v >= 0) & (v < picture.height)
    over.scatter(u[shown], v[shown], c=x_v[ok][shown], s=0.6, cmap="turbo", vmin=3, vmax=60)
    over.set_title("points over the camera frame (colour = range, 3 to 60 m)")
    over.axis("off")
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"frame_{sample}.png"
    figure.tight_layout()
    figure.savefig(target, dpi=90)
    plt.close(figure)
    return target


def main() -> None:
    """Parse arguments and draw."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("demo/lidar_viz"))
    args = parser.parse_args()
    print(draw(args.dataset, args.export, args.frame, args.out))


if __name__ == "__main__":
    main()
