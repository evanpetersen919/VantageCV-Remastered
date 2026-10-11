"""Read KITTI 3D detection AP out of OpenPCDet's evaluation log (LiDAR study, stage 2).

OpenPCDet prints, for every class, blocks such as::

    Car AP_R40@0.70, 0.70, 0.70:
    bbox AP:96.12, 90.50, 88.71
    bev  AP:90.31, 87.80, 85.52
    3d   AP:88.20, 78.31, 76.74
    aos  AP:95.90, 90.01, 88.20

The three numbers are easy, moderate and hard difficulty. Each class has a first block at its
own strict thresholds (0.70 for car, 0.50 for pedestrian and cyclist) and a second at looser
ones; the **first AP_R40 block of each class** is the standard benchmark number and is the one
read here. The registered outcome uses the **moderate 3D AP** (``preregistration.md``, section 4).
"""

import re
from typing import Dict

CLASSES = ("Car", "Pedestrian", "Cyclist")
_HEADER = re.compile(r"^\s*(?:.*?\bINFO\s+)?(\w+) AP_R40@([\d., ]+):\s*$")
_ROW = re.compile(
    r"^\s*(?:.*?\bINFO\s+)?(bbox|bev|3d|aos)\s+AP:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)"
)


def parse_log(text: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """``{class: {kind: {"easy", "moderate", "hard"}}}`` from the first AP_R40 block of a class.

    ``kind`` is ``bbox``, ``bev``, ``3d`` or ``aos``. Classes that do not appear are absent.
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    current = None
    for line in text.splitlines():
        header = _HEADER.match(line)
        if header:
            name = header.group(1)
            current = name if name not in results else None
            if current is not None:
                results[current] = {}
            continue
        row = _ROW.match(line)
        if row and current is not None:
            kind = row.group(1)
            results[current][kind] = {
                "easy": float(row.group(2)),
                "moderate": float(row.group(3)),
                "hard": float(row.group(4)),
            }
    return results


def primary_outcome(results: Dict[str, Dict[str, Dict[str, float]]]) -> float:
    """M: the mean moderate 3D AP over Car, Pedestrian and Cyclist."""
    return sum(results[name]["3d"]["moderate"] for name in CLASSES) / len(CLASSES)
