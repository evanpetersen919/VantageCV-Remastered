"""Merge two OpenPCDet ground-truth-sampling databases (docs/lidar/insertion_preregistration.md).

``create_kitti_infos`` writes, under a data root, ``gt_database/<id>_<name>_<k>.bin`` (the points of
one object, relative to its box centre, float32 ``x y z intensity``) and ``kitti_dbinfos_train.pkl``
(``{class name: [info, ...]}``, each info holding the ``path`` of its ``.bin`` relative to the root
and the box ``box3d_lidar``). ``merge_object_databases`` copies the objects of an extra root into a
target root and appends their infos, so the sampler can paste real and synthetic objects from one
database. Frame ids make the file names unique, so nothing collides.

With ``constants`` (``configs/kitti_derived_constants.json``) the copied objects' intensity is
drawn per point from KITTI's class and range conditioned histograms
(``src/sensors/lidar_intensity.py``); otherwise it is left as written.
"""

import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.ground_truth import semantic_classes as sem
from src.sensors.lidar_intensity import sample_intensity

DB_INFOS = "kitti_dbinfos_train.pkl"
DB_FOLDER = "gt_database"
CLASS_ID = {"Car": sem.CAR, "Pedestrian": sem.PERSON, "Cyclist": sem.RIDER}


def _load(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("rb") as handle:
        loaded: Dict[str, List[Dict[str, Any]]] = pickle.load(handle)
    return loaded


def _copy_with_reflectance(  # pylint: disable=too-many-arguments
    source: Path,
    destination: Path,
    info: Dict[str, Any],
    name: str,
    constants: Dict[str, Any],
    seed: int,
) -> None:
    """Copy one object's points with its intensity drawn from the class and range histograms."""
    points = np.fromfile(source, dtype=np.float32).reshape(-1, 4)
    centre = np.asarray(info["box3d_lidar"][:3], dtype=np.float64)
    ranges = np.hypot(points[:, 0] + centre[0], points[:, 1] + centre[1])
    class_ids = np.full(points.shape[0], CLASS_ID[name], dtype=np.uint8)
    points[:, 3] = sample_intensity(class_ids, ranges, constants, seed=seed)
    points.tofile(destination)


def merge_object_databases(
    target: Path,
    extra: Path,
    constants: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, int]:
    """Add the objects of the database under ``extra`` to the one under ``target``.

    Returns the number of objects added per class name. With ``constants``, an added object's
    intensity is redrawn (class from its name; range from its centre plus the point offset).
    """
    merged = _load(target / DB_INFOS)
    added: Dict[str, int] = {}
    rng_seed = seed
    for name, infos in _load(extra / DB_INFOS).items():
        for info in infos:
            source = extra / info["path"]
            destination = target / info["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if constants is not None and name in CLASS_ID:
                rng_seed += 1
                _copy_with_reflectance(source, destination, info, name, constants, rng_seed)
            else:
                shutil.copyfile(source, destination)
            merged.setdefault(name, []).append(info)
            added[name] = added.get(name, 0) + 1
    with (target / DB_INFOS).open("wb") as handle:
        pickle.dump(merged, handle)
    return added
