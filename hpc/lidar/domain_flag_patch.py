"""Append a domain flag to every KITTI scan OpenPCDet loads (stage 0 of the improvement study).

Imported first by the job-local copies of ``train.py`` and ``test.py`` and by the info-file step
when ``DOMAIN_FLAG=1`` (``hpc/lidar/run_lidar_arm.sbatch``). It wraps ``KittiDataset.get_lidar``
and adds a fifth column: 0 for a real KITTI frame (ids below 100000) and 1 for a synthetic one.
The job's config copies list ``flag`` as a fifth point feature. The point of the arm: with a
perfect domain marker available as an input, does the network protect real pedestrians, as the
all-zero intensity of the synthetic scans appeared to let it (docs/lidar/improvement_plan.md)?
"""

import numpy as np
from pcdet.datasets.kitti import kitti_dataset  # pylint: disable=import-error,no-name-in-module

SYNTHETIC_FROM_ID = 100000

_ORIGINAL = kitti_dataset.KittiDataset.get_lidar


def _get_lidar_with_flag(self, idx):  # type: ignore[no-untyped-def]
    """The scan with a fifth column: 1 for a synthetic frame, 0 for a real one."""
    points = _ORIGINAL(self, idx)
    flag = np.full((points.shape[0], 1), float(int(idx) >= SYNTHETIC_FROM_ID), dtype=np.float32)
    return np.concatenate([points.astype(np.float32), flag], axis=1)


kitti_dataset.KittiDataset.get_lidar = _get_lidar_with_flag
