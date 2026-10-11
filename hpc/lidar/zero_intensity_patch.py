"""Zero the intensity column of every KITTI scan OpenPCDet loads (the follow-up study's controls).

Imported first by the job-local copies of ``train.py`` and ``test.py`` and by the info-file step
when ``NO_INTENSITY=1`` (``hpc/lidar/run_lidar_arm.sbatch``). It wraps ``KittiDataset.get_lidar``,
which every path (training, ground-truth database, evaluation) reads points through, so real and
synthetic scans alike carry intensity 0 and the model's fourth input feature is constant. The
point of the control: if an empty intensity channel on the synthetic side hid a gain, removing
the channel on both sides shows it.
"""

from pcdet.datasets.kitti import kitti_dataset  # pylint: disable=import-error,no-name-in-module

_ORIGINAL = kitti_dataset.KittiDataset.get_lidar


def _get_lidar_without_intensity(self, idx):  # type: ignore[no-untyped-def]
    """The scan with its intensity column set to zero."""
    points = _ORIGINAL(self, idx)
    points[:, 3] = 0.0
    return points


kitti_dataset.KittiDataset.get_lidar = _get_lidar_without_intensity
