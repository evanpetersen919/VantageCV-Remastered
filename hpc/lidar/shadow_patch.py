"""Remove the scene points in a pasted object's shadow during ground-truth sampling (stage 2).

Imported first by the job-local copy of ``train.py`` when ``SHADOWS=1``
(``hpc/lidar/run_lidar_arm.sbatch``). OpenPCDet's ``DataBaseSampler.add_sampled_boxes_to_scene``
removes the scene points inside the pasted boxes with ``box_utils.remove_points_in_boxes3d`` and
then adds the objects' points. This wraps that one call, only while the sampler is running, to also
drop the points whose beams the pasted boxes would have intercepted (``src/sensors/shadow.py``).
"""

from pcdet.datasets.augmentor import database_sampler  # pylint: disable=import-error
from pcdet.utils import box_utils  # pylint: disable=import-error

from src.sensors.shadow import visible_mask

_ORIGINAL_REMOVE = box_utils.remove_points_in_boxes3d
_ORIGINAL_ADD = database_sampler.DataBaseSampler.add_sampled_boxes_to_scene
_STATE = {"pasting": False}


def _remove_with_shadows(points, boxes3d):  # type: ignore[no-untyped-def]
    """Points outside the boxes, minus those in the boxes' shadows while the sampler pastes."""
    kept = _ORIGINAL_REMOVE(points, boxes3d)
    if not _STATE["pasting"] or kept.shape[0] == 0:
        return kept
    corners = box_utils.boxes_to_corners_3d(boxes3d)
    return kept[visible_mask(kept[:, :3], corners)]


def _add_with_shadows(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    """The sampler's paste step with shadow removal switched on."""
    _STATE["pasting"] = True
    try:
        return _ORIGINAL_ADD(self, *args, **kwargs)
    finally:
        _STATE["pasting"] = False


box_utils.remove_points_in_boxes3d = _remove_with_shadows
database_sampler.DataBaseSampler.add_sampled_boxes_to_scene = _add_with_shadows
