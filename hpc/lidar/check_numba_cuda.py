"""Check, on a GPU node, the numba CUDA step of OpenPCDet's KITTI evaluation in isolation.

Run through hpc/lidar/check_numba_cuda.sh. Prints the torch and numba CUDA state, then runs the
rotated-IoU kernel that the KITTI AP calculation uses, on a few boxes. A segmentation fault or an
error here (and not in the full job) points at the numba and CUDA setup rather than the data.
"""

import faulthandler

import numpy as np
import torch

faulthandler.enable()
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))


def _loaded(name: str) -> list:  # type: ignore[type-arg]
    """Paths of libraries whose name contains ``name`` that this process has mapped."""
    with open("/proc/self/maps", encoding="utf-8") as maps:
        return sorted({line.split()[-1] for line in maps if name in line and ".so" in line})


torch.zeros(1).cuda()  # create torch's CUDA context first, as the evaluation does
print("libcuda mapped by torch:", _loaded("libcuda"))

# numba would otherwise find libcuda through the system cache, which here points at a different
# directory ("current") than the driver library torch loaded. Point numba at the one torch uses.
import os  # noqa: E402  pylint: disable=wrong-import-position

_driver = [p for p in _loaded("libcuda") if "libcuda.so" in p and "cudart" not in p]
if _driver and "NUMBA_CUDA_DRIVER" not in os.environ and os.environ.get("NO_DRIVER_FIX") != "1":
    os.environ["NUMBA_CUDA_DRIVER"] = _driver[0]
print("NUMBA_CUDA_DRIVER =", os.environ.get("NUMBA_CUDA_DRIVER"))

import numba  # noqa: E402  pylint: disable=wrong-import-position
from numba import cuda  # noqa: E402  pylint: disable=wrong-import-position

print("numba", numba.__version__, "numpy", np.__version__, "numba cuda available", cuda.is_available())
cuda.detect()

from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import (  # noqa: E402
    rotate_iou_gpu_eval,
)

boxes = np.array([[0, 0, 2, 1, 0.1], [5, 5, 2, 1, 0.5], [10, 0, 4, 2, 0.0]], dtype=np.float32)
print("rotated IoU of three boxes with themselves:")
print(rotate_iou_gpu_eval(boxes, boxes))

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval  # noqa: E402,F401

print("kitti_eval imported: OK")
