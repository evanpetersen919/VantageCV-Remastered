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
