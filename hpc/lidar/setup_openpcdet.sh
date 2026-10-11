#!/bin/bash
# One-time setup of OpenPCDet (PointPillars) for the LiDAR study (docs/lidar/preregistration.md, section 4).
# Run on the cluster login node, from the repository root:   bash hpc/lidar/setup_openpcdet.sh
#
# NOT TESTED ON A CLUSTER YET. OpenPCDet's CUDA extensions and spconv are the usual failure points; the script
# stops with a clear message at each check and never guesses a package name. Set these before running:
#   SPCONV_PACKAGE   the spconv wheel that matches the cluster's CUDA, e.g. spconv-cu118 (see the spconv README)
#   PYTHON           an interpreter OpenPCDet supports (3.8 to 3.10 is the safe range; default python3)
#   LIDAR_UNLOAD     modules to unload first when they conflict (e.g. anaconda3/2023.09_pytorch)
#   LIDAR_MODULES    modules for this environment only, set in hpc/config.local.env (e.g. "python39 cuda11.7/toolkit/11.7.1"),
#                    so the camera studies' environment is not touched
#   LIDAR_TORCH_INDEX  pip index for the CUDA build of PyTorch (default: TORCH_INDEX from hpc/config.env)
#   TORCH_VERSION / TORCHVISION_VERSION  pin both (e.g. 1.13.1 and 0.14.1, the range OpenPCDet is used with)
#   OPENPCDET_REF    git tag or commit to build (default v0.5.2, the newest tag)
#   NUMPY_VERSION    NumPy pin installed right after torch (default 1.23.5)
#   TORCH_CUDA_ARCH_LIST   GPU architectures to compile for, e.g. "8.0;8.6" (default: let torch detect)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.env"
[ -f "$HERE/config.local.env" ] && source "$HERE/config.local.env"
for m in ${LIDAR_UNLOAD:-}; do module unload "$m" 2>/dev/null || true; done   # e.g. anaconda3/2023.09_pytorch
for m in $MODULES ${LIDAR_MODULES:-}; do module load "$m"; done

: "${SPCONV_PACKAGE:?set SPCONV_PACKAGE to the spconv wheel for the CUDA version here, e.g. spconv-cu118}"
PYTHON="${PYTHON:-python3}"
OPENPCDET_REF="${OPENPCDET_REF:-v0.5.2}"   # the repository's newest tag (v0.6.0 is a README label, not a tag)
NUMPY_VERSION="${NUMPY_VERSION:-1.23.5}"   # torch 1.13 is built for NumPy 1.x; PyPI would otherwise pull NumPy 2
VENV="$HPC_ROOT/venv-lidar"
SRC="$HPC_ROOT/OpenPCDet"

echo "python: $($PYTHON --version)"
$PYTHON - <<'PY'
import sys
if sys.version_info >= (3, 11):
    print("WARNING: OpenPCDet predates Python 3.11; numba/numpy pins may not resolve. "
          "Set PYTHON to a 3.8-3.10 interpreter if the install below fails.")
PY

[ -d "$VENV" ] || $PYTHON -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel
pip install "setuptools<70"   # torch 1.13's extension builder imports pkg_resources, gone from setuptools 81+
LIDAR_TORCH_INDEX="${LIDAR_TORCH_INDEX:-$TORCH_INDEX}"
if [ -n "${TORCH_VERSION:-}" ]; then
  pip install "torch==$TORCH_VERSION" "torchvision==${TORCHVISION_VERSION:?set TORCHVISION_VERSION with TORCH_VERSION}" --index-url "$LIDAR_TORCH_INDEX"
else
  pip install torch torchvision --index-url "$LIDAR_TORCH_INDEX"
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
which nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found; load a CUDA module (set MODULES in hpc/config.local.env) so the extensions can build" >&2; exit 1; }

[ -d "$SRC" ] || git clone https://github.com/open-mmlab/OpenPCDet.git "$SRC"
git -C "$SRC" fetch --tags --quiet
git -C "$SRC" checkout --quiet "$OPENPCDET_REF" || {
  echo "ERROR: no such tag or commit: $OPENPCDET_REF. Tags available:" >&2
  git -C "$SRC" tag | tail -n 10 >&2
  exit 1
}
echo "OpenPCDet at $(git -C "$SRC" rev-parse --short HEAD) ($OPENPCDET_REF)"

pip install "$SPCONV_PACKAGE"
pip install -r "$SRC/requirements.txt"
pip install "numpy==$NUMPY_VERSION"   # the requirements pin an older NumPy; keep one that numba accepts
pip install scipy pycocotools
(cd "$SRC" && python setup.py develop)

python - <<'PY'
import pcdet
from pcdet.ops.iou3d_nms import iou3d_nms_cuda  # noqa: F401  (fails if the extension did not build)
print("pcdet", pcdet.__version__, "OK")
PY
echo "OpenPCDet is ready in $SRC (environment: $VENV)"
