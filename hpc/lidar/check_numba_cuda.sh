#!/bin/bash
# Run on a GPU node:  srun -p GPU --gres=gpu:1 -t 00:05:00 bash hpc/lidar/check_numba_cuda.sh
# Set KEEP_STUBS=1 to leave a CUDA "stubs" directory on LD_LIBRARY_PATH (the default removes it).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.env"
[ -f "$HERE/config.local.env" ] && source "$HERE/config.local.env"
for m in ${LIDAR_UNLOAD:-}; do module unload "$m" 2>/dev/null || true; done
for m in $MODULES ${LIDAR_MODULES:-}; do module load "$m"; done
source "$HPC_ROOT/venv-lidar/bin/activate"
export PYTHONPATH="$HPC_ROOT/repo:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
echo "LD_LIBRARY_PATH entries containing 'stubs':"
echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -i stubs || echo "  (none)"
if [ "${KEEP_STUBS:-0}" != "1" ]; then
  LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i stubs | paste -sd:)"
  export LD_LIBRARY_PATH
fi
echo "libcuda on the system:"
ldconfig -p 2>/dev/null | grep -i "libcuda.so" | head -3 || true
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || true
python -u "$HERE/lidar/check_numba_cuda.py"
