#!/bin/bash
# Run on a GPU node:  srun -p GPU --gres=gpu:1 -t 00:05:00 bash hpc/lidar/check_numba_cuda.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.env"
[ -f "$HERE/config.local.env" ] && source "$HERE/config.local.env"
for m in ${LIDAR_UNLOAD:-}; do module unload "$m" 2>/dev/null || true; done
for m in $MODULES ${LIDAR_MODULES:-}; do module load "$m"; done
source "$HPC_ROOT/venv-lidar/bin/activate"
export PYTHONPATH="$HPC_ROOT/repo:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
python "$HERE/lidar/check_numba_cuda.py"
