#!/bin/bash
# Run on a GPU node:  srun -p GPU --gres=gpu:1 -t 00:10:00 bash hpc/lidar/check_numba_variants.sh
# Variants C and D need NVIDIA's binding: pip install cuda-python==11.8.2 in venv-lidar first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.env"
[ -f "$HERE/config.local.env" ] && source "$HERE/config.local.env"
for m in ${LIDAR_UNLOAD:-}; do module unload "$m" 2>/dev/null || true; done
for m in $MODULES ${LIDAR_MODULES:-}; do module load "$m"; done
source "$HPC_ROOT/venv-lidar/bin/activate"
python -c "import cuda; print('cuda-python present:', getattr(cuda, '__version__', 'yes'))" 2>/dev/null \
  || echo "cuda-python is NOT installed (variants C and D will fail until it is)"
python -u "$HERE/lidar/check_numba_variants.py"
