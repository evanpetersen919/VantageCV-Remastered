#!/bin/bash
# Submit the LiDAR study (docs/lidar/preregistration.md, section 4), or one smoke-test job.
#
#   smoke test (1 epoch, arm A, seed 0):   EPOCHS=1 bash hpc/lidar/submit_lidar_study.sh smoke A 0
#   the registered arms, A B E x seeds 0-4 (15 jobs):   bash hpc/lidar/submit_lidar_study.sh
#   plus the exploratory sweep (B/E 250 and 500 x seeds 0-2, 12 jobs):   SWEEP=1 bash hpc/lidar/submit_lidar_study.sh
#
# Job names are lidar_<arm>_s<seed>. Run from the repository root.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.env"
[ -f "$HERE/config.local.env" ] && source "$HERE/config.local.env"

submit() {  # NAME ARM SEED
  local ARGS=(--partition "$SLURM_PARTITION" --gres "$SLURM_GRES" --cpus-per-task "$SLURM_CPUS"
              --mem "$SLURM_MEM" --time "$SLURM_TIME" --job-name "$1" --chdir "$HPC_ROOT/logs")
  [ -n "$SLURM_ACCOUNT" ] && ARGS+=(--account "$SLURM_ACCOUNT")
  sbatch "${ARGS[@]}" \
    --export=ALL,HPC_CONFIG="$HERE/config.env",NAME="$1",ARM="$2",SEED="$3",EPOCHS="${EPOCHS:-80}" \
    "$HERE/lidar/run_lidar_arm.sbatch"
}

if [ "$#" -eq 3 ]; then
  submit "$1" "$2" "$3"
  exit 0
fi

COUNT=0
for ARM in A B E; do
  for SEED in 0 1 2 3 4; do
    submit "lidar_${ARM}_s${SEED}" "$ARM" "$SEED"; COUNT=$((COUNT+1))
  done
done
if [ "${SWEEP:-0}" = "1" ]; then
  for ARM in B250 E250 B500 E500; do
    for SEED in 0 1 2; do
      submit "lidar_${ARM}_s${SEED}" "$ARM" "$SEED"; COUNT=$((COUNT+1))
    done
  done
fi
echo "Submitted $COUNT jobs. Check with: squeue -u \$USER"
