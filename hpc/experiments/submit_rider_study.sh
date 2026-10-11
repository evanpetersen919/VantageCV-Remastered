#!/bin/bash
# The cyclist study (docs/riders/preregistration.md, amendments 2 and 3): arms built by
# bin/prepare_rider_synthetic_arms.py from the rider_headroom bundle and rider_v18.
#   registered:   B, C, E x seeds 0-4 (15 jobs), plus A seeds 3 and 4 (A seeds 0-2 exist from the
#                 headroom check: same base, validation list and recipe)
#   exploratory:  B250, E250, B500, E500 x seeds 0-2 (12 jobs), not used for the decision
# Names hpc_rider_<arm>_s<seed>. Same recipe for every arm (YOLOv10m from scratch, 200 epochs,
# AdamW fixed, imgsz 960; hpc/run_rider_arm.sbatch). Run from the repository root.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
COUNT=0
for ARM in B C E; do
  for SEED in 0 1 2 3 4; do
    bash "$HERE/submit_rider.sh" "hpc_rider_${ARM}_s${SEED}" "$ARM" "$SEED"; COUNT=$((COUNT+1))
  done
done
for SEED in 3 4; do
  bash "$HERE/submit_rider.sh" "hpc_rider_A_s${SEED}" "A" "$SEED"; COUNT=$((COUNT+1))
done
if [ "${SWEEP:-1}" = "1" ]; then
  for ARM in B250 E250 B500 E500; do
    for SEED in 0 1 2; do
      bash "$HERE/submit_rider.sh" "hpc_rider_${ARM}_s${SEED}" "$ARM" "$SEED"; COUNT=$((COUNT+1))
    done
  done
fi
echo "Submitted $COUNT jobs. Check with: squeue -u \$USER"
