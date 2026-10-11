#!/bin/bash
# Submit the match-table jobs of the cyclist study's bootstrap: arms B and E, seeds 0-4 (10 jobs).
#   bash hpc/submit_rider_matches.sh
# Each job re-scores runs/hpc_rider_<arm>_s<seed>/weights/best.pt on BDD100K val and writes
# results/matches/hpc_rider_<arm>_s<seed>_bdd100k_matches.npz. Run from the repository root.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.env"
[ -f "$HERE/config.local.env" ] && source "$HERE/config.local.env"
COUNT=0
for ARM in B E; do
  for SEED in 0 1 2 3 4; do
    NAME="hpc_rider_${ARM}_s${SEED}"
    ARGS=(--partition "$SLURM_PARTITION" --gres "$SLURM_GRES" --cpus-per-task "$SLURM_CPUS"
          --mem "$SLURM_MEM" --time "${EVAL_TIME:-03:00:00}" --job-name "matches_${ARM}${SEED}"
          --chdir "$HPC_ROOT/logs")
    [ -n "$SLURM_ACCOUNT" ] && ARGS+=(--account "$SLURM_ACCOUNT")
    sbatch "${ARGS[@]}" --export=ALL,HPC_CONFIG="$HERE/config.env",NAME="$NAME" \
      "$HERE/rider_matches.sbatch"
    COUNT=$((COUNT+1))
  done
done
echo "Submitted $COUNT jobs. Check with: squeue -u \$USER"
