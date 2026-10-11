#!/bin/bash
#SBATCH --job-name=vcv-matches
#SBATCH --output=%x_%j.log
# Re-score a trained cyclist-study model on BDD100K val and keep its per-detection match table
# (src/evaluation/match_table.py), which the paired bootstrap of the registered decision rule needs
# (bin/bootstrap_rider_study.py). No training: it reads runs/$NAME/weights/best.pt.
# Submit with hpc/submit_rider_matches.sh.
set -euo pipefail
: "${NAME:?}" "${HPC_CONFIG:?}"
IMGSZ="${IMGSZ:-960}"

source "$HPC_CONFIG"
[ -f "$(dirname "$HPC_CONFIG")/config.local.env" ] && source "$(dirname "$HPC_CONFIG")/config.local.env"
for m in $MODULES; do module load "$m"; done
source "$HPC_ROOT/venv/bin/activate"
cd "$HPC_ROOT/repo"
export PYTHONPATH=.
WEIGHTS="$HPC_ROOT/runs/$NAME/weights/best.pt"
[ -f "$WEIGHTS" ] || { echo "ERROR: no weights at $WEIGHTS" >&2; exit 1; }
mkdir -p "$HPC_ROOT/results/matches"

# the same settings as run_rider_arm.sbatch, so the table is the same measurement as the stored score
python bin/evaluate_detector.py --weights "$WEIGHTS" \
  --class-space ours --profile riders --benchmark bdd100k \
  --labels "$HPC_ROOT/data/bdd100k/labels/100k/val" --images "$HPC_ROOT/data/bdd100k/images/100k/val" \
  --imgsz "$IMGSZ" --conf 0.001 --iou 0.6 --max-det 100 --device 0 \
  --out "$HPC_ROOT/results/matches/${NAME}_bdd100k_riders_rescored.json" \
  --save-matches "$HPC_ROOT/results/matches/${NAME}_bdd100k_matches.npz"
echo "=== $NAME: MATCHES DONE ==="
