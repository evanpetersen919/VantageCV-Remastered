#!/bin/bash
# Submit the LiDAR follow-up study (docs/lidar/followup_preregistration.md): the fix chain and the
# intensity-off controls, seeds 0-4. Run from the repository root.
#
#   bash hpc/lidar/submit_lidar_followup.sh             # every arm available in the bundle
#   ARMS="E1 E2" bash hpc/lidar/submit_lidar_followup.sh # only these chain arms
#   CONTROLS=0 bash hpc/lidar/submit_lidar_followup.sh   # no intensity-off controls
#
# Chain arms (list name = result name): E1 E2 E3 E4. Controls reuse a list with the intensity column
# zeroed in every frame and write results as Ai, Bi, E0i and E3i.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../config.env"
[ -f "$HERE/../config.local.env" ] && source "$HERE/../config.local.env"
DATA_LISTS="${DATA_LISTS:-$HPC_ROOT/data/lidar_study/kitti/ImageSets}"
unset SCORE_THRESH

COUNT=0
for ARM in ${ARMS:-E1 E2 E3 E4}; do
  if [ ! -f "$DATA_LISTS/$ARM.txt" ]; then echo "skip $ARM (no list $DATA_LISTS/$ARM.txt)"; continue; fi
  for SEED in 0 1 2 3 4; do
    bash "$HERE/submit_lidar_study.sh" "lidar_${ARM}_s${SEED}" "$ARM" "$SEED"; COUNT=$((COUNT+1))
  done
done

if [ "${CONTROLS:-1}" = "1" ]; then
  # list, result name
  for PAIR in "A Ai" "B Bi" "E E0i" "E3 E3i"; do
    set -- $PAIR
    LIST="$1"; NAME="$2"
    if [ ! -f "$DATA_LISTS/$LIST.txt" ]; then echo "skip control $NAME (no list $LIST)"; continue; fi
    for SEED in 0 1 2 3 4; do
      NO_INTENSITY=1 RESULT_NAME="$NAME" \
        bash "$HERE/submit_lidar_study.sh" "lidar_${NAME}_s${SEED}" "$LIST" "$SEED"; COUNT=$((COUNT+1))
    done
  done
fi
echo "Submitted $COUNT jobs. Check with: squeue -u \$USER"
