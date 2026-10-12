#!/bin/bash
# Submit the LiDAR improvement study, stages 0 and 1 (docs/lidar/improvement_preregistration.md), seeds 0-4.
# Run from the repository root.
#
#   bash hpc/lidar/submit_lidar_improvement.sh              # every arm whose list is in the bundle
#   GROUPS_="stage1" bash hpc/lidar/submit_lidar_improvement.sh
#   SMOKE=1 bash hpc/lidar/submit_lidar_improvement.sh     # one seed (0) of every arm with EPOCHS=1
#
# Result names: S1..S4 (stage 1); Astep, E1step, E3a_rdb, S4_rdb, E0i_flag, E4_flag (stage 0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../config.env"
[ -f "$HERE/../config.local.env" ] && source "$HERE/../config.local.env"
DATA_LISTS="${DATA_LISTS:-$HPC_ROOT/data/lidar_study/kitti/ImageSets}"
unset SCORE_THRESH
GROUPS_="${GROUPS_:-stage1 stage0}"
SEEDS="0 1 2 3 4"
[ "${SMOKE:-0}" = "1" ] && SEEDS="0"

COUNT=0
run() {  # RESULT_NAME LIST EPOCHS [extra env assignments...]
  local NAME="$1" LIST="$2" EP="$3"; shift 3
  if [ ! -f "$DATA_LISTS/$LIST.txt" ]; then echo "skip $NAME (no list $LIST)"; return; fi
  for SEED in $SEEDS; do
    if [ "${SMOKE:-0}" = "1" ]; then EP=1; fi
    env "$@" RESULT_NAME="$NAME" EPOCHS="$EP" \
      bash "$HERE/submit_lidar_study.sh" "lidar_${NAME}_s${SEED}" "$LIST" "$SEED"
    COUNT=$((COUNT+1))
  done
}

for GROUP in $GROUPS_; do
  case "$GROUP" in
    stage1)
      for ARM in S1 S2 S3 S4; do run "$ARM" "$ARM" 80 X=1; done ;;
    stage0)
      run Astep A 296 X=1                      # 371 frames x 296 epochs: as many visits as 1,371 x 80
      run E1step E1 123 X=1                    # 1,371 x 123: as many visits as E1x3 (2,113 x 80)
      run E3a_rdb E3a 80 REAL_DB_ONLY=1
      run S4_rdb S4 80 REAL_DB_ONLY=1
      run E0i_flag E 80 NO_INTENSITY=1 DOMAIN_FLAG=1
      run E4_flag E4 80 DOMAIN_FLAG=1 ;;
    *) echo "unknown group $GROUP" >&2; exit 1 ;;
  esac
done
echo "Submitted $COUNT jobs. Check with: squeue -u \$USER"
