#!/bin/bash
#SBATCH --job-name=vcv-lidar
#SBATCH --output=%x_%j.log
# One arm of the LiDAR study (docs/lidar/preregistration.md, section 4): train PointPillars on one arm's
# frames, pick the checkpoint on the held-out slice, score it on KITTI's validation frames, and write
# results/lidar_<ARM>_s<SEED>_kitti_val.json (read by bin/analyze_lidar_study.py).
# Submit with hpc/lidar/submit_lidar_study.sh. NOT TESTED ON A CLUSTER YET: run a smoke test first
# (EPOCHS=1 bash hpc/lidar/submit_lidar_study.sh smoke A 0).
set -euo pipefail
: "${NAME:?}" "${ARM:?}" "${SEED:?}" "${HPC_CONFIG:?}"
EPOCHS="${EPOCHS:-80}"            # OpenPCDet's KITTI PointPillars default
SELECT_LAST="${SELECT_LAST:-5}"   # last epochs considered for checkpoint selection on the held-out slice
# Smoke tests only: a one-epoch model predicts nothing at the standard 0.1 score threshold, and OpenPCDet's
# numba IoU kernel crashes on zero detections. SCORE_THRESH=0.0 makes it emit boxes. Unset for real runs.
EXTRA_SETS=""
[ -n "${SCORE_THRESH:-}" ] && EXTRA_SETS="MODEL.POST_PROCESSING.SCORE_THRESH ${SCORE_THRESH}"
CFG="tools/cfgs/kitti_models/pointpillar.yaml"

source "$HPC_CONFIG"
[ -f "$(dirname "$HPC_CONFIG")/config.local.env" ] && source "$(dirname "$HPC_CONFIG")/config.local.env"
for m in ${LIDAR_UNLOAD:-}; do module unload "$m" 2>/dev/null || true; done   # e.g. anaconda3/2023.09_pytorch
for m in $MODULES ${LIDAR_MODULES:-}; do module load "$m"; done
source "$HPC_ROOT/venv-lidar/bin/activate"
export PYTHONPATH="$HPC_ROOT/repo:${PYTHONPATH:-}"
SRC="$HPC_ROOT/OpenPCDet"
DATA="$HPC_ROOT/data/lidar_study/kitti"
[ -d "$DATA/training" ] || { echo "ERROR: $DATA/training is missing (upload the lidar_study bundle first)" >&2; exit 1; }
[ -f "$DATA/ImageSets/$ARM.txt" ] || { echo "ERROR: unknown arm $ARM (no $DATA/ImageSets/$ARM.txt)" >&2; exit 1; }

WORK="${SLURM_TMPDIR:-${TMPDIR:-$HPC_ROOT/work}}/vcv-lidar-${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$WORK"' EXIT
# Two roots over the same frames: one where "val" is the held-out slice (checkpoint selection) and
# one where "val" is KITTI's validation list (final scoring). Training lists are the arm's frames.
for ROLE in select final; do
  ROOT="$WORK/$ROLE"
  mkdir -p "$ROOT/ImageSets"
  ln -s "$DATA/training" "$ROOT/training"
  cp "$DATA/ImageSets/$ARM.txt" "$ROOT/ImageSets/train.txt"
  if [ "$ROLE" = select ]; then cp "$DATA/ImageSets/hold.txt" "$ROOT/ImageSets/val.txt"
  else cp "$DATA/ImageSets/val.txt" "$ROOT/ImageSets/val.txt"; fi
  : > "$ROOT/ImageSets/test.txt"   # create_kitti_infos also builds a test split; an empty list is enough
done

cd "$SRC"
# OpenPCDet's create_kitti_infos CLI hard-codes data/kitti under the repo; call the function with our paths.
for ROLE in select final; do
python - "$WORK/$ROLE" <<'PY'
import sys
from pathlib import Path
import yaml
from easydict import EasyDict
import pcdet.datasets.kitti.kitti_dataset as kitti_dataset
from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos
# this commit imports Path only under its own __main__, but create_groundtruth_database uses it
kitti_dataset.Path = Path
root = Path(sys.argv[1])
cfg = EasyDict(yaml.safe_load(open("tools/cfgs/dataset_configs/kitti_dataset.yaml")))
cfg.DATA_PATH = str(root)
create_kitti_infos(dataset_cfg=cfg, class_names=["Car", "Pedestrian", "Cyclist"],
                   data_path=root, save_path=root)
PY
done

# OpenPCDet's train.py fixes its seed at 666 (two places) with --fix_random_seed; use a job-local copy with ours.
cp tools/train.py "tools/train_${NAME}.py"
sed -i "s/666/${SEED}/g" "tools/train_${NAME}.py"
if grep -q "666" "tools/train_${NAME}.py" || ! grep -q "random_seed(${SEED}" "tools/train_${NAME}.py"; then
  echo "ERROR: seed patch did not apply (train.py changed upstream?)" >&2; exit 1
fi

# KITTI's road-plane files (planes/) are a separate download that is not part of the object-detection
# archives, and the synthetic frames have none. Ground-truth sampling then places pasted objects at their
# original heights (USE_ROAD_PLANE False); the same for every arm (docs/lidar/preregistration.md, amendment 6).
cd tools
LOCAL="${NAME}"
sed "s/USE_ROAD_PLANE: True/USE_ROAD_PLANE: False/" cfgs/dataset_configs/kitti_dataset.yaml   > "cfgs/dataset_configs/kitti_dataset_${LOCAL}.yaml"
sed -e "s/USE_ROAD_PLANE: True/USE_ROAD_PLANE: False/"     -e "s#cfgs/dataset_configs/kitti_dataset.yaml#cfgs/dataset_configs/kitti_dataset_${LOCAL}.yaml#"     cfgs/kitti_models/pointpillar.yaml > "cfgs/kitti_models/pointpillar_${LOCAL}.yaml"
grep -q "USE_ROAD_PLANE: True" "cfgs/kitti_models/pointpillar_${LOCAL}.yaml" "cfgs/dataset_configs/kitti_dataset_${LOCAL}.yaml"   && { echo "ERROR: road plane still enabled in the local config" >&2; exit 1; }
trap 'rm -rf "$WORK" "train_${NAME}.py" "cfgs/kitti_models/pointpillar_${LOCAL}.yaml" "cfgs/dataset_configs/kitti_dataset_${LOCAL}.yaml"' EXIT
export PYTHONFAULTHANDLER=1   # a segmentation fault then prints the Python stack that led to it
# numba 0.56 built-in CUDA driver layer segfaults creating a context against this cluster's 580-series driver
# (hpc/lidar/check_numba_variants.py: only the NVIDIA cuda-python binding works). The KITTI evaluation's
# rotated-IoU kernel is a numba CUDA kernel, so every job uses the binding.
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
# OpenPCDet resumes from any checkpoint already in its output folder: start every job clean.
rm -rf "$SRC/output/kitti_models/pointpillar_${LOCAL}/${NAME}"
echo "=== $NAME: training PointPillars (arm $ARM, seed $SEED, $EPOCHS epochs, $(wc -l < "$WORK/select/ImageSets/train.txt") frames) ==="
python "train_${NAME}.py" --cfg_file "cfgs/kitti_models/pointpillar_${LOCAL}.yaml" --extra_tag "$NAME" \
  --fix_random_seed --ckpt_save_interval 1 --max_ckpt_save_num "$SELECT_LAST" --workers 4 \
  --set OPTIMIZATION.NUM_EPOCHS "$EPOCHS" DATA_CONFIG.DATA_PATH "$WORK/select" $EXTRA_SETS

OUT="$SRC/output/kitti_models/pointpillar_${LOCAL}/$NAME"   # OpenPCDet writes under its repo root, not tools/
echo "=== $NAME: choosing the checkpoint on the held-out slice ==="
BEST=""; BEST_M=-1
for EP in $(seq $((EPOCHS - SELECT_LAST + 1)) "$EPOCHS"); do
  CK="$OUT/ckpt/checkpoint_epoch_${EP}.pth"
  [ -f "$CK" ] || continue
  python test.py --cfg_file "cfgs/kitti_models/pointpillar_${LOCAL}.yaml" --extra_tag "$NAME" --ckpt "$CK" \
    --set DATA_CONFIG.DATA_PATH "$WORK/select" $EXTRA_SETS > "$WORK/select_${EP}.log" 2>&1
  M=$(python -c "
import sys
from src.evaluation.openpcdet_results import parse_log, primary_outcome
print(primary_outcome(parse_log(open('$WORK/select_${EP}.log').read())))")
  echo "epoch $EP held-out M = $M"
  if python -c "import sys; sys.exit(0 if $M > $BEST_M else 1)"; then BEST="$CK"; BEST_M="$M"; BEST_EP="$EP"; fi
done
[ -n "$BEST" ] || { echo "ERROR: no checkpoint was evaluated" >&2; exit 1; }

echo "=== $NAME: scoring epoch $BEST_EP on KITTI validation ==="
python test.py --cfg_file "cfgs/kitti_models/pointpillar_${LOCAL}.yaml" --extra_tag "$NAME" --ckpt "$BEST" \
  --set DATA_CONFIG.DATA_PATH "$WORK/final" $EXTRA_SETS > "$WORK/final.log" 2>&1
mkdir -p "$HPC_ROOT/results"
python - "$WORK/final.log" "$HPC_ROOT/results/lidar_${ARM}_s${SEED}_kitti_val.json" "$ARM" "$SEED" "$BEST_EP" "$BEST_M" <<'PY'
import json, sys
from src.evaluation.openpcdet_results import parse_log, primary_outcome
log, out, arm, seed, epoch, held = sys.argv[1:7]
classes = parse_log(open(log).read())
json.dump({"arm": arm, "seed": int(seed), "epoch": int(epoch), "held_out_M": float(held),
           "M": primary_outcome(classes), "classes": classes}, open(out, "w"), indent=1)
print("M =", primary_outcome(classes))
PY
echo "=== $NAME: ALL DONE ==="
