# LiDAR study on the cluster

Registered in `docs/lidar/preregistration.md` (section 4 and amendments 1 to 5). Arms A (real base), B (+1,000 real
extras), E (+1,000 synthetic) at seeds 0 to 4, PointPillars from OpenPCDet, scored on KITTI's validation frames.

**Status: written but not yet run on a cluster.** The first run is the test; expect to fix the environment (spconv
wheel, CUDA module, Python version) before anything trains.

## Order

1. **Build and upload the data** (on the PC; 10.65 GB, 6,340 frames):
   ```
   PYTHONPATH=. python bin/prepare_lidar_arms.py --kitti F:/datasets/kitti --synthetic F:/datasets/kitti_synthetic --out F:/datasets/lidar_study
   tar -cf F:\hpc_upload\vantagecv_lidar_study.tar -C F:\datasets lidar_study
   scp F:\hpc_upload\vantagecv_lidar_study.tar <user>@<cluster>:/scratch/<user>/VantageCV/
   ```
   On the cluster: `cd .../VantageCV/data && tar -xf ../vantagecv_lidar_study.tar` (gives `data/lidar_study/kitti`).
2. **Install OpenPCDet** once, on the login node. Add the modules to `hpc/config.local.env` (read by the
   LiDAR scripts only), then run the setup. For a cluster that offers `python39` and `cuda11.7/toolkit/11.7.1`:
   ```
   echo 'export LIDAR_MODULES="python39 cuda11.7/toolkit/11.7.1"' >> hpc/config.local.env
   echo 'export LIDAR_TORCH_INDEX="https://download.pytorch.org/whl/cu117"' >> hpc/config.local.env
   cd .../VantageCV/repo && git pull
   TORCH_VERSION=1.13.1 TORCHVISION_VERSION=0.14.1 SPCONV_PACKAGE=spconv-cu117 PYTHON=python3 \
     bash hpc/lidar/setup_openpcdet.sh
   ```
3. **Smoke test** (one epoch of arm A, seed 0), and read its log before anything else:
   ```
   EPOCHS=1 SELECT_LAST=1 bash hpc/lidar/submit_lidar_study.sh smoke A 0
   ```
   A good run ends with `ALL DONE` and writes `results/lidar_A_s0_kitti_val.json` with a Car, Pedestrian and
   Cyclist AP table. A one-epoch score is meaningless; the smoke test only proves the chain.
4. **Submit the study:** `bash hpc/lidar/submit_lidar_study.sh` (15 jobs), with `SWEEP=1` for the 12 exploratory ones.
5. **Bring results back and score** (PC):
   ```
   scp "<user>@<cluster>:/scratch/<user>/VantageCV/results/lidar_*_kitti_val.json" results/
   PYTHONPATH=. python bin/analyze_lidar_study.py
   ```

## What each job does

Copies nothing large (frames are read through symlinks), makes two sets of OpenPCDet info files (one where `val` is
the 200-frame held-out slice, one where it is KITTI's 3,769 validation frames), trains PointPillars for 80 epochs,
evaluates the last 5 checkpoints on the held-out slice, scores the best on the validation frames, and writes the
parsed per-class AP. The seed is patched into a job-local copy of OpenPCDet's `train.py` (it hard-codes 666).

## Known weak points

- `Python 3.12` is outside OpenPCDet's tested range; the setup script warns and `PYTHON` should point at 3.8 to 3.10.
- spconv must match the CUDA toolkit; the script refuses to guess a wheel.
- The log parser reads OpenPCDet's printed `AP_R40` blocks (`src/evaluation/openpcdet_results.py`); if a newer
  OpenPCDet changes that text the parser will raise rather than return a wrong number.
- The paired bootstrap of the decision rule needs per-scan detections, which this setup does not save; the verdict
  from `analyze_lidar_study.py` is provisional.
