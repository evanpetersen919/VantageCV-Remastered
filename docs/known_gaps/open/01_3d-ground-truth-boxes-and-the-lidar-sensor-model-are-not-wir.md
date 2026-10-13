### [RESOLVED in 2.0] 3D ground-truth boxes and the LiDAR sensor model are not wired into the live pipeline or export
Resolved. The COCO export carries a KITTI-convention `box3d` per object and a `kitti_P2` per image (1.2), and 2.0
adds a LiDAR sensor: 64 beams sampled from the engine's metric depth, matched to the KITTI Velodyne's field of view,
exported in KITTI layout with exact 3D boxes (`src/sensors/depth_lidar.py`, `bin/export_lidar_kitti.py`).

What is still not done, and is recorded rather than hidden: the scans are a forward sector (not 360 degrees);
intensity is 0 unless sampled values are requested, and sampled intensity was found to hurt a detector; there is no
multi-return or beam divergence; the older ray-casting model in `src/sensors/lidar_model.py` is still not in the
dataset path. Whether synthetic scans improve a detector is measured in experiments 39 to 42 of
`EXPERIMENT_LOG.md`: not in the setup tested.
