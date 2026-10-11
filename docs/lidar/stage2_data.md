# The stage 2 synthetic LiDAR set: what was rendered and how it compares with KITTI (2026-10-10)

The synthetic arm E of the LiDAR study (`preregistration.md`, section 4) draws from this set. This is the section 5
table the preregistration promised, written before any detector is trained on it. Nothing here is a gate; it is
reported as measured.

**Set.** `live_dataset/kitti_v18`: `urban_dense_v18.yaml`, `--profile kitti` (camera 1.71 m, nearly level, no hood),
`--views ego ego ego rider`, 300 scenarios (seeds 90100 to 90399), 1,199 frames, exact labels, depth and semantic maps,
annotation cutoff 80 m for every class, no frame rejected, camera self-check error 0.54 px. Exported by
`bin/export_lidar_kitti.py` to `F:/datasets/kitti_synthetic` (frames numbered from 100000; velodyne `.bin`, `label_2`,
`calib`, blank `image_2`), 542 MB. Sensor: `KITTI_PATTERN` (64 channels, +2.0 to -24.8 degrees, 81.6 degrees across,
0.18 degree steps, 80 m).

## Density against the real scans (measured on 499 KITTI training scans, every 15th frame)

| Quantity | KITTI (measured) | Synthetic (1,199 scans) | Ratio |
|---|---|---|---|
| Points inside the camera image view, per scan | 19,084 | 16,105 | 0.84 |
| Points per fully visible car, 0 to 10 m | 1,773 (n = 26) | 2,697 (n = 959) | 1.5 |
| 10 to 20 m | 777 (n = 184) | 780 (n = 1,294) | 1.0 |
| 20 to 30 m | 210 (n = 165) | 266 (n = 867) | 1.3 |
| 30 to 40 m | 91 (n = 132) | 115 (n = 609) | 1.3 |
| 40 to 80 m | 19 (n = 233) | 42 (n = 1,085) | 2.2 |

Real cars here are untruncated and fully visible (KITTI occlusion level 0); synthetic cars are those with visible
fraction above 0.9. The synthetic cars return more points at range, as expected of a surface with no dropout (dark or
specular paint, rain, glass); the pattern was not tuned further.

## Class mix against the real labels (objects per frame)

| Class | KITTI training (7,481 frames) | Synthetic (1,199 frames) | Ratio |
|---|---|---|---|
| Car | 28,742 (3.84 per frame) | 11,655 (9.72) | 2.5 |
| Pedestrian | 4,487 (0.60) | 8,530 (7.11) | **11.9** |
| Cyclist | 1,627 (0.22) | 461 (0.38) | 1.8 |
| Truck | 1,094 (0.15) | 640 (0.53) | 3.7 |
| Van (no synthetic class) | 2,914 | 0 | |
| Misc | 973 | 180 | |

**The synthetic scenes are far denser than KITTI's**, and pedestrian-heavy in particular (7 per frame against 0.6): the
generator places pedestrians along every sidewalk and crosswalk of a dense urban block, where KITTI's recording route is
mostly residential and suburban roads. The cyclist views add the 0.38 cyclists per frame. KITTI also labels vans, trams
and sitting people, which the generator has no class for; those objects, where present in synthetic scenes, are
labelled as cars or trucks or not at all.

## What is not done

- No detector has been trained on this set.
- Training uses `ImageSets/synthetic.txt` order; the registered arm E draws S = 1,000 of these frames by a seeded
  shuffle, so about 1,000 of the 1,199 are used and the class mix above applies to that subset.
- The density and class-mix differences above are reported before results so they can be read against them.
