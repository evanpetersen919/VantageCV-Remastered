# LiDAR stage 1, check 3.3: label consistency of the depth-derived points (2026-10-10)

Registered in `preregistration.md` section 3.3. Tool: `bin/check_lidar_labels.py`; result file
`results/lidar_label_check.json`.

**Data.** `live_dataset/rider_v18` (1,387 frames with a car or person label, 1920 x 1080, exact labels, depth maps and 3D
boxes). For each frame a scan was sampled from the engine depth (64 channels, 0.1 degree azimuth step, 80 m limit). Each
point takes the instance whose exact visible mask covers the pixel it was read from, and is expressed in the label frame
of the 3D boxes (the level camera frame, `src/export/box3d_formats.py`). The check counts, per object, the share of its
points that lie inside **its own 3D box grown by 0.1 m**. The 3D boxes come from the scenario's geometry and the points
from the engine's rendered depth, so the two are independent.

**Rule (a choice, registered):** for `car`, at least 95% of the instance-labelled points inside the expanded box, pooled
over every car with at least 20 points.

## Result: passed

| Class | Objects with >= 20 points | Labelled points | Pooled share inside the box (+0.1 m) | Per-object share p5 / median |
|---|---|---|---|---|
| **car** (the rule) | 7,563 | 3,943,291 | **99.79%** | 98.3% / 100% |
| person (information only) | 2,589 | 722,915 | 99.99% | 100% / 100% |

- **Cars with fewer than 20 points:** 1,517 of 9,080 (17%), at a median distance of 45 m. These are far cars that the
  0.1 degree and 64-channel pattern covers with few beams; they are excluded by the rule's own minimum, not mislabelled.
- The few per-object shares below 100% (5th percentile 98.3%) are points at the box's edge: wheel arches, mirrors and
  the box's own looseness, which a 0.1 m margin only partly absorbs.

## What it shows and what it does not

- The points of a labelled instance fall where its 3D box says it is, to within 0.1 m for 99.8% of them, so the 3D
  boxes and the engine's depth agree, and the exact masks assign the right points to the right objects.
- It does **not** show that the object boxes are tight (a box larger than the object would pass), only that points are
  not outside the box. Tightness of the 3D boxes is a property of the generator's dimensions, as in the camera studies.
- It is measured on the rider batch's rider-aimed views, not on KITTI-like ego views; the points are from the wide 106
  degree frame and not yet cropped to the KITTI field. The crop removes points and cannot move one.
- Pedestrians are better than cars here (99.99%) because the person boxes are loose around the body; that is a sign of
  the box convention, not of a better surface.

## Stage 1 status

| Check | Result |
|---|---|
| 3.1 range accuracy (median <= 20 mm, p99 <= 100 mm up to 50 m) | passed: 0.04 mm / 1.4 mm (`stage1_range_check.md`) |
| 3.2 self-consistency (re-projection, depth envelope) | passed |
| 3.3 label consistency (car >= 95% inside the box) | **passed: 99.79%** |

Stage 1 is complete. The remaining stage 1 item is the density comparison of section 5 (points per car by range against
KITTI's measured scans), which needs synthetic frames rendered from KITTI's sensor height and field and is part of
preparing stage 2.
