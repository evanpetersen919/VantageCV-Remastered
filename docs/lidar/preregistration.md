# LiDAR from engine depth: preregistration (version 2.0)

Written 2026-10-10, **before any LiDAR code is written and before any point cloud is produced**. Status: stage 1 not
started, stage 2 blocked on data (section 6). Every rule below is fixed now; anything changed later is added as a dated
amendment at the end, never edited in place.

## 1. Question

Two questions, in order. The second is only asked if the first is answered.

1. **Stage 1 (generator validity).** Does a point cloud derived from the engine's depth captures reproduce the scene's
   geometry, with exact per-point class and instance labels, to a tolerance fixed here?
2. **Stage 2 (sim-to-real).** Does adding synthetic point clouds to a small real LiDAR training set improve 3D detection
   on real LiDAR validation scans, **beyond what the same number of extra real scans would give**?

As in the rider study, the second half of the second question is the point: a gain over a real-only baseline can come
from showing the detector any extra scans.

## 2. Why depth-derived, and what that means

- The repository already has a ray-casting LiDAR model (`src/sensors/lidar_model.py`), but it casts against the
  procedural meshes only (roads, buildings, planted objects). The vehicles, pedestrians, riders and most props are
  engine assets whose triangles are not in Python, so that model would produce a city with no cars in it. It is **not
  used** for the dataset.
- The engine renders depth for every frame (`depth/*.png`: 16-bit, metres x 256, along the camera's viewing axis,
  `DEPTH_SCALE` 256). Measured on one frame of the v18 probe (1920 x 1080): 2.04 m to 255.996 m, 19.0% of pixels zero
  (sky and anything beyond the range). That depth includes everything the camera sees.
- A scan is therefore **sampled from the depth image**: for each beam direction, project into the image, read the depth,
  convert the along-axis depth z to range, `r = z * sqrt(1 + tx^2 + ty^2)` with `(tx, ty)` the tangents of the ray in the
  camera frame, and emit the point if a return exists. Class and instance come from reading the semantic and
  exact-instance maps at the same pixel.
- **Consequence stated in advance.** Only what the camera sees can produce a return: a single forward capture gives a
  sector of about 106 degrees horizontal by 74 degrees vertical (`docs/experiments/37`), not 360 degrees. KITTI's 3D
  benchmark labels and scores only objects inside the front camera's field of view (as its development kit describes it;
  to be confirmed at download, not verified locally), so stage 2 needs only that sector. A
  full 360 degree sweep (four captures, 90 degrees apart, from one position) is a later extension, not part of this
  registration.

## 3. Stage 1: generator validity

Three checks, all computable locally and none needing real LiDAR data.

**3.1 Range accuracy against known geometry.** Place surfaces of known position in the scene (the camera self-check's
four known squares and a set of axis-aligned boxes at 5, 10, 20, 30 and 50 m), render depth, derive the points and
compare each point's range with the analytic range to that surface.
- Reported: median and 99th percentile of the absolute error, per range band.
- **Pass** (a **choice**, an engineering tolerance and not a fitted value): median absolute error at most 0.02 m and
  99th percentile at most 0.10 m for ranges up to 50 m. The depth encoding's own step is 1/256 m = 3.9 mm, so the
  median bound is about five steps.

**3.2 Self-consistency.** Re-project every emitted point through the camera model; the pixel must equal the pixel it was
read from to within 0.5 px, and the point's depth must equal the depth map's value there. This is an identity check on
the conversion, and any failure is a bug.

**3.3 Label consistency.** For every object that has an exact 3D box (`box3d`), count the points whose instance id is
that object and the points that fall inside the object's box.
- Reported: per class, the share of an object's instance-labelled points that lie inside its box (expected near 1 for
  rigid vehicles; lower for pedestrians, whose box is looser than the body), and the number of visible objects with no
  point.
- **Pass** (a **choice**): for `car`, at least 95% of instance-labelled points inside the 3D box expanded by 0.1 m, over
  all cars with at least 20 points.

The preregistered sensor model for stage 1 is a model, not a measurement of any device:
- 64 channels, vertical angles uniform over the camera's vertical field (+-36.87 degrees), 0.1 degree azimuth step over
  the camera's horizontal field. The count is chosen to resemble a rotating 64-beam unit; the **KITTI sensor's published
  figures** (64 beams, +2 to -24.8 degrees vertical, about 0.09 degree azimuth step; Geiger et al. 2013) are the target
  for stage 2 and are **not verified locally**.
- No intensity (the engine's depth carries none). A reflectivity channel, if added, is derived from the semantic class
  and is a **MODEL** value labelled as such, not a measurement.
- No dropout, no weather attenuation, no motion distortion (the scene is a frozen pose).

## 4. Stage 2: the sim-to-real study (registered now, run when the data exist)

**Test set:** KITTI 3D object detection, the official validation split, scored once per arm and seed. **Classes:** car,
pedestrian, cyclist (KITTI's names). **Metric:** KITTI AP at 40 recall points (AP_R40), 3D box, moderate difficulty,
IoU 0.7 for car and 0.5 for pedestrian and cyclist. **Primary outcome M:** the mean of the three classes' moderate 3D AP.

**Arms** (one real base set, same recipe, differ only in what is added):

| Arm | Training scans |
|---|---|
| A | Real base only |
| B | A + S real scans (extra KITTI training scans not in the base) |
| E | A + S synthetic scans from this generator |

- **Real base:** a fixed random 10% of KITTI's training split (**choice**; small on purpose, to be the regime where a
  supplement could matter, as in the camera studies). The size is fixed at download and recorded in `split.json`.
- **S = 1,000 synthetic scans** (**choice**), with S = 250 and 500 as an exploratory sweep not used for the decision.
  If the real extra pool is smaller than S, B uses the whole pool and E is matched to that count, and this is reported.
- **Detector and recipe:** PointPillars as implemented in OpenPCDet at its published KITTI configuration, **recipe**
  fixed in the run script before the first run. Seeds 0 to 4 (five per arm).
- **Synthetic scene settings:** the scans are taken from a camera at KITTI's LiDAR mounting height and with the
  image's field of view cropped to KITTI's, so the point density per object and the sensor height match (**choice**; the
  KITTI figures are read from the dataset's calibration files at download, not from memory).

**Decision rule** (the same as the rider study, applied to M):
- d_s = M_E(s) - M_B(s), s = 0..4. **Helps beyond real scans** only if all hold: the mean of d_s is above 0; the 95%
  paired t-interval (4 degrees of freedom) excludes 0; a paired bootstrap over validation scans (10,000 resamples) gives a
  95% interval that excludes 0; and E's car AP is no more than 1.0 AP below A's (guard, **choice**).
- **Hurts** if both intervals lie entirely below 0. **Otherwise inconclusive**, reported as such.

Secondary, not decisive: E against A; per-class AP; AP by range band; the effect of S.

## 5. What the synthetic scans must look like (reported, not gated)

A side-by-side table against the real training scans: points per car by range band, the per-class share of points, the
height of the ground plane, the empty-sky share. Targets are **measured from the real scans** once downloaded; a
tolerance chosen before seeing them would be a guess, so none is set.

## 6. Prerequisites and what is blocked

| Item | State |
|---|---|
| Depth capture, exact instance and semantic maps, 3D boxes | exist, tested |
| Point sampling from depth, `.bin` export, KITTI calibration for the sector | to build (stage 1) |
| KITTI 3D object data (velodyne, calib, labels) | **not present locally**; needs the person's download under the dataset's licence |
| OpenPCDet on the cluster | not installed; needs a build check on the cluster's CUDA |
| Cluster compute | the camera studies' scripts are the template |

Stage 1 can be done without any external data. Stage 2 cannot start until the KITTI download exists. If KITTI is not
obtainable, stage 2 is **not** replaced by an unregistered alternative: a different real LiDAR dataset (nuScenes, Waymo)
would be a new amendment with its own detector recipe and rules, written before use.

## 7. Limits stated in advance

- A depth image is a rasterised view: thin structures (wires, railings, foliage edges) alias to the pixel grid and
  transparent surfaces (glass) return the surface behind or in front depending on the depth buffer, where a real LiDAR
  shows partial returns. This is not corrected.
- No intensity, dropout, multi-return or weather effects, so the synthetic scans are cleaner than real ones. A model
  trained on them may learn that cleanliness; reported, not mitigated.
- One forward sector, one sensor height, a frozen scene. Nothing here says anything about 360 degree detectors or about
  moving objects.
- Five seeds and KITTI's small validation split give wide intervals for the rarer classes; a true small gain may come out
  inconclusive.
- The synthetic world is a single engine's city, a different place from KITTI's German streets (driving side, signage,
  vehicle models). Only geometry and point statistics are the claim.

## Amendments

**2026-10-10, amendment 1 (no rule changed).** KITTI's benchmark page states that work without a peer-reviewed paper
"must be evaluated on a split of the training set" and may not be submitted to the test server. Stage 2 already does
this: it trains on a subset of KITTI's labelled training split and is scored on the held-out validation part of that
split, using KITTI's own metric code locally. **No submission is made to the KITTI test server**, and the results are not
presented as leaderboard entries. The test split's labels are not available and are not used.

**2026-10-10, amendment 2 (sampling detail fixed before any engine depth is sampled; no rule changed).** A first
implementation (`src/sensors/depth_lidar.py`, tested on synthetic planes only, no engine depth yet) showed that reading
the nearest depth pixel puts a slanted surface off by a pixel's footprint along the beam: on a level ground plane
between 20 and 50 m the nearest-pixel range is off by a median 0.14 m and up to 0.30 m, which would fail check 3.1 on
the ground for a reason that has nothing to do with the engine. The registered sampler therefore **interpolates
inverse depth between the four neighbouring pixel centres** (exact on a plane) when all four have a surface and their
depths agree within 5% of the nearest (a **choice**), and otherwise uses the nearest pixel, so nothing is blended across
an object edge. On the synthetic ground plane the interpolated range error is below 1 cm. Check 3.1 is measured on the
registered sampler; the nearest-pixel error is reported beside it. The default sensor pattern is 64 channels over
the camera's vertical field and 0.1 degree azimuth steps with a 120 m range limit (a **choice**).

**2026-10-10, amendment 3 (KITTI sensor facts measured from the downloaded training data; the stage 2 synthetic setup
is fixed from them before any synthetic scan is made; no decision rule changed).** All numbers below are measured on
`F:/datasets/kitti/training` (7,481 frames, extracted by `bin/extract_kitti.py`; 150 random scans for the scan
statistics, every tenth label file for the mounting height), not quoted from a paper.

| Measured | Value |
|---|---|
| Points per scan | 119,450 mean; 27,309 mean inside the front camera's horizontal field |
| Elevation of returns | minimum -24.86 degrees, 1st percentile -23.21, 99th percentile +2.82 (the maximum, +9.07, is a few outliers) |
| Range inside the camera field | median 9.8 m, 95th percentile 42.3 m, maximum 80.0 m |
| Front camera (calibration P2) | 721.5 px focal length, 1242 x 375 image: **81.6 degrees horizontal, 29.2 degrees vertical** |
| Camera height above ground | median 1.71 m (cars' bottom centre in the camera frame; the 5th to 95th percentile spread, 1.03 to 2.34 m, is terrain slope, not mounting) |

Consequences, fixed now:
1. **The synthetic scans are cropped to KITTI's camera field**: 81.6 degrees horizontal. Elevations run from -24.8 to
   +2.0 degrees (the lower edge matches the measured -24.86; the upper edge is the measured 99th percentile rounded
   down, so the few high returns KITTI has above the horizon are not copied). A 106 x 74 degree frame is far wider than
   this, so every synthetic frame is a crop and a scan from a wide frame has its sky and upper facades removed, which
   matches what KITTI keeps.
2. **Sensor height 1.71 m** for the synthetic ego camera (the measured median), not the generator's 1.25 to 2.0 m range.
3. **Range limit 80 m** (the measured maximum of KITTI returns in the camera field).
4. The azimuth step was **not** measured cleanly (the one-band estimate mixed channels and gave a median 0.18 degree,
   twice the published ~0.09); it stays a **choice** at 0.1 degree, and the points-per-scan comparison in section 5 is
   the check that the density is right: the synthetic scan should have about 27,000 points in the sector.

**2026-10-10, amendment 4 (wording of the split; no rule changed).** Section 4 says "KITTI 3D object detection, the
official validation split". KITTI itself publishes only a labelled training set (7,481 frames) and an unlabelled test
set; the train/validation division used by 3D detection work is the **standard 3,712 / 3,769 frame split** (Chen et al.
2015, as shipped by OpenPCDet). The two lists are stored in `configs/kitti_splits/` (taken from the OpenPCDet repository,
Apache-2.0; checked here: disjoint, and together exactly frames 000000 to 007480). Everywhere the preregistration says
"validation split" it means this one. The 10% real base of section 4 is drawn from the 3,712 training frames (about 371),
seeded, and the validation frames are never used for training or for choosing a checkpoint other than by the rule
fixed for the camera studies (a held-out slice of the training frames).

**2026-10-10, amendment 5 (azimuth step corrected from a measured density check; written before any stage 2 scan is
rendered; no decision rule changed).** Amendment 3 left the azimuth step a choice at 0.1 degree and said the density
check would decide. A 16-frame probe in the kitti profile (`--profile kitti`, 1.71 m, nearly level; probe frames are not
study data) gave 51,332 points per scan against KITTI's measured level, so the choice was wrong. Amendment 3's remark
that 0.18 degrees is "twice the published 0.09" was also wrong: KITTI was recorded at 10 Hz, where the unit's azimuth
step is about 0.17 to 0.18 degrees; 119,450 points per revolution over 64 channels is about 1,866 returns per channel,
a step of 0.19 degrees, and the measured step was 0.179. The step is therefore **0.18 degrees** (measured).

Targets restated with like-for-like definitions, measured on 499 KITTI training scans (every 15th frame): **points inside
the image view (u in [0, 1242), v in [0, 375)) mean 19,084**; for **untruncated, fully visible cars** the median points
per car by range are 1,773 (0 to 10 m, 26 cars), 777 (10 to 20 m, 184), 210 (20 to 30 m, 165), 91 (30 to 40 m, 132) and 19
(40 to 80 m, 233). The earlier "27,309" counted the whole 81.6 degree sector without the vertical limit of the image.
Probe at 0.18 degrees: 16,457 points in the image view (86% of KITTI's) and per-car medians 3,012, 812, 341, 116 and 47
(9, 18, 15, 11 and 15 cars; fully visible, same distance bands), that is +70%, +5%, +62%, +27% and about 2.5 times
KITTI's. The synthetic cars return more points than real ones at range, as expected of a clean surface with no dropout
(dark and specular paint, rain); the pattern is not tuned further, as that would be fitting the supplement to the test
distribution, and the full set's table (section 5) is reported as measured.
