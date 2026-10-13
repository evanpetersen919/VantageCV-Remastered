# LiDAR improvement study, stage 2: synthetic objects pasted into real scans, registered before any arm is trained

Written 2026-10-12, after experiment 41 (`docs/experiments/41_*.md`) and before any stage 2 arm is built or trained.
Everything not changed here is as in `docs/lidar/improvement_preregistration.md`: PointPillars (OpenPCDet 31f6758af3),
371-frame real base A (subset seed 0), checkpoint chosen among the last 5 epochs on the 200-frame hold slice, scored on
KITTI's 3,769 validation frames, seeds 0 to 4, M = mean moderate 3D AP_R40 of Car, Pedestrian and Cyclist. Changes
after this point are dated amendments.

## 1. Why this stage

Experiment 41 found that sensor and label realism changes of synthetic *scenes* did not resolve, that a domain flag
(the network telling real from synthetic) was the largest effect, and that the best synthetic-scene arms stayed at or
below real-only training with equal optimizer steps. The literature (`docs/lidar/improvement_plan.md`) reports its
gains from inserting rendered objects into **real** scans instead: the scene, its background and its intensity stay
real, so there is no scene-level domain to separate. This stage tests that, with the machinery OpenPCDet already has
(ground-truth sampling) and one addition from that literature (occlusion handling).

No synthetic frame is in any stage 2 training list: the training frames are the 371 real frames of A, for 296 epochs
(the same optimizer steps as `Astep`). Synthetic objects enter only through the ground-truth-sampling database.

## 2. Arms

All arms train on list `A`, 296 epochs, seeds 0 to 4. The synthetic objects are the labelled Car, Pedestrian and
Cyclist objects (at least 5 returns, as OpenPCDet's database requires) of the 1,000 `S4` frames (`kitti_v19` with the
measured ring table, dropout, noise and minimum-points rule, intensity 0), ids from 100000.

| Arm | Paste database | Pasting | Reflectance of pasted synthetic objects |
|---|---|---|---|
| `Astep` (done) | real objects of A | OpenPCDet default | n/a |
| `Astep_occ` | real objects of A | + shadow removal | n/a |
| `O1` | real + synthetic objects | OpenPCDet default | 0 (as exported) |
| `O2` | real + synthetic objects | + shadow removal | 0 |
| `O3` | real + synthetic objects | + shadow removal | drawn per point from KITTI's class and range histograms |

**Shadow removal** (`src/sensors/shadow.py`): for each pasted box, scene points whose direction (azimuth, elevation
from the sensor) falls inside the convex hull of the box's projected corners, and whose range is not less than the
box's nearest corner minus 0.3 m, are removed, in addition to OpenPCDet's removal of points inside the box. It models
the shadow a real object would cast on the beams behind it. It does **not** remove pasted points that a nearer real
surface would hide.

**Reflectance for O3**: the draw of `src/sensors/lidar_intensity.py` (the histograms of
`configs/kitti_derived_constants.json`) with the class of the object and the range of each point; it is applied to the
pasted synthetic objects only, never to a whole scene (the whole-scene use was refuted as implemented in entry 40).

## 3. Hypotheses (direction and reason, stated before results)

- **H13 (synthetic objects as paste candidates).** `O1 - Astep` M above 0, mainly Pedestrian and Cyclist: the
  database offers many more, and more varied, objects than the real base's 371 frames (earlier counts: 229 real
  against 730 to 4,490 synthetic pedestrians with at least 5 returns).
- **H14 (shadow removal).** `Astep_occ - Astep` at or above 0 and `O2 - O1` above 0: pasted objects that leave the
  background behind them untouched are an inconsistency the network can use to tell pasted from real.
- **H15 (reflectance).** `O3 - O2` above 0: with intensity 0 the pasted objects are separable by that channel alone
  (entry 41: a separable domain is what the network exploits); a real-looking value removes that cue.

## 4. Decision rule and reporting

- **Confirmatory (one test): `O3 - Astep_occ` on M.** The method helps if the mean of d_s = M_O3(s) - M_Astep_occ(s),
  s = 0..4, is above 0, its 95% paired t-interval (4 degrees of freedom) excludes 0, and O3's mean Car AP is no more
  than 1.0 AP below `Astep_occ`'s. `Astep_occ` is the control that shares the pasting machinery and differs only in
  the objects.
- **Reported, not confirmatory:** `O3 - Astep`, `O3 - B` (1,000 real extras), `O1 - Astep`, `O2 - O1`, `O3 - O2`,
  `Astep_occ - Astep`, each with its paired interval on M and per class.
- No arm is chosen by its result; every arm and seed is published, including those that fail.

## 5. Limits stated in advance

- Five seeds detect about 2 to 3 M and about 3 on pedestrian (sd about 1.7); a smaller gain is not resolved by design.
- One real subset; validation frames share recording drives with the training split; no scan bootstrap yet.
- The synthetic objects are the geometry of rendered Rocketbox pedestrians, parametric bicycles with riders, and City
  Sample vehicles, sampled once by the beam pattern of the frame they were rendered in; they are pasted at their
  original positions and orientations (OpenPCDet's database sampler), not re-sampled at the pasted location.
- Shadow removal ignores pasted points hidden by nearer real surfaces; the box silhouette is an upper bound of the
  object's, so a little real background around the edges may be removed.
- The intensity histograms are marginal (class and range), not material or incidence conditioned.
- The database holds synthetic objects with intensity 0 in `O1` and `O2`: they remain separable by that channel by
  construction; `O3` is the arm that removes it.

## Result (2026-10-13)

Full write-up: `docs/experiments/42_lidar-synthetic-objects-pasted-into-real-scans.md`; numbers in
`results/lidar_insertion_analysis.json`.

- **Confirmatory `O3 - Astep_occ`: M -3.71 [-5.32, -2.10]**, Car guard holds (-0.19). Not supported.
- H13 unresolved (`O1 - Astep` +0.3 [-0.3, +0.9]); H14 not supported (`Astep_occ - Astep` -1.9, pedestrian -7.0);
  H15 not supported (`O3 - O2` -2.5, pedestrian -7.8).
- No amendments were made to the registered design. A diagnostic outside the registered arms suggests the shadow
  step removes points of real labelled objects (6.6% of real pedestrian points on random pasted boxes); a variant that
  protects them is not registered or run.
