## LiDAR study: synthetic scans against the same number of real extras (2026-10-11)

Pre-registered in `docs/lidar/preregistration.md` (section 4, amendments 1 to 6), before any scan was rendered or any
detector trained. Data and its comparison with KITTI: `docs/lidar/stage2_data.md`.

**Question.** Does adding 1,000 synthetic LiDAR scans (depth-derived, KITTI-matched sensor pattern, exact 3D boxes) to a
small real KITTI base improve 3D detection on real KITTI validation scans, **beyond what 1,000 extra real scans give**?

**Arms** (same real base: 371 frames, 10% of KITTI's 3,712-frame standard training split, seed 0; same recipe:
OpenPCDet PointPillars at commit 31f6758af3, 80 epochs, seeds 0 to 4; checkpoint chosen among the last 5 epochs on a
200-frame held-out slice of the training split; scored on the 3,769-frame standard validation split):

| Arm | Training frames |
|---|---|
| A | real base (371) |
| B | A + 1,000 real frames from the rest of the training split |
| E | A + 1,000 synthetic scans (`kitti_v18`) |

Primary outcome M: mean moderate 3D AP (AP_R40) of Car, Pedestrian and Cyclist. Recipe note (amendment 6): ground-truth
sampling runs without KITTI's road-plane files, for every arm.

## Result (KITTI validation, 5 seeds, AP points)

| Arm | M | Car | Pedestrian | Cyclist |
|---|---|---|---|---|
| A real base | 66.3 | 88.9 | 56.2 | 53.9 |
| B + 1,000 real | **72.1** | 89.5 | 61.3 | 65.4 |
| E + 1,000 synthetic | 66.0 | 88.8 | 52.1 | 57.3 |

**The registered comparison, E against B** (paired by seed): d = M_E - M_B = -7.18, -5.04, -6.14, -6.95, -4.92; mean
**-6.05**, 95% paired t-interval **[-7.35, -4.75]**; all five seeds negative. The Car guard holds (E - A = -0.16). The
rule's "helps beyond real scans" is **not met**, and the t-interval half of its "hurts" condition is met (it means
*worse than the same number of real scans*, not worse than none).

**The bootstrap (criterion 3) has not been computed**; it needs per-scan detections, which these jobs do not save. The
verdict is **provisional**, as in entry 38.

## Secondary comparisons (exploratory)

| Comparison | M | Car | Pedestrian | Cyclist |
|---|---|---|---|---|
| E - A (synthetic against nothing) | -0.30 [-2.54, +1.94] | -0.16 [-0.27, -0.04] | **-4.13** [-5.58, -2.67] | +3.39 [-3.01, +9.78] |
| B - A (real extras against nothing) | **+5.75** [+4.50, +6.99] | +0.57 | +5.12 | +11.54 |
| E - B | -6.05 [-7.35, -4.75] | -0.73 | -9.25 | -8.16 |

(mean difference, 95% paired t-interval over 5 seeds.)

- **The synthetic scans did not improve the detector overall.** E - A is -0.30 with an interval that spans zero: no gain
  and no resolvable loss in M.
- **The classes disagree.** Pedestrian AP fell by 4.1 (interval excludes 0); Cyclist rose by 3.4 but the interval
  [-3.0, +9.8] includes 0 at five seeds, so it is not resolved; Car is flat.
- **Real extras help a lot**, mostly on Cyclist (+11.5) and Pedestrian (+5.1).

## Exploratory sweep (3 seeds, M)

| Added frames | Synthetic (E) | Real (B) |
|---|---|---|
| 250 | 66.4 | 69.4 |
| 500 | 65.8 | 70.3 |
| 1,000 | 66.0 (5 seeds) | 72.1 (5 seeds) |

Real frames help more with more of them; synthetic frames stay at the base's level at every size.

## What this does and does not say

- It says **these 1,000 synthetic scans, as generated, gave no net gain** to a PointPillars detector on this benchmark,
  where the same number of real scans gave +5.8 M.
- The data-side differences measured before training (`stage2_data.md`) are candidate explanations and were **not
  tested here**: synthetic scenes hold about 12 times as many pedestrians per frame as KITTI (7.1 against 0.6) and the
  synthetic cars return more points at range (no dropout), with no intensity channel. The pedestrian loss is in line
  with the first of these, but no run isolates it.

## Caveats on the benchmark itself (read before using any absolute number)

1. **The absolute APs are higher than published numbers for this detector.** Arm A, with 10% of the training data,
   scores Car 88.9, above OpenPCDet's published full-data PointPillars (about 77 moderate). I did not find the cause. The
   likely contributor is a property of the standard split: its training and validation frames are drawn from the same
   recorded drives (adjacent frame ids alternate between the two lists), so the validation scenes are not independent of
   the training scenes. The held-out slice, drawn from the training split, scored higher than the validation frames in
   every arm (A 70.9 against 66.3), consistent with that. A sequence-disjoint check was not run.
2. **That property favours the real arms.** Real extras come from the same drives as the validation frames, which
   synthetic scans cannot; part of the B advantage over E may be this, not real-data quality as such. It cannot be
   separated with this split, and it is the first reason not to read E - B as the pure value of real over synthetic.
3. Within-pipeline comparisons between A, B and E use one recipe and are valid as comparisons; absolute levels are not
   comparable with other papers.
4. Image size is a blank 1242 x 375 placeholder for every frame (the study reads no images), and 2 of the 1,000
   synthetic frames have no object in view and carry KITTI's `DontCare` marker.

## Limits

- Bootstrap pending (provisional verdict); 5 seeds; one base size; one synthetic set; one detector; the sweep has 3
  seeds and is exploratory.
- Forward-view only; no intensity, dropout or noise in the synthetic scans.

Files: `results/lidar_{A,B,E}_s{0..4}_kitti_val.json`, `results/lidar_{B,E}{250,500}_s{0..2}_kitti_val.json`,
`results/lidar_study_analysis.json`; analysis tool `bin/analyze_lidar_study.py`.
