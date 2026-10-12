# LiDAR follow-up: why did synthetic scans not help, and does fixing the measured gaps change that?

Written 2026-10-11, **before any variant is built or any detector is trained on one**. It follows experiment 39
(`docs/experiments/39_lidar-synthetic-scans-against-real-extras.md`), where 1,000 synthetic scans gave no net gain
(E - A = -0.30 M; pedestrian -4.1) while 1,000 real scans gave +5.8. Everything not changed here is as in
`preregistration.md` (stage 2 recipe, arms A and B, amendments 1 to 6). Changes after this point are dated amendments.

## 1. What the audit measured (evidence for each fix)

`bin/audit_lidar_domain_gap.py` (400 frames each side; `results/lidar_domain_gap.json`) and the KITTI label statistics
of the training split (`configs/kitti_derived_constants.json`, from `bin/derive_kitti_constants.py`):

| # | Gap | Real KITTI | Synthetic (kitti_v18) |
|---|---|---|---|
| G1 | Labelled range, 99th percentile | Car 74 m, Pedestrian 58 m, Cyclist 56 m | labels go to 80 m: pedestrians 3.2 per frame beyond 40 m against 0.01 |
| G2 | Vehicle size | Car length 3.89 m (p95 4.53); long vehicles are `Van`, ignored by the Car evaluation | mean "Car" length 4.91 m; **71% of synthetic Cars exceed KITTI's Car p95** |
| G3 | Pedestrian density | 0.57 within 40 m per frame; 78% of frames have none | 3.9 per frame (about 7 times) |
| G4 | Pedestrian box convention | width 0.63 m, length 0.82 m | width 0.53 m, length 0.70 m (smaller; also height spread 0.04 against 0.12) |
| G5 | Intensity | mean 0.25 on all points, 0.2 on cars, 0.3 to 0.4 on pedestrians, falling with range | exactly 0 on every point |
| G6 | Points on a pedestrian, 10 to 40 m | 176 / 66 / 26 | 98 / 37 / 20 (about 0.55 times) |

(Cars: points per clean object agree within 1.0 to 1.3 times out to 40 m, and are 2 times at 40 to 80 m.) An earlier
guess in the experiment 39 discussion, that pedestrian boxes were too *loose*, is **refuted by G4**: they are smaller
than KITTI's.

## 2. Hypotheses (predicted before results; direction and why)

- **H1 (labels, G1 and G2).** Capping labelled range at KITTI's and calling over-long vehicles `Van` removes label noise
  and a size bias; this should raise E toward A and B on Car and Pedestrian.
- **H2 (box convention, G4).** Giving synthetic pedestrians KITTI's mean footprint removes a systematic IoU loss
  against KITTI's larger pedestrian boxes; expected to help Pedestrian AP.
- **H3 (density, G3).** KITTI-like pedestrian density stops the detector being trained on a scene statistic that KITTI
  does not have; expected to help Pedestrian AP.
- **H4 (intensity, G5).** An all-zero channel is a feature mismatch for a network that takes intensity as input;
  class and range conditioned reflectance should help, and the intensity-off controls should show a smaller or no effect.

Not addressed here, stated as open: G6 (fewer points on pedestrians), dropout and noise, the shared-drive property of
the standard split, motorcycle-like classes.

## 3. Variants (all add S = 1,000 scans to the same real base A; seeds 0 to 4; recipe unchanged)

Transforms (constants from `configs/kitti_derived_constants.json`, training split only):

- **T1 range cap**: a label beyond the class's cap (Car 74.2 m, Pedestrian 58.4 m, Cyclist 55.8 m) is dropped; the
  object stays in the scan unlabelled, as in KITTI.
- **T2 van relabel**: a `Car` longer than 4.53 m is written as `Van`.
- **T3 pedestrian footprint**: pedestrian box width and length are set to 0.628 m and 0.818 m (location of the
  bottom centre, heading and height unchanged).
- **T4 density**: scans from the new `urban_dense_v19_kitti.yaml` render (pedestrian occupancy fraction drawn in
  [0, 0.1], a sixth of v18's 0.3), `live_dataset/kitti_v19`.
- **T5 intensity**: each point's reflectance drawn from the KITTI-derived histogram of its class (car, pedestrian,
  cyclist, background) and range band (`derive_kitti_constants.py`).

| Arm | Scans | Transforms |
|---|---|---|
| E0 | kitti_v18 (done) | none |
| E1 | kitti_v18, same 1,000 as E0 | T1, T2 |
| E2 | kitti_v18, same 1,000 | T1, T2, T3 |
| E3 | kitti_v19 | T1, T2, T3 (T4 is the data) |
| E4 | kitti_v19 | T1, T2, T3, T5 |
| A-i, B-i, E0-i, E3-i | as A, B, E0, E3 | **intensity zeroed in every frame, real and synthetic** (3 features instead of 4) |

The chain E0 to E4 is fixed in advance, so each step's effect is a paired difference with a defined meaning. The
intensity-off controls separate "the empty channel hid a gain" (E0-i against A-i) from "the fixes work without
intensity" (E3-i against A-i).

## 4. Decision rule and reporting

- **Confirmatory (one test): E4 - A.** The fixes make synthetic scans helpful if the mean of the paired differences
  d_s = M_E4(s) - M_A(s), s = 0..4, is above 0, its 95% paired t-interval (4 degrees of freedom) excludes 0, and the
  Car guard holds (E4's Car AP no more than 1.0 AP below A's). M is the mean moderate 3D AP of Car, Pedestrian, Cyclist.
- **Reported against real extras (descriptive): E4 - B.**
- **Attribution (descriptive):** the steps E1 - E0, E2 - E1, E3 - E2, E4 - E3 and the intensity-off contrasts, each with
  its paired interval and per class. Eight arms, so no step is called an effect without its interval excluding 0, and
  nothing here is selected on results: there is no best-of-N pick.
- **If E4 - A is not resolved**, that is reported as such, and the audit's open items (G6, dropout and noise) remain.
- Every arm, seed and class is published, including those that fail.

## 5. Limits stated in advance

- The validation frames are reused from experiment 39 and share recording drives with the training split. Real arms are
  advantaged by that; no drive-disjoint split is available locally.
- The chain is cumulative: an interaction between fixes cannot be separated from their sum.
- The KITTI-derived constants are annotation conventions of one dataset; matching them is a choice about this
  benchmark, not about LiDAR in general.
- Per-scan detections are not saved by the jobs; OpenPCDet writes `result.pkl` into each job's output folder on the
  cluster, which is the input a bootstrap would need.
- Five seeds, one base size, one detector.

## Amendments

**2026-10-11, amendment 1 (arms added after the first batch of results; written before any of these
arms is trained; the confirmatory test is unchanged).** The first batch (E1, E2 and the intensity-off controls
A-i, B-i, E0-i, five seeds) did not support H1, H2 or H4: E1 - E0 = -0.86 M [-1.46, -0.27]; E2 - E1 pedestrian
-3.1 [-7.7, +1.5]; E0-i - A-i pedestrian -13.8 [-15.3, -12.3] against E0 - A pedestrian -4.1; the synthetic scans
helped Cyclist in the intensity-off control (+4.9 [+1.5, +8.3]). Pedestrian AP fell from 56.2 (A) through 52.1 (E0),
49.3 (E1) to 46.2 (E2). Two further sets of arms are added, all descriptive, none a new confirmatory test:

1. **A direct test of H3 on frames already rendered.** `E1lo`: the 299 `kitti_v18` frames that have 0 to 2 pedestrians
   within 40 m (KITTI's level: 94% of its frames have two or fewer), with the E1 conventions (T1, T2). `E1rnd`:
   299 frames drawn at random from the same frames with the same conventions, the size-matched control. Both add
   299 frames to A. E1lo - E1rnd isolates pedestrian density at equal frame count and equal conventions.
2. **The chain without T3.** `E3a` = `kitti_v19` with T1 and T2 only; `E4a` = `E3a` with T5. E2's pedestrian result
   is in the wrong direction (unresolved), so the chain's later steps are also run without the footprint change,
   and E3 - E3a is the footprint effect on KITTI-density scenes. The original E3, E4 and E3-i stay as registered.

Everything else, including the single confirmatory test (E4 - A), is unchanged. These additions are exploratory:
there are now more arms than the original design, and no arm is picked by its result.
