# LiDAR improvement study, stages 0 and 1: registered before any new arm is exported or trained

Written 2026-10-12. It implements the first two stages of `docs/lidar/improvement_plan.md`, which follows experiments
39 and 40 (`docs/experiments/39_*.md`, `40_*.md`). Everything not changed here is as in
`docs/lidar/preregistration.md` and `followup_preregistration.md`: PointPillars (OpenPCDet 31f6758af3), 371-frame real
base A (subset seed 0), 80 epochs unless stated, checkpoint chosen among the last 5 epochs on the 200-frame hold slice,
scored on KITTI's 3,769 validation frames, seeds 0 to 4, outcome M = mean moderate 3D AP_R40 of Car, Pedestrian and
Cyclist. Changes after this point are dated amendments.

## 1. What changes, and the evidence for it

Constants come from `bin/derive_beam_pattern.py` (300 real KITTI training scans, 250 synthetic frames of `kitti_v19`;
`configs/kitti_beam_pattern.json`):

| Transform | Measured basis | What it does |
|---|---|---|
| S1 beam table | return elevations of real scans fall on 64 rings in two blocks: lower 32 irregular, 0.39 to 0.63 deg; upper 32 about 0.32 deg; the synthetic sensor was uniform at 0.425 deg | the sampler uses the measured table |
| S2 dropout and noise | per-ring fill of real scans against synthetic at the same rings: 24, 29, 36, 48, 63, 67, 77% on the seven lowest rings and about 0.7 to 1.0 elsewhere; ground scatter 2.7 cm real against 1.4 cm synthetic | each return is kept with the ring's real-over-synthetic fill ratio (capped at 1); ranges get Gaussian noise of sigma 2.3 cm (the extra scatter, `sqrt(2.66^2 - 1.39^2)`) |
| S3 minimum points | pedestrians within 40 m with fewer than 5 returns in the box: 0.7% of KITTI labels against 15.5% (v18) and 13.4% (v19) | an object with fewer than 5 returns in its box is left unlabelled (the object stays in the scan, as in KITTI); 5 is a **choice** |

Limit of S2: the keep ratio compares scenes as well as sensors (open sky above the horizon differs), so it mixes real
dropout with scene differences; it is applied per ring only.

## 2. Arms

All stage 1 arms add the same 1,000 `kitti_v19` frames to A (the frames of `E3a`), with T1 range cap and T2 Van rule,
intensity 0, new ids 910000 to 940000.

| Arm | Transforms on top of `E3a` |
|---|---|
| `E3a` (done) | none |
| `S1` | beam table |
| `S2` | beam table, dropout, noise |
| `S3` | beam table, minimum points |
| `S4` | beam table, dropout, noise, minimum points |

Stage 0 (recipe, no new data):

| Arm | What | Frames x epochs |
|---|---|---|
| `Astep` | A for as many optimizer steps as the 1,371-frame arms | 371 x 296 (about 109.8k visits against 109.7k) |
| `E1step` | E1 for as many steps as `E1x3` | 1,371 x 123 (about 168.6k against 169.0k) |
| `E3a_rdb`, `S4_rdb` | as `E3a`, `S4`, but the ground-truth-sampling database is built from real frames only (ids below 100000) | 80 |
| `E0i_flag`, `E4_flag` | as `E0i` (intensity off, both domains) and `E4` (sampled intensity), with a 5th point feature: 0 for real, 1 for synthetic | 80 |

## 3. Hypotheses (direction and reason, stated before results)

- **H7 (ring table).** `S1 - E3a` pedestrian above 0. Test on scans, before any training: points on a pedestrian at
  10 to 20, 20 to 30 and 30 to 40 m should rise from 98 / 37 / 20 (audit G6) toward KITTI's 176 / 66 / 26; the
  agent's re-sampling suggested 1.26x.
- **H8 (dropout and noise).** `S2 - S1` Car and Pedestrian at or above 0.
- **H9 (minimum points).** `S3 - S1` pedestrian above 0.
- **H10 (real-only database).** `E3a_rdb - E3a` and `S4_rdb - S4` pedestrian above 0 (mechanism: pasted pedestrians
  stop being mostly synthetic).
- **H11 (separability).** `E4_flag - E4` and `E0i_flag - E0i` pedestrian above 0 (a domain flag lets the network
  protect real pedestrians as zero intensity did).
- **H12 (baseline).** `Astep - A` above 0 (A is under-trained); `E1x3 - E1step` pedestrian above 0 isolates real
  share from steps.

## 4. Decision rule and reporting

- **Confirmatory (one test): `S4 - Astep` on M.** The fixes make synthetic scans help beyond real-only training with
  equal optimizer steps if the mean of d_s = M_S4(s) - M_Astep(s), s = 0..4, is above 0, its 95% paired t-interval
  (4 degrees of freedom) excludes 0, and S4's mean Car AP is no more than 1.0 AP below Astep's.
- **Reported, not confirmatory:** `S4 - A` (entry 39's baseline), `S4 - B`, `S4 - Ax3`, and every contrast of section 3
  with its interval, per class. No arm is chosen by its result; every arm and seed is published.
- **Pre-training check (H7):** if the 10 to 20 m pedestrian point count has not risen above 130 of the 176 KITTI
  value, the sensor change is reported as not having worked as intended before any training is judged.
- The scan bootstrap (`per-scan detections`) is still not computed; verdicts are provisional on the t-interval.

## 5. Limits stated in advance

- Five seeds detect about 2 to 3 M and about 5 on cyclist; the real-only database and flag arms (H10, H11) were
  planned at 5 seeds although 8 would be needed to detect 6 pedestrian AP (the measured pedestrian sd is about 3).
- One real subset; the validation frames share recording drives with the training split.
- The beam table is a fit to returns (the exact calibration table is not on disk); S2's keep ratios include scene
  differences; 5 returns is a choice.
- About 25 descriptive contrasts are uncorrected; the single confirmatory test is `S4 - Astep`.
- Intensity stays 0 for synthetic frames in every stage 1 arm (sampled reflectance was refuted as implemented), so the
  synthetic domain remains perfectly separable by intensity; H11 tests whether that matters.

## Result (2026-10-12)

Full write-up: `docs/experiments/41_lidar-improvement-stages-0-and-1.md`; numbers in
`results/lidar_improvement_analysis.json`.

- **Confirmatory `S4 - Astep`: M -3.15 [-3.74, -2.55]**, all seeds negative, Car guard holds. Not supported.
- H7 pre-training check met by the table (135 against the 130 threshold; dropout lowers it to 115.5); H7, H8, H9 are
  unresolved in AP. H10 not supported. H11 supported (E4_flag - E4 pedestrian +9.4; E0i_flag - E0i +14.6). H12
  supported: Astep - A +2.3 and E1x3 - E1step +2.4.
- No amendments were made to the registered design.
