## LiDAR follow-up: fixing the measured domain gaps, and a dilution test (2026-10-12)

Pre-registered in `docs/lidar/followup_preregistration.md` (sections 1 to 5, amendments 1 and 2), written before any
variant was built. Follows entry 39 (1,000 synthetic scans: E - A = -0.30 M, pedestrian -4.1; 1,000 real scans: +5.8).
Audit of the gaps: `results/lidar_domain_gap.json`. Analysis: `bin/analyze_lidar_followup.py`
(`results/lidar_followup_analysis.json`).

**Question.** Do the label, density and intensity conventions that the audit found to differ from KITTI explain why
synthetic scans did not help, and does fixing them make synthetic scans help a detector trained on a small real base?

**Setup.** Same real base A (371 frames), recipe and scoring as entry 39; seeds 0 to 4 for every arm; 97 runs in all.
Every arm adds 1,000 scans to A unless stated.

| Arm | Frames added | What differs |
|---|---|---|
| E0 (entry 39's E) | 1,000 `kitti_v18` | nothing fixed |
| E1 | same 1,000 | range cap (T1), Van relabel (T2) |
| E2 | same 1,000 | + KITTI pedestrian footprint (T3) |
| E3 | 1,000 `kitti_v19` (KITTI-like density) | T1, T2, T3 |
| E4 | `kitti_v19` | T1, T2, T3, sampled reflectance (T5) |
| E3a, E4a | `kitti_v19` | E3 and E4 without T3 (amendment 1) |
| E1lo, E1rnd | 299 `kitti_v18` | low-density frames, and 299 random frames, with E1's conventions |
| Ax3, E1x3 | none / E1's 1,000 | real base listed three times (amendment 2) |
| Ai, Bi, E0i, E3i | as A, B, E0, E3 | intensity zeroed in every frame, real and synthetic |

## The registered test

**E4 - A, M: -3.95 [-5.99, -1.92]**, all five seeds below A; Car guard holds (-0.68). The rule requires a positive
interval, so **the fixes do not make synthetic scans helpful**; the interval lies below 0, so E4 is worse than the
real base alone. Pedestrian AP is the whole story (-12.5 [-14.0, -11.1]); cyclist is +1.3 [-4.6, +7.2], unresolved.
The bootstrap over scans is still not computed (per-scan detections are in each cluster job's `result.pkl`), so the
verdict stays provisional, but the t-interval is not close to 0.

## Arm means (KITTI validation, moderate 3D AP, 5 seeds)

| Arm | M | Car | Pedestrian | Cyclist |
|---|---|---|---|---|
| A | 66.3 | 88.9 | 56.2 | 53.9 |
| Ax3 | 68.4 | 88.8 | 59.7 | 56.8 |
| B (+1,000 real) | 72.1 | 89.5 | 61.3 | 65.4 |
| E0 | 66.0 | 88.8 | 52.1 | 57.3 |
| E1 | 65.2 | 88.8 | 49.3 | 57.4 |
| E2 | 64.3 | 88.8 | 46.2 | 57.9 |
| E3 | 65.4 | 88.6 | 51.7 | 55.9 |
| E4 | 62.4 | 88.2 | 43.7 | 55.2 |
| E3a | 65.7 | 88.7 | 52.3 | 56.0 |
| E4a | 62.6 | 88.2 | 46.4 | 53.1 |
| E1lo | 66.7 | 88.7 | 54.6 | 56.7 |
| E1rnd | 66.2 | 88.7 | 54.4 | 55.6 |
| E1x3 | 69.6 | 88.9 | 58.2 | 61.8 |

## Each hypothesis, by its registered reading

| | Prediction | Contrast, paired by seed, 95% interval | Result |
|---|---|---|---|
| H1 labels (T1, T2) | E rises toward A | E1 - E0: M -0.86 [-1.46, -0.27]; pedestrian -2.7 [-5.7, +0.2] | **not supported** (slightly worse) |
| H2 footprint (T3) | pedestrian rises | E2 - E1 pedestrian -3.1 [-7.7, +1.5]; E3 - E3a pedestrian -0.6 [-3.3, +2.1] | **not supported**, unresolved |
| H3 density (T4) | pedestrian rises | E1lo - E1rnd pedestrian +0.15 [-2.9, +3.2]; E3a - E1 pedestrian +3.0 [-2.3, +8.2] | **unresolved** |
| H4 intensity (T5) | E rises | E4 - E3 M **-3.0 [-3.7, -2.4]** (pedestrian -8.0); E4a - E3a M **-3.1 [-4.0, -2.2]** | **refuted**: resolved harm |
| H6 dilution | pedestrian recovers at 3x real | E1x3 - E1 pedestrian **+8.9 [+3.6, +14.2]**; E1x3 - Ax3 pedestrian -1.4 [-3.7, +0.8] against E1 - A -6.9 | **supported** on both stated conditions |

Beyond the registered readings:

- **With three times the real share, synthetic scans are no longer harmful.** E1x3 - Ax3 is +1.2 [-0.7, +3.2] on M,
  and cyclist +5.0 [+1.2, +8.8] is resolved; pedestrian is -1.4 (unresolved). So most of the pedestrian damage in
  entries 39 and here is **dilution**: with 371 real frames in a training set of 1,371, the pedestrian class is
  learned mostly from synthetic pedestrians.
- **Repetition alone helps.** Ax3 - A is +2.1 [+0.7, +3.5] on M (pedestrian +3.5 [+1.9, +5.0]), with no new
  information. The base A under-trains in 80 epochs relative to the 1,371-frame arms, so entry 39's gain of B over A
  (+5.8) contains a step-count effect. Post hoc (not registered): against the step-matched base, B - Ax3 is
  **+3.7 [+2.8, +4.5]** (cyclist +8.7), and E0 - Ax3 is -2.4 [-4.0, -0.8], E1 - Ax3 -3.3 [-4.9, -1.6].
- **Intensity matters and the synthetic channel does not carry it.** Removing intensity from everything costs real
  data too (Bi - Ai +4.5 against B - A +5.8) but hurts E3 more (E3i - E3 M -4.3, Car -1.8, pedestrian -11.8). With
  intensity off, synthetic scans help cyclist (+4.9 [+1.5, +8.3] for E0i - Ai; +4.5 for E3i - Ai) and wreck pedestrian
  (-13.8 and -13.2).
- **Candidate explanation for T5, not tested:** a reflectance drawn per point from a class and range histogram has
  the right marginals but none of the structure real intensity has on a surface, so the network may learn from noise
  it will not see in KITTI. No run here isolates this.

## What this means

- The audit's gaps were real and measured, but **fixing the labels, footprint, density and intensity as specified
  did not make synthetic LiDAR scans helpful on this recipe**; the confirmatory test is negative.
- The one result that changes the picture is the dilution test. Synthetic scans were hurting mostly through their
  share of the training set, not through their labels; at 3x real share they are neutral on pedestrians and help
  cyclists (+5.0 over the step-matched base).
- Sampled reflectance made things worse, not better. Intensity is a real gap (G5) that this transform does not close.
- Still open from the audit: points per pedestrian (G6, 0.55 times KITTI's), dropout and noise, the shared-drive
  property of the standard split (absolute AP above published values).

## Limits

- The bootstrap over validation scans is not computed for any LiDAR arm; every interval here is a paired t-interval
  over five seeds (4 degrees of freedom).
- One confirmatory test; about 25 descriptive contrasts are reported without correction. The step-matched comparisons
  against Ax3 are post hoc.
- The follow-up shares entry 39's recipe: PointPillars, 10% real base, one base subset (seed 0), 80 epochs, standard
  split with shared recording drives, no road-plane ground-truth sampling.
- E1x3 changes the real share and the number of optimizer steps together; that is why Ax3 is in the table, and the
  two cannot be separated further here.
