## LiDAR improvement study, stages 0 and 1: measured sensor effects, label rule, and recipe controls (2026-10-12)

Pre-registered in `docs/lidar/improvement_preregistration.md` (written and committed before any arm was exported or
trained; plan in `docs/lidar/improvement_plan.md`). Follows entries 39 and 40. Analysis:
`bin/analyze_lidar_followup.py --study improvement` (`results/lidar_improvement_analysis.json`). 50 new runs, seeds 0
to 4, same recipe and scoring as entries 39 and 40.

**Question.** (Stage 1) Does a more faithful sensor and label convention, all measured on real KITTI scans (ring
table, per-ring dropout, range noise, minimum-points rule), make 1,000 synthetic scans help? (Stage 0) Are the earlier
results explained by recipe effects: under-training of the real base, the ground-truth-sampling database, a domain
marker, real share against optimizer steps?

## The registered test

**S4 - Astep, M: -3.15 [-3.74, -2.55]**, every seed negative (-3.8, -3.1, -2.7, -2.7, -3.5); Car guard holds (-0.04).
Not supported: with the same number of optimizer steps on real frames only, adding the fully corrected synthetic
scans is **worse**, almost entirely through pedestrians (-8.0 [-10.2, -5.8]). Against the older, under-trained base A
it is -0.83 [-2.95, +1.29].

## Arm means (KITTI validation, moderate 3D AP, 5 seeds)

| Arm | M | Car | Pedestrian | Cyclist |
|---|---|---|---|---|
| A (371 frames, 80 epochs) | 66.3 | 88.9 | 56.2 | 53.9 |
| Astep (371 frames, 296 epochs) | **68.7** | 88.7 | 60.4 | 56.8 |
| B (+1,000 real) | 72.1 | 89.5 | 61.3 | 65.4 |
| E3a (entry 40) | 65.7 | 88.7 | 52.3 | 56.0 |
| S1 table | 65.3 | 88.7 | 52.5 | 54.7 |
| S2 table, dropout, noise | 65.9 | 88.7 | 53.7 | 55.2 |
| S3 table, minimum points | 65.7 | 88.7 | 53.0 | 55.4 |
| S4 all three | 65.5 | 88.7 | 52.4 | 55.4 |
| E3a_rdb (real-only database) | 64.8 | 88.8 | 53.3 | 52.4 |
| S4_rdb | 64.3 | 88.6 | 52.4 | 52.0 |
| E1step (E1, 1,371 x 123 epochs) | 67.2 | 88.8 | 52.2 | 60.7 |
| E1x3 | 69.6 | 88.9 | 58.2 | 61.8 |
| E4 | 62.4 | 88.2 | 43.7 | 55.2 |
| E4_flag | 66.8 | 88.9 | 53.1 | 58.5 |
| E0i | 61.4 | 87.9 | 39.3 | 56.9 |
| E0i_flag | 67.3 | 88.7 | 53.9 | 59.2 |

## Each hypothesis, by its registered reading

| | Contrast (paired by seed, 95% interval) | Result |
|---|---|---|
| Pre-training check (H7) | median points on a pedestrian at 10 to 20 m: 98 (E3a) to 135 (S1) against KITTI's 175.5; 115.5 after dropout (S2) | threshold of 130 met by the table; dropout takes it back below |
| H7 ring table | S1 - E3a pedestrian +0.2 [-2.1, +2.5]; M -0.4 [-2.0, +1.3] | **unresolved**: more points per pedestrian, no AP change |
| H8 dropout and noise | S2 - S1 Car +0.05 [-0.04, +0.14]; pedestrian +1.2 [-1.6, +4.1] | **unresolved** |
| H9 minimum points | S3 - S1 pedestrian +0.5 [-2.7, +3.8] | **unresolved** |
| H10 real-only database | E3a_rdb - E3a pedestrian +1.1 [-1.1, +3.2], cyclist **-3.6 [-6.2, -1.1]**; S4_rdb - S4 M **-1.2 [-2.0, -0.4]** | **not supported**; pasted synthetic cyclists were helping cyclists |
| H11 domain flag | E4_flag - E4 pedestrian **+9.4 [+6.7, +12.1]**, M **+4.5 [+3.1, +5.8]**; E0i_flag - E0i pedestrian **+14.6 [+13.4, +15.8]**, M **+5.9 [+4.7, +7.1]** | **supported**, both arms |
| H12 baseline | Astep - A M **+2.3 [+0.6, +4.1]** (pedestrian +4.2 [+3.0, +5.5]) | **supported**: A under-trains |
| H12 share vs steps | E1x3 - E1step M **+2.4 [+1.1, +3.7]**, pedestrian +6.0 [+2.0, +10.1]; E1step - E1 M +2.1 [+0.6, +3.5] | real share **and** steps each matter; dilution is isolated by E1step |

## What this means

- **Sensor and label realism changes did not resolve at n = 5.** The ring table put more points on pedestrians but the
  detector's AP did not move beyond noise; dropout, noise and the minimum-points rule are the same. This does not show
  they are irrelevant (the minimum detectable difference is about 2 to 3 on M and about 3 on pedestrian), only that
  no step helped by an amount this design resolves.
- **The domain marker is the strong effect.** Giving the network a 0/1 real-or-synthetic input recovers most of the
  pedestrian loss in both arms where intensity did not mark the domain (+9.4 and +14.6 pedestrian). This supports the
  separability pattern of entry 40's analysis: the network does better when it can tell the domains apart. It is a
  training-time property: at test time the flag is 0 (all KITTI frames are real).
- **The best synthetic arms are still not above real-only training with equal steps.** E4_flag - Astep is -1.8
  [-2.7, -1.0] (pedestrian -7.4, cyclist +1.7 [+0.1, +3.4]); E1x3 - Astep is +1.0 [-0.7, +2.7] (cyclist +5.0 [+1.9,
  +8.0], pedestrian -2.2 [-5.2, +0.8]). Real extras remain clearly ahead (post hoc: B - Astep +3.4 [+2.7, +4.2]).
- **Entry 39's headline needs the step-matched baseline.** A gains +2.3 simply from more optimizer steps on the same
  371 frames; every earlier "helps" or "hurts" statement against A should be read against Astep.
- **Cyclists are where synthetic data repeatedly shows a gain** (E1x3 +5.0, E4_flag +1.7 over Astep, E0i_flag +7.2
  over Ai), but each is a different contrast with its own interval and none is the registered test.

## Limits

- Five seeds; the real-only database and flag arms were planned at five although the pedestrian sd (about 3) would
  need eight to detect 6 AP; the flag effects are far larger than that, the database effect is not.
- `E0i_flag - Ai` (+2.7 M, +7.2 cyclist [+3.6, +10.8]) compares against an under-trained, step-unmatched base; the
  step-matched intensity-off control (Ai at 296 epochs) was not registered and not run.
- One real subset; the validation frames share recording drives with the training split; no scan bootstrap yet.
- The beam table is a fit to returns, S2's keep ratios include scene differences, and the 5-point rule is a choice.
- About 25 descriptive contrasts are uncorrected; the post hoc group is labelled. One confirmatory test.
- The flag gives the network the domain at training time; it does not add real information, and the arm says
  nothing yet about object insertion (stage 2), which keeps the real background and intensity.
