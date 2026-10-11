## Cyclist study: synthetic riders against the same number of real extras (2026-10-11)

Pre-registered in `docs/riders/preregistration.md` (sections 1 to 5, amendments 2 and 3), before any rider image was
rendered or any detector trained on one. Data and its audit: `docs/riders/audit_v18.md`.

**Question.** Does adding 1,000 synthetic images of cyclists (Rocketbox riders on the parametric bicycle, exact labels)
improve detection of `rider` and `bike` on real BDD100K images, **beyond what 1,000 extra real rare-class images give**?

**Arms** (same real base of 1,838 BDD100K images, same recipe: YOLOv10m from scratch, 200 epochs, 960 px, AdamW fixed;
seeds 0 to 4):

| Arm | Training images |
|---|---|
| A | real base |
| B | A + 1,000 real images containing a rider, bike or motor |
| C | A with repeat-factor sampling (t = 9 x the smallest class share), 2,111 listed images |
| E | A + 1,000 synthetic images (`rider_v18`, exact labels) |

Primary outcome M: mean AP@[.5:.95] of `rider` and `bike` (amendment 2). Scored on BDD100K val (10,000 images);
Cityscapes val reported as secondary.

## Result (BDD100K, mean over 5 seeds, AP points)

| Arm | M (rider, bike) | rider | bike | motor | person | car |
|---|---|---|---|---|---|---|
| A real base | 8.02 | 7.06 | 8.98 | 4.25 | 21.11 | 43.43 |
| C repeat-factor | 9.76 | 9.01 | 10.52 | 5.83 | 20.78 | 42.96 |
| **E + 1,000 synthetic** | **12.93** | 12.36 | 13.50 | 4.94 | 24.03 | 44.20 |
| B + 1,000 real | 19.20 | 18.42 | 19.98 | 13.61 | 26.75 | 45.10 |

**The registered comparison, E against B** (paired by seed): d = M_E - M_B = -5.63, -7.52, -5.84, -5.96, -6.40; mean
**-6.27**, 95% paired t-interval **[-7.21, -5.33]**. All five seeds are negative. The guards hold (E's car AP is +0.76
and person AP +2.92 against A). By section 5, "synthetic data helps beyond real rare-class images" is **not met**, and
because the interval lies entirely below 0 the t-interval half of the "hurts" rule is met.

**The bootstrap (criterion 3) has not been computed.** It needs each model's per-image detections, which the result
files do not contain. The verdict is therefore **provisional**: the registered "hurts" requires both the t-interval
and the bootstrap interval to lie below 0. The weights are on the cluster and the detections can be dumped and
re-scored; it is expected, not shown, to agree.

**Reading the word "hurts".** In this rule it means *worse than the same number of real images*. It does not mean
worse than not adding the data.

## Secondary comparisons (exploratory, reported as measured)

| Comparison (BDD100K) | Mean difference in M | 95% t-interval | Per seed |
|---|---|---|---|
| E - A (synthetic against nothing) | **+4.91** | [+4.29, +5.53] | +4.48, +5.62, +4.93, +5.11, +4.40 |
| E - C (synthetic against repeat-factor) | **+3.17** | [+2.04, +4.29] | +3.51, +2.46, +4.24, +3.61, +2.02 |
| B - A (real extras against nothing) | +11.18 | [+9.74, +12.61] | all positive |
| C - A | +1.74 | [+0.47, +3.02] | all positive |

- The synthetic images recover **44% of the gain that real extras give** (E - A over B - A) on BDD100K, 50% on
  Cityscapes.
- By class: `rider` +5.30 over A, `bike` +4.52; both lag the real extras by about 6 AP (E - B: rider -6.06, bike -6.48).
- `motor` does not move (4.94 against 4.25): the synthetic set contains no motorcycles.
- `person` rises by 2.9 AP with E, a by-product of the pedestrians in the synthetic frames.
- Cityscapes agrees in sign and size (E - B mean -5.89, interval [-6.65, -5.14]; E - A +5.79; E - C +3.60).

## Exploratory sweep (3 seeds, not used for the decision), M on BDD100K

| Added images | Synthetic (E) | Real (B) |
|---|---|---|
| 250 | 11.04 | 13.19 |
| 500 | 11.49 | 16.56 |
| 1,000 | 12.93 (5 seeds) | 19.20 (5 seeds) |

Synthetic gains grow slowly with more images; real gains grow much faster. At every size the real extras are ahead.

## What this means

- Synthetic cyclists **do help** a detector trained on a small real base: +4.9 AP in M, positive for all seeds, and ahead
  of repeat-factor sampling by +3.2.
- They **do not match real cyclists** of the same count: they supply about half the gain. The preregistered question
  (beyond real extras) is answered: no.
- The audit stated the likely reasons before training (`audit_v18.md`): almost all synthetic bikes are ridden while
  65% of BDD100K bikes are parked, the views are mostly from behind and rarely truncated, and the rendered rider height
  distribution is skewed. These are candidate explanations, not tested ones; no run here isolates any of them.

## Limits

- The bootstrap interval is pending (above); the decision is provisional.
- One real base size (1,838 images, by design a low-data regime), one synthetic set, one recipe.
- The sweep has 3 seeds and is exploratory.
- Cyclists only; motorcycles and scooters were not built, so `motor` is untested.

Files: `results/hpc_rider_{A,B,C,E}_s{0..4}_{bdd100k,cityscapes}_riders.json`, `results/hpc_rider_{B,E}{250,500}_s{0..2}_*`,
`results/rider_study_analysis.json`; analysis tool `bin/analyze_rider_study.py`.
