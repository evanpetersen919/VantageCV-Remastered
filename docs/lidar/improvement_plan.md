# LiDAR improvement plan (2026-10-12)

Written after experiments 39 and 40, from four independent reviews of the pipeline: the sensor model, the scene and
label content, the training recipe and statistics, and the published sim-to-real LiDAR literature. Every number below
is a measurement or a cited result; anything not verified is marked. **Nothing here has been trained yet**: effects
are hypotheses, to be registered before they are run (`followup_preregistration.md` style).

## 1. Where things stand

- Synthetic scans did not help (E4 - A -3.95 M [-5.99, -1.92]); the 1,000-frame arm is at best neutral
  (E0 - A -0.30 M, n.s.), pedestrians are hurt most, cyclists nominally helped (post hoc, fragile).
- The literature agrees: no replicated paper shows pure synthetic scans added to a small real base improving
  detection. Gains come from very large synthetic pretraining, real-scan-derived simulators, or **real backgrounds with
  inserted objects**. A CARLA study (Marcus et al., 2025) also found heuristic simulated intensity hurts transfer.
- Statistical limits of what we have: n = 5 seeds detects only about 2 to 3 AP on M and about 5 on cyclist; everything
  is conditional on one 371-frame real subset; seed pairing gives no variance reduction (mean seed correlation 0.03);
  the baseline A is under-trained (Ax3 - A +2.1 M), so entry 39's "+5.8 for real extras" includes a step effect.

## 2. Findings, by source

### Sensor model (measured on KITTI scans against synthetic scans)

| Gap | Real | Synthetic | Fix |
|---|---|---|---|
| Vertical ring layout | two blocks: about 0.5 deg below -8.4 deg, about 0.3 deg above | uniform 0.425 deg | explicit elevation table; re-sampling 120 frames with two blocks gave 1.26x points on pedestrians |
| Dropout | near-ground rings 23 to 69% filled | 96 to 100% | per-ring, per-range keep mask measured from KITTI |
| Range noise | ground sigma about 3 cm | about 1.4 to 2.5 cm | add about 2 cm Gaussian (low impact) |
| Intensity | real structure | sampled per point, refuted | physical model (albedo x cos incidence / r^2) or none |
| Sweep skew, glass, beam divergence | present | absent | lower priority |

### Scene and labels (not in the first audit)

| Gap | KITTI | v18 / v19 |
|---|---|---|
| Pedestrians within 40 m with under 5 points in the box | 0.7% | 15.5% / 13.4% |
| Cars with 0 points | 0.3% | 4.1% / 6.1% |
| Pedestrians beyond 30 m, share of labelled | 18% | 64% (v18) |
| Pedestrians with a neighbour under 1.5 m | 1% | 52 to 57% |
| Distinct (h, w, l) among Cars | 688 | 10 (3 after the Van rule) |
| Labelled-object points, share of FOV points | 7.4% | 22% / 20% |
| Cyclists with abs(x) under 2 m | 10 to 13% | 62 to 66% |
| Objects with at least 5 points in 371 real vs 1,000 synthetic frames, pedestrians | 229 | 4,490 (v18) / 730 (v19) |

The last row matters most: **the ground-truth-sampling database is built from every frame in the arm's list**, so
pasted pedestrians are mostly synthetic whatever the frame share. This is a candidate mechanism for the dilution
result and costs one line to ablate (the agent did not verify OpenPCDet's sample-group defaults; check the cluster
config).

### Recipe and statistics

- Domain separability looks like the strongest pattern in the existing data: arms where the synthetic intensity is
  exactly 0 (a perfect domain flag) lose 4 to 7 pedestrian AP against their real-only arm; arms where intensity does
  not mark the domain lose 10 to 14. The ranges do not overlap. This is a pattern, not a test.
- Batch-norm statistics are about 73% synthetic in the 1,000-frame arms.
- Fixed epochs mean A sees about 10x fewer steps than a full-split run.

## 3. Plan, in order

Costs use the measured Ax3 (about 1 h) and E1x3 (about 2.5 h) job times; per-run logs are not in the repo.

### Stage 0: make the comparison trustworthy (no new data)

| # | Experiment | Why | Arms | Seeds | GPU-h |
|---|---|---|---|---|---|
| 0.1 | Matched-step controls | A is under-trained; E1x3 - E1 confounds share with steps | A x 240 epochs, A x 800 epochs, E1 x 120 epochs | 5 | about 32 |
| 0.2 | Real-only GT-sampling database | cheapest mechanism test for dilution | E1 and E1x3 with the database built from A only | 8 | about 24 |
| 0.3 | Domain flag | tests the separability pattern directly | E0i and E4 with a 5th input feature (0 real, 1 synthetic) | 5 | about 15 |
| 0.4 | More real subsets, the bootstrap, a drive-disjoint split | everything is conditional on one subset | A, Ax3, B, E1x3 on seeds 1 and 2 of the base | 3 | about 32 |

Decision rule (to register): replace A by the matched-step baseline for every later "helps" test; the dilution reading
stands only if E1x3 - E1 x 120 pedestrian is above 0 with its interval.

### Stage 1: fix the sensor and the labels (re-export, no re-render)

One re-export from the stored depth maps: (a) KITTI-like two-block elevation table, (b) per-ring dropout mask and
about 2 cm noise, (c) drop labels with fewer than 5 points in the box and match KITTI's per-band pedestrian counts,
(d) pitch and roll jitter of scan and labels. Expectation from measurements: pedestrian points per object about 0.55x
to about 0.7x KITTI's, label noise on pedestrians from 15% empty boxes to under 1%. Arms: E1 conventions plus each
step in turn, then all together, against the Stage 0 baseline. About 25 GPU-h per 5-seed arm pair.

### Stage 2: the structural change the literature supports (object insertion)

Render isolated Rocketbox pedestrians, cyclists and cars, sample them with the corrected beam pattern, and merge them
into **real** KITTI scans: for each beam, keep the nearer of the real and inserted return, delete real points inside
or behind the inserted object, and give the object its class's KITTI intensity (median, or an honest ablation).
Reference results (LiDAR-Aug, Real-Aug, AP_R11, different recipes, so only deltas transfer): pedestrian +2 to +11 AP
over no insertion or GT-sampling alone; ray-based collision handling worth about +5.5 AP. It keeps the real
background and intensity, and never dilutes real frames. Cost: medium (a new insertion tool plus object renders).

### Stage 3 (only if Stages 0 to 2 leave a gap)

- Re-render with KITTI-like vehicle size diversity (per-instance scale), pedestrian grouping and heading, more
  background clutter, cross-traffic cyclists (rows 4, 5, 6, 9 of the scene table).
- Synthetic pretraining then real fine-tuning, with matched-step controls (published gains +0 to +4).
- A learned depth-only raydrop model (published +1 to +3 AP) once the hand-measured dropout is in.

## 4. What would count as success

Pre-register one confirmatory test before Stage 1 or 2 runs: the best arm against the Stage 0 matched-step baseline on
M, five seeds or more, Car guard 1.0 AP, with the paired interval and the scan bootstrap. Report the arms that fail.

## 5. Not verified

- OpenPCDet's GT-sampling defaults and the training schedule on the cluster (config lives there).
- Whether pasted synthetic objects actually drive the pedestrian loss (experiment 0.2 tests it).
- Why 15% of camera-visible synthetic pedestrians have under 5 LiDAR points (box offset or occlusion mix suspected,
  not diagnosed).
- The exact HDL-64E elevation table (not on disk; a two-block fit from scans was used).
