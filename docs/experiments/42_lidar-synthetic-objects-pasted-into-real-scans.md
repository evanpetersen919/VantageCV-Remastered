## LiDAR stage 2: synthetic objects pasted into real scans (2026-10-13)

Pre-registered in `docs/lidar/insertion_preregistration.md` (written and committed before any arm was trained).
Follows entry 41. Analysis: `bin/analyze_lidar_followup.py --study insertion`
(`results/lidar_insertion_analysis.json`). 20 runs, seeds 0 to 4, same recipe and scoring as entries 39 to 41.

**Question.** Entry 41 found that synthetic *scenes* did not beat real-only training with equal optimizer steps. Do
synthetic *objects*, pasted into real scans through the ground-truth-sampling database, help, with the real scene,
background and intensity unchanged? No synthetic frame is in any training list: the 371 real frames of A, 296 epochs
(the steps of `Astep`).

| Arm | Paste database | Shadow removal | Intensity of pasted synthetic objects |
|---|---|---|---|
| `Astep` (entry 41) | real | no | n/a |
| `Astep_occ` | real | yes | n/a |
| `O1` | real + synthetic (1,000 `S4` frames) | no | 0 |
| `O2` | real + synthetic | yes | 0 |
| `O3` | real + synthetic | yes | KITTI class and range histograms |

## The registered test

**O3 - Astep_occ, M: -3.71 [-5.32, -2.10]**; Car guard holds (-0.19). Not supported: the method is **worse** than the
same machinery on real objects only, through pedestrians (-7.3 [-10.8, -3.7]).

## Arm means (KITTI validation, moderate 3D AP, 5 seeds)

| Arm | M | Car | Pedestrian | Cyclist |
|---|---|---|---|---|
| Astep | **68.7** | 88.7 | 60.4 | 56.8 |
| Astep_occ | 66.8 | 88.7 | 53.4 | 58.2 |
| O1 | **68.9** | 88.7 | 60.3 | 57.9 |
| O2 | 65.6 | 88.4 | 53.9 | 54.4 |
| O3 | 63.1 | 88.5 | 46.2 | 54.5 |
| B (+1,000 real frames) | 72.1 | 89.5 | 61.3 | 65.4 |

## Each hypothesis, by its registered reading

| | Contrast (paired by seed, 95% interval) | Result |
|---|---|---|
| H13 synthetic objects as paste candidates | O1 - Astep M **+0.3 [-0.3, +0.9]**; pedestrian -0.1 [-2.1, +1.9]; cyclist +1.1 [-0.5, +2.6] | **unresolved**: neutral |
| H14 shadow removal | Astep_occ - Astep M **-1.9 [-2.5, -1.2]**, pedestrian **-7.0 [-9.5, -4.5]**; O2 - O1 M **-3.4 [-4.5, -2.3]** | **not supported**: harmful, resolved |
| H15 reflectance on pasted objects | O3 - O2 M **-2.5 [-4.9, -0.1]**, pedestrian **-7.8 [-10.8, -4.7]** | **not supported**: harmful |

## What this means

- **The clean arm, `O1`, is neutral.** Adding 1,000 frames' worth of synthetic objects as paste candidates, with no
  synthetic scene, no shadow handling and intensity 0, left every class within noise of `Astep`. This is the most
  direct use of synthetic objects tested so far, and at this design it neither helps nor hurts.
- **Both components added to the method hurt, each by a resolved margin.** Sampled reflectance on pasted objects
  repeats entry 40's pattern (sampled intensity costs pedestrians heavily). Shadow removal costs pedestrians **with no
  synthetic object in the database at all** (`Astep_occ - Astep`), so the harm comes from the shadow step itself.
- **Candidate mechanism for the shadow harm, tested only offline.** The hook removes every scene point in a pasted
  box's shadow, **including points of real labelled objects, whose boxes remain as labels**. On the 150 real frames of
  A with 6 random non-colliding pasted boxes per frame (a diagnostic, not the sampler's placement), shadows removed 4.8%
  of all points but **6.6% of the points of real pedestrians, 6.9% of cars and 8.7% of cyclists**. A pedestrian whose
  points vanish behind a pasted car is a label without evidence. This was not trained on and may not be the whole
  story: the hull of the projected corners also takes real background around the pasted object's edge.
- **So this is a negative result for the implementation tested, not for object insertion.** The literature's
  ray-based handling removes pasted points hidden by real surfaces and avoids shadowing labelled objects; this hook did
  neither. A variant that keeps real labelled objects' points (or drops a paste that would shadow one) is the
  obvious next arm and is not registered or run here.

## Limits

- Five seeds; minimum detectable difference about 2 to 3 on M and about 3 on pedestrian.
- Same real subset, same drive-sharing validation frames, no scan bootstrap.
- The pasted objects keep their original positions and orientations (OpenPCDet's sampler) and carry the sampling
  pattern of the frame they were rendered in; the ring table, dropout and noise of stage 1 are in them, since they come
  from `S4`.
- Shadow removal ignores pasted points hidden by nearer real surfaces.
- The shadow diagnostic uses random placements and an approximate camera-to-LiDAR conversion; its magnitudes are
  indicative.
- About 10 descriptive contrasts uncorrected; one confirmatory test.
