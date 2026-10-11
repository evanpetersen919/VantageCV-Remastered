# The v18 rider batch: label audit and side-by-side with BDD100K (2026-10-10)

The synthetic arm E of the rider study (`preregistration.md`, sections 3 and 6) is drawn from this batch. This is the
audit and the match table the preregistration promised, written before any detector is trained on it. Nothing here is
a gate; section 6 asks for the table and a count of hard defects, and both are reported as measured.

**Batch.** `live_dataset/rider_v18`: `urban_dense_v18.yaml`, profile v7, `--views rider rider rider rider`, 350
scenarios (seeds 80000 to 80349), 1,399 frames of 1920 x 1080, exact labels, semantic and depth maps, `--classes
riders`, annotation cutoff 50 m for rider and bike (amendment 3). No frame was rejected; the camera self-check
reprojection error was 0.52 px. 1,196 frames carry a rider or bike label (1,189 a rider, 1,124 a bike), above S = 1,000.

## Hard defects (counted over 1,353 rider and 1,257 bike labels)

| Check | Failures |
|---|---|
| Missing mask | 0 |
| Empty mask | 0 |
| Visibility outside (0, 1] | 0 |
| Box outside the image | 0 |
| **Box not the tight box of its mask (> 1 px)** | **10 (0.38% of labels), all bikes** |

The 10 are bike boxes larger than their masks by 2 to 5 px on a side, on boxes 8 to 48 px tall, with visibility 0.2 to
0.9. Their extra box pixels are 893 in total; **196 of them (22%) are in another labelled object's mask** (mostly the
rider: a depth tie at a contact point goes to one object in the class map) and **697 (78%) belong to no labelled object**.
The 78% is not explained: the box comes from each object's own depth against the scene depth, the mask from the
priority-resolved class map, and the two can differ where an unlabelled surface (a pole, a tree trunk) ties with the
bike. This was not investigated further. It affects 10 of 2,610 labels, all small and partly hidden; the audit script
exits non-zero because of them and that is expected.

## Match to BDD100K (train), medians and percentiles measured on both sides

| Quantity | BDD100K | v18 rider batch | Reading |
|---|---|---|---|
| Rider box height, 720p equivalent, p5 | 16 | 14.7 | close |
| p25 | 33.8 | 34.7 | close |
| p50 | 59 | 64.0 | close |
| p75 | 102 | **124.7** | 22% taller |
| p95 | 237 | 214 | a little low |
| Riders per image containing one | 1.26 | 1.14 | slightly fewer |
| Bikes per image containing one | 1.66 | 1.12 | **fewer** |
| Rider box bottom edge, share of image height (median) | 395/720 = 0.55 | 0.55 | matches |
| Rider box width / height, median | 1 / 1.87 = 0.53 | **0.43** | narrower (more views from behind) |
| Truncated boxes | 5.0% (riders) | **0.5%** | far fewer |
| **Unridden bikes** (no rider box overlapping 30%) | **65.2%** | **2.4%** (30 of 1,257) | **structurally different** |
| Riders with no overlapping bike | not measured | 9.3% (126 of 1,353) | bike hidden |
| Day / night images | not matched here | 819 / 580 | |

## Gaps stated before training

1. **The synthetic bikes are almost all ridden; BDD's are mostly parked.** 65% of BDD100K bike boxes have no rider on
   them (parked bikes, bikes on racks, carried). This generator places only cyclists. A detector trained on arm E sees
   bikes only with a person on them, so its `bike` AP on real parked bikes may not benefit, and the headroom check's
   real extras (which contain parked bikes) carry that advantage in arm B. This is a property of the data, not a bug,
   and a reason the study could return a mixed result: `rider` could gain where `bike` does not. Parked bikes are not
   built and are not part of release 1.4.
2. **Boxes are less often truncated and more from behind.** The views are aimed at the cyclist and the trunk-avoidance
   rule keeps it away from the frame edge.
3. **The upper quartile of rider sizes is high** (p75 125 against 102). A likely cause is that far, small riders are lost
   more often than near ones to occlusion and the 50 m cutoff; this was not tested.
4. The 10 non-tight bike boxes above.

Files: `results/rider_v18_label_audit.json`; the audit tool is `bin/audit_cyclist_labels.py`.
