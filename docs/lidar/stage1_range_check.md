# LiDAR stage 1, checks 3.1 and 3.2: range accuracy and self-consistency (2026-10-10)

Registered in `preregistration.md` (section 3, amendment 2) before this was run. Tool: `bin/check_lidar_ranges.py`,
result file `results/lidar_range_check.json`. This is the first time the sampler saw engine depth.

**Setup.** For each wall distance of 5, 10, 20, 30 and 50 m, one scene of a level ground plane and one vertical wall
(60 m half-width, 30 m high) was loaded into the game. The engine's depth was captured from a level camera 1.71 m above
the ground with the same `CaptureObjectMasks` call the datasets use (1920 x 1080), and a scan was sampled with
`src/sensors/depth_lidar.py` (64 channels, 0.1 degree azimuth step, 80 m limit; the registered pattern is the 106 x 74
degree frame, not yet cropped to KITTI's field). Every return's range was compared with the analytic range of the
surface its beam meets, `min(D / d_z, h / d_y)`, which uses only the placed geometry and the camera pose.

## 3.1 Range accuracy (absolute error, 263,947 returns up to 50 m, 289,060 over all bands)

| Range band | Points | Registered sampler (interpolated) median / p99 | Nearest pixel median / p99 |
|---|---|---|---|
| 0 to 10 m | 143,846 | 0.02 mm / 0.07 mm | 1.7 mm / 24.6 mm |
| 10 to 20 m | 50,853 | 0.12 mm / 0.25 mm | 0.13 mm / 97.5 mm |
| 20 to 30 m | 32,493 | 0.46 mm / 0.60 mm | 0.47 mm / 168.7 mm |
| 30 to 50 m | 36,755 | 1.0 mm / 1.5 mm | 1.1 mm / 295.7 mm |
| 50 to 80 m (not part of the rule) | 25,113 | 2.8 mm / 3.9 mm | 2.8 mm / 498 mm |
| **Up to 50 m (the rule)** | 263,947 | **0.04 mm / 1.4 mm** | 0.9 mm / **104.6 mm** |

**Rule** (median at most 20 mm, 99th percentile at most 100 mm, up to 50 m): **passed** with a margin of two orders of
magnitude. The nearest-pixel sampler would have **failed** the 99th-percentile limit (104.6 mm against 100 mm), which is
the reason amendment 2 chose interpolation, now confirmed on engine depth and not only on synthetic planes.

## 3.2 Self-consistency (every wall distance)

- Every point re-projects into the pixel it was read from, except 4 of about 59,000 points per scene whose float32
  position sits on a pixel boundary; none lies outside its pixel (measured distance outside: 0.0 px). **Passed.**
- The depth at each point lies within the min and max of the surrounding 3 x 3 depth pixels for **100%** of points (the
  interpolation never invents a surface). **Passed.**

## What this does and does not show

- It shows the **geometry of the conversion is right** for flat surfaces at known distances, including the sub-millimetre
  agreement of engine depth with placed geometry.
- It does **not** test curved or small objects (cars, pedestrians), where the engine's rasterised depth meets thin parts
  and where there is no independent ground truth for the surface. Check 3.3 (points against the exact 3D boxes of real
  frames) is the test for those, and has not been run.
- The scene's surfaces are fronto-parallel or level: a surface at a grazing angle has a larger depth gradient between
  pixels and is where interpolation is least exact. The ground plane is the grazing case here and passed.
- The pattern used is the default, not the KITTI-matched one of amendment 3 (81.6 degree crop, -24.8 to +2.0 degrees).
  The crop only removes returns; it cannot change a return's range.
