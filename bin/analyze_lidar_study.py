"""Apply the registered decision rule of the LiDAR study (docs/lidar/preregistration.md, section 4).

    python bin/analyze_lidar_study.py [--results DIR] [--out FILE]

Reads ``results/lidar_<arm>_s<seed>_kitti_val.json`` (written by ``hpc/lidar/run_lidar_arm.sbatch``
from OpenPCDet's evaluation log) for arms A, B, E (seeds 0 to 4) and the exploratory sweep arms.
The primary outcome M is the mean moderate 3D AP of Car, Pedestrian and Cyclist on KITTI's
validation frames. Paired by seed, ``d_s = M_E(s) - M_B(s)``; the rule of section 4 is:

1. the mean of d_s is above 0;
2. the 95% paired t-interval (4 degrees of freedom) excludes 0;
3. a paired bootstrap over validation scans excludes 0 (**not computed here**: it needs
   per-scan detections; the verdict is provisional without it);
4. the guard: E's mean Car 3D AP is no more than 1.0 AP below A's.

"Hurts" is both intervals below 0; otherwise inconclusive. Nothing is tuned after seeing results.
"""

# pylint: disable=duplicate-code
# The argument parsing and the report layout mirror the cyclist study's analysis by design.

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.evaluation.paired_stats import paired_interval

ARMS = ("A", "B", "E")
SEEDS = (0, 1, 2, 3, 4)
SWEEP_ARMS = ("B250", "E250", "B500", "E500")
SWEEP_SEEDS = (0, 1, 2)
CLASSES = ("Car", "Pedestrian", "Cyclist")
GUARD_TOLERANCE_AP = 1.0


def load(results: Path, arm: str, seed: int) -> Dict[str, Any]:
    """One run's result file (``M`` and the per-class AP tables)."""
    path = results / f"lidar_{arm}_s{seed}_kitti_val.json"
    loaded: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def class_ap(run: Dict[str, Any], name: str) -> float:
    """Moderate 3D AP of ``name`` in one run."""
    return float(run["classes"][name]["3d"]["moderate"])


def verdict(interval: Dict[str, float], guard_holds: bool) -> str:
    """The provisional verdict from the t-interval and the guard."""
    if interval["low"] > 0 and guard_holds:
        return "helps by the t-interval and guard; bootstrap (criterion 3) pending"
    if interval["high"] < 0:
        return "hurts by the t-interval; bootstrap pending"
    return "inconclusive: the t-interval includes 0 or the guard failed"


def analyse(results: Path) -> Dict[str, Any]:
    """The whole analysis as a JSON-serialisable dictionary."""
    runs = {a: {s: load(results, a, s) for s in SEEDS} for a in ARMS}
    m = {a: [float(runs[a][s]["M"]) for s in SEEDS] for a in ARMS}
    per_class = {
        a: {c: mean(class_ap(runs[a][s], c) for s in SEEDS) for c in CLASSES} for a in ARMS
    }
    d = [m["E"][i] - m["B"][i] for i in range(len(SEEDS))]
    interval = paired_interval(d)
    guard = per_class["E"]["Car"] - per_class["A"]["Car"]
    guard_holds = guard >= -GUARD_TOLERANCE_AP
    sweep: Dict[str, Any] = {}
    for arm in SWEEP_ARMS:
        values = [float(load(results, arm, s)["M"]) for s in SWEEP_SEEDS]
        sweep[arm] = {"M_mean": mean(values), "M_by_seed": values}
    return {
        "M_by_seed": m,
        "M_mean": {a: mean(m[a]) for a in ARMS},
        "d_E_minus_B": d,
        "d_interval": interval,
        "car_guard_E_minus_A": guard,
        "guard_holds": guard_holds,
        "verdict": verdict(interval, guard_holds),
        "E_minus_A": mean(m["E"]) - mean(m["A"]),
        "per_class_mean": per_class,
        "sweep": sweep,
    }


def render(analysis: Dict[str, Any]) -> List[str]:
    """A plain-text report of ``analysis``."""
    interval = analysis["d_interval"]
    lines = [
        "LiDAR study on KITTI validation (M = mean moderate 3D AP of Car, Pedestrian, Cyclist)",
        "M by arm: " + ", ".join(f"{a} {v:.2f}" for a, v in analysis["M_mean"].items()),
        "d = M_E - M_B by seed: " + ", ".join(f"{v:+.2f}" for v in analysis["d_E_minus_B"]),
        f"mean d {interval['mean']:+.2f}, 95% t-interval [{interval['low']:+.2f}, "
        f"{interval['high']:+.2f}] (n = {int(interval['n'])})",
        f"guard (Car E - A): {analysis['car_guard_E_minus_A']:+.2f} "
        f"(holds: {analysis['guard_holds']})",
        f"VERDICT (provisional): {analysis['verdict']}",
        f"E - A: {analysis['E_minus_A']:+.2f}",
    ]
    for arm, classes in analysis["per_class_mean"].items():
        lines.append(f"  {arm}: " + ", ".join(f"{c} {classes[c]:.2f}" for c in CLASSES))
    lines.append(
        "sweep M: " + ", ".join(f"{a} {v['M_mean']:.2f}" for a, v in analysis["sweep"].items())
    )
    return lines


def main() -> None:
    """Parse arguments, analyse, print and write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/lidar_study_analysis.json"))
    args = parser.parse_args()
    analysis = analyse(args.results)
    print("\n".join(render(analysis)))
    args.out.write_text(json.dumps(analysis, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
