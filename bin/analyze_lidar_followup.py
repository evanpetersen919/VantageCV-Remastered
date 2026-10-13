"""Apply the LiDAR follow-up's registered rule and contrasts.

The design is in docs/lidar/followup_preregistration.md.

    python bin/analyze_lidar_followup.py [--results DIR] [--out FILE]

Reads ``results/lidar_<arm>_s<seed>_kitti_val.json`` for every arm of the follow-up (seeds 0 to 4).
The single confirmatory test is ``E4 - A`` on M, the mean moderate 3D AP of Car, Pedestrian and
Cyclist: the mean of the paired differences is above 0, its 95% paired t-interval (4 degrees of
freedom) excludes 0, and E4's Car AP is no more than 1.0 AP below A's. Every other contrast is
descriptive, reported with its paired interval on M and on each class.
"""

# pylint: disable=duplicate-code
# The report layout mirrors the LiDAR study's analysis by design.

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

from src.evaluation.paired_stats import paired_interval

SEEDS = (0, 1, 2, 3, 4)
CLASSES = ("Car", "Pedestrian", "Cyclist")
GUARD_TOLERANCE_AP = 1.0
CONFIRMATORY = ("E4", "A")

# (left, right) pairs reported as left - right; grouped by the question each answers.
CONTRASTS: Tuple[Tuple[str, Sequence[Tuple[str, str]]], ...] = (
    ("confirmatory and reference", (("E4", "A"), ("E4", "B"), ("E0", "A"))),
    ("chain (amendment 1 order)", (("E1", "E0"), ("E2", "E1"), ("E3", "E2"), ("E4", "E3"))),
    ("chain without the footprint", (("E3a", "E1"), ("E4a", "E3a"), ("E3", "E3a"), ("E4", "E4a"))),
    ("density at equal frame count", (("E1lo", "E1rnd"), ("E1lo", "A"), ("E1rnd", "A"))),
    ("dilution (H6)", (("E1x3", "E1"), ("E1x3", "Ax3"), ("Ax3", "A"), ("E1", "A"))),
    ("intensity off", (("E0i", "Ai"), ("E3i", "Ai"), ("E3i", "E3"), ("Bi", "Ai"))),
    # Not registered: added after Ax3 - A showed repetition alone helps, so A under-trains.
    (
        "post hoc, not registered (step-matched baseline)",
        (("B", "Ax3"), ("E0", "Ax3"), ("E1", "Ax3")),
    ),
)
# The improvement study (docs/lidar/improvement_preregistration.md): confirmatory test S4 - Astep.
IMPROVEMENT_CONFIRMATORY = ("S4", "Astep")
IMPROVEMENT_CONTRASTS: Tuple[Tuple[str, Sequence[Tuple[str, str]]], ...] = (
    ("confirmatory and reference", (("S4", "Astep"), ("S4", "A"), ("S4", "B"), ("S4", "Ax3"))),
    (
        "H7 ring table, H8 dropout and noise, H9 minimum points",
        (
            ("S1", "E3a"),
            ("S2", "S1"),
            ("S3", "S1"),
            ("S4", "S2"),
            ("S4", "S3"),
            ("S4", "E3a"),
        ),
    ),
    ("H10 real-only ground-truth database", (("E3a_rdb", "E3a"), ("S4_rdb", "S4"))),
    ("H11 domain flag", (("E4_flag", "E4"), ("E0i_flag", "E0i"))),
    ("H12 matched steps", (("Astep", "A"), ("E1x3", "E1step"), ("E1step", "E1"), ("E1x3", "Ax3"))),
    ("post hoc, not registered", (("B", "Astep"), ("E3a", "Astep"), ("S4_rdb", "Astep"))),
)
STUDIES = {
    "followup": (CONTRASTS, CONFIRMATORY, "results/lidar_followup_analysis.json"),
    "improvement": (
        IMPROVEMENT_CONTRASTS,
        IMPROVEMENT_CONFIRMATORY,
        "results/lidar_improvement_analysis.json",
    ),
}
ARM_NAMES = {
    "E0": "E",  # the first study's arm E is E0 in the follow-up chain
}


def load(results: Path, arm: str, seed: int) -> Dict[str, Any]:
    """One run's result file (``M`` and the per-class AP tables)."""
    name = ARM_NAMES.get(arm, arm)
    path = results / f"lidar_{name}_s{seed}_kitti_val.json"
    loaded: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def series(results: Path, arm: str) -> Dict[str, List[float]]:
    """``{"M": [...], "Car": [...], ...}`` over seeds for one arm."""
    runs = [load(results, arm, s) for s in SEEDS]
    out: Dict[str, List[float]] = {"M": [float(r["M"]) for r in runs]}
    for name in CLASSES:
        out[name] = [float(r["classes"][name]["3d"]["moderate"]) for r in runs]
    return out


def contrast(left: Dict[str, List[float]], right: Dict[str, List[float]]) -> Dict[str, Any]:
    """Paired interval of ``left - right`` on M and on each class."""
    return {
        key: paired_interval([a - b for a, b in zip(left[key], right[key])])
        for key in ("M",) + CLASSES
    }


def analyse(results: Path, study: str = "followup") -> Dict[str, Any]:
    """The whole analysis as a JSON-serialisable dictionary."""
    contrasts, confirmatory, _ = STUDIES[study]
    names = sorted({arm for _, pairs in contrasts for pair in pairs for arm in pair})
    data = {arm: series(results, arm) for arm in names}
    groups: Dict[str, Any] = {}
    for title, pairs in contrasts:
        groups[title] = {f"{a} - {b}": contrast(data[a], data[b]) for a, b in pairs}
    left, right = confirmatory
    primary = groups["confirmatory and reference"][f"{left} - {right}"]["M"]
    guard = mean(data[left]["Car"]) - mean(data[right]["Car"])
    guard_holds = guard >= -GUARD_TOLERANCE_AP
    helps = primary["low"] > 0 and guard_holds
    return {
        "arm_means": {a: {k: mean(v) for k, v in data[a].items()} for a in names},
        "contrasts": groups,
        "confirmatory": {
            "test": f"{left} - {right}",
            "interval": primary,
            "car_guard": guard,
            "guard_holds": guard_holds,
            "supported": helps,
        },
    }


def render(analysis: Dict[str, Any]) -> List[str]:
    """A plain-text report of ``analysis``."""
    lines = ["Arm means (moderate 3D AP; M = mean of the three classes)"]
    for arm, values in analysis["arm_means"].items():
        parts = ", ".join(f"{k} {values[k]:.2f}" for k in ("M",) + CLASSES)
        lines.append(f"  {arm:6s} {parts}")
    for title, rows in analysis["contrasts"].items():
        lines.append(f"\n{title}")
        for name, intervals in rows.items():
            cells = []
            for key in ("M",) + CLASSES:
                i = intervals[key]
                cells.append(f"{key} {i['mean']:+.2f} [{i['low']:+.2f}, {i['high']:+.2f}]")
            lines.append(f"  {name:14s} " + "  ".join(cells))
    c = analysis["confirmatory"]
    i = c["interval"]
    lines.append(
        f"\nCONFIRMATORY {c['test']}: {i['mean']:+.2f} [{i['low']:+.2f}, {i['high']:+.2f}], "
        f"Car guard {c['car_guard']:+.2f} (holds {c['guard_holds']}) -> "
        f"{'supported' if c['supported'] else 'not supported'}"
    )
    return lines


def main() -> None:
    """Parse arguments, analyse, print and write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--study", choices=sorted(STUDIES), default="followup")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    analysis = analyse(args.results, args.study)
    args.out = args.out or Path(STUDIES[args.study][2])
    print("\n".join(render(analysis)))
    args.out.write_text(json.dumps(analysis, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
