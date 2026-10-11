"""Apply the registered decision rule of the cyclist study (docs/riders/preregistration.md).

    python bin/analyze_rider_study.py [--results DIR] [--out FILE]

Reads ``results/hpc_rider_<arm>_s<seed>_<benchmark>_riders.json`` for arms A, B, C, E (seeds 0 to
4) and the exploratory sweep arms B250, E250, B500, E500 (seeds 0 to 2). With amendment 2 the
primary outcome M is the mean AP@[.5:.95] of ``rider`` and ``bike`` on BDD100K. Paired by seed,
``d_s = M_E(s) - M_B(s)`` and the rule of section 5 is evaluated:

1. the mean of d_s is above 0;
2. the 95% paired t-interval (4 degrees of freedom) excludes 0;
3. a paired bootstrap over validation images gives an interval that excludes 0 (**not computed
   here**: it needs each model's per-image detections, which the result files do not contain; the
   verdict is therefore provisional until the detections are dumped and re-scored);
4. the guards: mean car AP and mean person AP of E each no more than 1.0 AP below A's.

"Hurts" is both intervals (2 and 3) entirely below 0; otherwise the result is inconclusive.
Nothing is tuned after seeing results. Reported, not decisive: E against A and C, every class, the
sweep, Cityscapes.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Sequence

from scipy import stats

ARMS = ("A", "B", "C", "E")
SEEDS = (0, 1, 2, 3, 4)
SWEEP_ARMS = ("B250", "E250", "B500", "E500")
SWEEP_SEEDS = (0, 1, 2)
PRIMARY = ("rider", "bike")
GUARDS = ("car", "person")
GUARD_TOLERANCE_AP = 1.0
CLASSES = ("rider", "bike", "motor", "person", "car", "bus", "truck")


def load(results: Path, arm: str, seed: int, benchmark: str) -> Dict[str, float]:
    """Per-class AP@[.5:.95] of one run, in AP points (0 to 100)."""
    path = results / f"hpc_rider_{arm}_s{seed}_{benchmark}_riders.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: 100.0 * v for k, v in data["overall"]["per_class_ap"].items()}


def primary(per_class: Dict[str, float]) -> float:
    """M: the mean AP of rider and bike."""
    return mean(per_class[name] for name in PRIMARY)


def paired_interval(differences: Sequence[float]) -> Dict[str, float]:
    """Mean and 95% t-interval of paired differences (n - 1 degrees of freedom)."""
    n = len(differences)
    centre = mean(differences)
    spread = stdev(differences) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.975, n - 1)) * spread / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": centre, "low": centre - half, "high": centre + half, "n": n}


def verdict(interval: Dict[str, float], guards_hold: bool) -> str:
    """The provisional verdict from the t-interval and guards (the bootstrap is pending)."""
    if interval["low"] > 0 and interval["mean"] > 0 and guards_hold:
        return "helps by the t-interval and guards; bootstrap (criterion 3) pending"
    if interval["high"] < 0:
        return "hurts by the t-interval; bootstrap (criterion 2/3) pending"
    return "inconclusive: the t-interval includes 0 or a guard failed"


def analyse(results: Path, benchmark: str = "bdd100k") -> Dict[str, Any]:
    """The whole analysis for one benchmark as a JSON-serialisable dictionary."""
    runs = {a: {s: load(results, a, s, benchmark) for s in SEEDS} for a in ARMS}
    m = {a: [primary(runs[a][s]) for s in SEEDS] for a in ARMS}
    per_class = {a: {c: mean(runs[a][s][c] for s in SEEDS) for c in CLASSES} for a in ARMS}
    d = [m["E"][i] - m["B"][i] for i in range(len(SEEDS))]
    interval = paired_interval(d)
    guards = {c: per_class["E"][c] - per_class["A"][c] for c in GUARDS}
    guards_hold = all(value >= -GUARD_TOLERANCE_AP for value in guards.values())
    sweep: Dict[str, Any] = {}
    for arm in SWEEP_ARMS:
        values = [primary(load(results, arm, s, benchmark)) for s in SWEEP_SEEDS]
        sweep[arm] = {"M_mean": mean(values), "M_by_seed": values}
    return {
        "benchmark": benchmark,
        "M_by_seed": m,
        "M_mean": {a: mean(m[a]) for a in ARMS},
        "d_E_minus_B": d,
        "d_interval": interval,
        "guards_E_minus_A": guards,
        "guards_hold": guards_hold,
        "verdict": verdict(interval, guards_hold),
        "E_minus_A": mean(m["E"]) - mean(m["A"]),
        "E_minus_C": mean(m["E"]) - mean(m["C"]),
        "per_class_mean": per_class,
        "sweep": sweep,
    }


def render(analysis: Dict[str, Any]) -> List[str]:
    """A plain-text report of ``analysis``."""
    interval = analysis["d_interval"]
    lines = [
        f"Cyclist study on {analysis['benchmark']} (M = mean AP of rider and bike)",
        "M by arm: " + ", ".join(f"{a} {v:.2f}" for a, v in analysis["M_mean"].items()),
        "d = M_E - M_B by seed: " + ", ".join(f"{v:+.2f}" for v in analysis["d_E_minus_B"]),
        f"mean d {interval['mean']:+.2f}, 95% t-interval [{interval['low']:+.2f}, "
        f"{interval['high']:+.2f}] (n = {int(interval['n'])})",
        "guards E - A: "
        + ", ".join(f"{c} {v:+.2f}" for c, v in analysis["guards_E_minus_A"].items())
        + f" (hold: {analysis['guards_hold']})",
        f"VERDICT (provisional): {analysis['verdict']}",
        f"E - A {analysis['E_minus_A']:+.2f}; E - C {analysis['E_minus_C']:+.2f}",
        "per class mean AP:",
    ]
    for arm in ARMS:
        classes = analysis["per_class_mean"][arm]
        lines.append(f"  {arm}: " + ", ".join(f"{c} {classes[c]:.2f}" for c in CLASSES))
    lines.append(
        "sweep M: " + ", ".join(f"{a} {v['M_mean']:.2f}" for a, v in analysis["sweep"].items())
    )
    return lines


def main() -> None:
    """Parse arguments, analyse both benchmarks, print and write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/rider_study_analysis.json"))
    args = parser.parse_args()
    analyses = {b: analyse(args.results, b) for b in ("bdd100k", "cityscapes")}
    for analysis in analyses.values():
        print("\n".join(render(analysis)))
        print()
    args.out.write_text(json.dumps(analyses, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
