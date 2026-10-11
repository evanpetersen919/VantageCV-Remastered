"""Paired statistics shared by the studies' decision rules (docs/riders and docs/lidar)."""

import math
from statistics import mean, stdev
from typing import Dict, Sequence

from scipy import stats


def paired_interval(differences: Sequence[float]) -> Dict[str, float]:
    """Mean and 95% t-interval of paired differences (n - 1 degrees of freedom)."""
    n = len(differences)
    centre = mean(differences)
    if n < 2:
        return {"mean": centre, "low": float("nan"), "high": float("nan"), "n": n}
    half = float(stats.t.ppf(0.975, n - 1)) * stdev(differences) / math.sqrt(n)
    return {"mean": centre, "low": centre - half, "high": centre + half, "n": n}
