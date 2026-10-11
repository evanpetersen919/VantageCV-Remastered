"""Paired bootstrap over validation images for the cyclist study (preregistration, criterion 3).

    python bin/bootstrap_rider_study.py --tables results/matches --results results \\
        --resamples 10000 --workers 4 --out results/rider_study_bootstrap.json

Inputs are the per-detection match tables written by
``bin/evaluate_detector.py --save-matches``
(``hpc_rider_<arm>_s<seed>_bdd100k_matches.npz``) for arms B and E, seeds 0 to 4. Each resample
draws the validation images with replacement, the same draw for every model (so the comparison is
paired), recomputes M = mean AP@[.5:.95] of rider and bike for each model, averages M over the
five seeds per arm, and records the difference E - B. The 95% interval is the 2.5th and 97.5th
percentiles of the differences.

Before resampling, each model's full-sample AP from the table is compared with the AP in its
published results file (``--results``); the run stops if they differ by more than ``TOLERANCE``,
since the table must be the same measurement as the score it stands in for.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.evaluation.match_table import average_precision, load_table

ARMS = ("B", "E")
SEEDS = (0, 1, 2, 3, 4)
RIDER_ID, BIKE_ID = 10, 2  # category ids of the riders profile
NAMES = {RIDER_ID: "rider", BIKE_ID: "bike"}
TOLERANCE = 5e-4  # AP (0 to 1 scale): table against the published results file

_TABLES: Dict[Tuple[str, int], Dict[str, Any]] = {}


def _load_all(directory: Path) -> None:
    for arm in ARMS:
        for seed in SEEDS:
            path = directory / f"hpc_rider_{arm}_s{seed}_bdd100k_matches.npz"
            _TABLES[(arm, seed)] = load_table(path)


def _m(arm: str, seed: int, weights: Any) -> float:
    classes = _TABLES[(arm, seed)]["classes"]
    return float(np.mean([average_precision(classes[c], weights) for c in (RIDER_ID, BIKE_ID)]))


def _difference(weights: Any) -> float:
    m_e = np.mean([_m("E", s, weights) for s in SEEDS])
    m_b = np.mean([_m("B", s, weights) for s in SEEDS])
    return float(m_e - m_b)


def _chunk(job: Tuple[str, int, int, int]) -> List[float]:
    directory, seed, count, images = job
    _load_all(Path(directory))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        draw = rng.integers(0, images, images)
        out.append(_difference(np.bincount(draw, minlength=images).astype(np.float64)))
    return out


def validate(directory: Path, results: Path) -> Dict[str, float]:
    """Largest gap between a table's full-sample AP and its published per-class AP."""
    _load_all(directory)
    worst = 0.0
    for (arm, seed), table in _TABLES.items():
        published = json.loads(
            (results / f"hpc_rider_{arm}_s{seed}_bdd100k_riders.json").read_text("utf-8")
        )["overall"]["per_class_ap"]
        for category, name in NAMES.items():
            gap = abs(average_precision(table["classes"][category]) - published[name])
            worst = max(worst, gap)
    return {"max_abs_gap": worst, "tolerance": TOLERANCE}


def main() -> None:
    """Validate, resample, summarise and write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/rider_study_bootstrap.json"))
    args = parser.parse_args()

    check = validate(args.tables, args.results)
    print("table against published AP:", check, flush=True)
    if check["max_abs_gap"] > TOLERANCE:
        raise SystemExit("match tables do not reproduce the published AP; stopping")
    images = len(next(iter(_TABLES.values()))["images"])
    full = _difference(np.ones(images))
    per_worker = -(-args.resamples // args.workers)
    jobs = [
        (str(args.tables), args.seed * 1000 + k, per_worker, images) for k in range(args.workers)
    ]
    with Pool(args.workers) as pool:
        draws = [d for chunk in pool.map(_chunk, jobs) for d in chunk][: args.resamples]
    values = np.array(draws)
    summary = {
        "resamples": int(len(values)),
        "images": images,
        "full_sample_E_minus_B": full * 100.0,
        "bootstrap_mean": float(values.mean()) * 100.0,
        "ci95_low": float(np.percentile(values, 2.5)) * 100.0,
        "ci95_high": float(np.percentile(values, 97.5)) * 100.0,
        "share_above_zero": float((values > 0).mean()),
        "validation": check,
        "unit": "AP points (x100)",
    }
    args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
