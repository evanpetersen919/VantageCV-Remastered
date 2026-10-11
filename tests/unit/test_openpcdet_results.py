"""Tests for reading OpenPCDet's KITTI evaluation log (``src/evaluation/openpcdet_results``)."""

import pytest

from src.evaluation import openpcdet_results as parse

LOG = """
2026-10-11 10:00:01,123   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.0000, 80.0000, 70.0000
3d   AP:50.0000, 40.0000, 30.0000
2026-10-11 10:00:01,124   INFO  Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.1200, 90.5000, 88.7100
bev  AP:90.3100, 87.8000, 85.5200
3d   AP:88.2000, 78.3100, 76.7400
aos  AP:95.9000, 90.0100, 88.2000
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.0000, 97.0000, 96.0000
3d   AP:97.0000, 96.0000, 95.0000
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:60.0000, 55.0000, 50.0000
3d   AP:45.0000, 41.5000, 38.0000
Pedestrian AP_R40@0.50, 0.25, 0.25:
3d   AP:70.0000, 65.0000, 60.0000
Cyclist AP_R40@0.50, 0.50, 0.50:
3d   AP:70.5000, 62.2500, 58.0000
"""


def test_first_block_of_each_class_is_read() -> None:
    """The strict-threshold AP_R40 block wins over the R11 block and the loose block."""
    results = parse.parse_log(LOG)
    assert results["Car"]["3d"]["moderate"] == 78.31
    assert results["Pedestrian"]["3d"]["moderate"] == 41.5
    assert results["Cyclist"]["3d"]["easy"] == 70.5


def test_primary_outcome_is_mean_moderate_3d() -> None:
    """M = (78.31 + 41.5 + 62.25) / 3."""
    results = parse.parse_log(LOG)
    assert parse.primary_outcome(results) == pytest.approx((78.31 + 41.5 + 62.25) / 3.0)


def test_missing_class_raises() -> None:
    """A log without Cyclist gives no outcome rather than a mean over two classes."""
    partial = LOG.split("Cyclist", maxsplit=1)[0]
    with pytest.raises(KeyError):
        parse.primary_outcome(parse.parse_log(partial))
