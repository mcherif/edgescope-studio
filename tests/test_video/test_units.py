from __future__ import annotations

import numpy as np

from edgescope.video.metrics import Percentiles, improvement_percent, summarize


def test_summarize_basic_stats() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    stats = summarize(values)
    assert isinstance(stats, Percentiles)
    assert stats.mean == 2.5
    assert stats.p50 == 2.5
    assert np.isclose(stats.p95, np.percentile(values, 95))
    assert np.isclose(stats.p99, np.percentile(values, 99))


def test_summarize_ignores_nan_inf() -> None:
    values = [1.0, 2.0, float("nan"), float("inf"), -float("inf")]
    stats = summarize(values)
    assert stats.mean == 1.5
    assert stats.p50 == 1.5


def test_summarize_empty_or_invalid() -> None:
    empty = summarize([])
    assert empty == Percentiles(0.0, 0.0, 0.0, 0.0)

    invalid = summarize([float("nan"), float("inf"), -float("inf")])
    assert invalid == Percentiles(0.0, 0.0, 0.0, 0.0)


def test_improvement_percent() -> None:
    assert improvement_percent(10.0, 8.0) == 20.0
    assert improvement_percent(10.0, 10.0) == 0.0
    assert improvement_percent(0.0, 10.0) == 0.0
    assert improvement_percent(-1.0, 10.0) == 0.0
