from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from edgescope.video.compositing import BackgroundBlur
    from edgescope.video.metrics import Percentiles, improvement_percent, summarize
except ModuleNotFoundError as e:
    if e.name != "edgescope":
        raise
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from edgescope.video.compositing import BackgroundBlur
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


def test_background_blur_optimized_output_shape_type() -> None:
    h, w = 64, 96
    frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    alpha = np.random.rand(h, w).astype(np.float32)

    blur = BackgroundBlur(
        blur_scale=0.5,
        sigma=2.0,
        comp_mode="soft",
        alpha_path="optimized",
    )

    out1 = blur.apply(frame, alpha)
    out2 = blur.apply(frame, alpha * 0.5)

    assert out1.shape == frame.shape
    assert out1.dtype == np.uint8
    assert out2.shape == frame.shape
    assert out2.dtype == np.uint8
