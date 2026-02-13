from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Percentiles:
    mean: float
    p50: float
    p95: float
    p99: float


def summarize(values: Iterable[float]) -> Percentiles:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return Percentiles(0.0, 0.0, 0.0, 0.0)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return Percentiles(0.0, 0.0, 0.0, 0.0)
    return Percentiles(
        mean=float(arr.mean()),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def improvement_percent(baseline: float, new: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - new) / baseline * 100.0
