#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Pure-function statistics used for cross-commit benchmark comparisons.

All functions are deterministic and depend only on summary statistics
``(n, mean, stddev)`` per group — no raw samples required. This makes them
suitable for driving the Google Doc benchmark report without exposing raw
trace durations.

Sign convention throughout this module: *before* is group 1, *after* is
group 2. A positive ``pct_diff`` means *after* is slower.

Callers MUST go through these functions instead of recomputing stats
inline so that every number in the benchmark report can be traced back to
a single, tested source.
"""

from __future__ import annotations

import math

from scipy import stats

# Minimum number of hypothesis tests in a group before we bother running
# Benjamini-Hochberg. Below this, the correction is noisy and provides
# little value.
BH_MIN_TESTS: int = 6


def pct_diff(mean_before: float, mean_after: float) -> float:
    """Percentage change ``(after - before) / before * 100``.

    Returns ``nan`` when the input is not a usable baseline (zero,
    negative, or non-finite).
    """
    if (
        not math.isfinite(mean_before)
        or not math.isfinite(mean_after)
        or mean_before <= 0.0
    ):
        return float("nan")
    return (mean_after - mean_before) / mean_before * 100.0


def welch_t_p(
    n1: int,
    mean1: float,
    stdev1: float,
    n2: int,
    mean2: float,
    stdev2: float,
) -> tuple[float, float, float]:
    """Welch's two-sample t-test from summary statistics.

    Returns ``(t, p, df)`` where ``t`` is Welch's t-statistic (with sign
    positive when *before* is greater than *after*), ``p`` is the two-sided
    p-value, and ``df`` is Welch–Satterthwaite degrees of freedom.

    Returns ``(nan, nan, nan)`` when either group has insufficient data
    (``n <= 1``) or zero pooled variance.
    """
    if n1 <= 1 or n2 <= 1:
        return float("nan"), float("nan"), float("nan")
    if not all(math.isfinite(v) for v in (mean1, mean2, stdev1, stdev2)):
        return float("nan"), float("nan"), float("nan")

    var1 = stdev1 * stdev1
    var2 = stdev2 * stdev2
    se2 = var1 / n1 + var2 / n2
    if se2 <= 0.0:
        return float("nan"), float("nan"), float("nan")

    se = math.sqrt(se2)
    t_stat = (mean1 - mean2) / se

    num = se2 * se2
    denom = (var1 * var1) / (n1 * n1 * (n1 - 1)) + (var2 * var2) / (n2 * n2 * (n2 - 1))
    df = num / denom if denom > 0.0 else float("nan")

    if not math.isfinite(df) or df <= 0.0:
        return t_stat, float("nan"), df

    p = float(stats.t.sf(abs(t_stat), df) * 2.0)
    return t_stat, p, df


def cohen_d(
    mean1: float,
    stdev1: float,
    mean2: float,
    stdev2: float,
) -> float:
    """Cohen's d (pooled, equal-weighted) from summary statistics.

    ``d = (mean2 - mean1) / sqrt((stdev1^2 + stdev2^2) / 2)``. Positive
    values mean *after* is greater than *before*.

    Returns ``nan`` when pooled std is zero or inputs are non-finite.
    """
    if not all(math.isfinite(v) for v in (mean1, mean2, stdev1, stdev2)):
        return float("nan")
    pooled_var = (stdev1 * stdev1 + stdev2 * stdev2) / 2.0
    if pooled_var <= 0.0:
        return float("nan")
    return (mean2 - mean1) / math.sqrt(pooled_var)


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini–Hochberg adjusted q-values.

    Applies BH FDR correction to ``p_values`` and returns a list of the
    same length with the adjusted q-values. Preserves input order.

    Returns a list of ``nan`` values with the same length when:
        * the input is empty,
        * the number of finite, non-null entries is below
          :data:`BH_MIN_TESTS` (correction is noisy for few tests),
        * all entries are non-finite.

    Non-finite entries are passed through as ``nan``. The correction is
    computed over the finite entries; ties are handled by
    ``scipy.stats.false_discovery_control``.
    """
    n = len(p_values)
    if n == 0:
        return []

    finite_indices = [
        i
        for i, p in enumerate(p_values)
        if isinstance(p, (int, float)) and math.isfinite(p)
    ]
    if len(finite_indices) < BH_MIN_TESTS:
        return [float("nan")] * n

    finite_pvals = [float(p_values[i]) for i in finite_indices]
    adjusted = stats.false_discovery_control(finite_pvals, method="bh")

    out = [float("nan")] * n
    for idx, q in zip(finite_indices, adjusted):
        out[idx] = float(q)
    return out
