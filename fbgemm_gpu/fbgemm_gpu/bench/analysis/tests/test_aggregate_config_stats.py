#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Tests for the per-iteration total reconstruction used by
``aggregate_config_stats --emit-total``."""

from __future__ import annotations

import unittest

from fbgemm_gpu.bench.analysis.aggregate_config_stats import per_iteration_totals
from fbgemm_gpu.bench.analysis.types import KernelStats


class PerIterationTotalsTest(unittest.TestCase):
    def test_primary_once_plus_sort_twice_per_iter(self) -> None:
        # primary fires 1x/iter (3 iters); sort fires 2x/iter (6 launches).
        totals = per_iteration_totals(
            {"prim": [10.0, 12.0, 11.0], "sort": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]}
        )
        self.assertEqual(totals, [12.0, 16.0, 17.0])

    def test_single_kernel_once_per_iter_is_identity(self) -> None:
        self.assertEqual(
            per_iteration_totals({"prim": [10.0, 12.0, 11.0]}), [10.0, 12.0, 11.0]
        )

    def test_empty_inputs_return_none(self) -> None:
        self.assertIsNone(per_iteration_totals({}))
        self.assertIsNone(per_iteration_totals({"a": []}))

    def test_non_integer_launches_per_iter_returns_none(self) -> None:
        # n_iter = 3 (min length); b has 5 launches (not a multiple of 3).
        # remainder 2 exceeds the small-skew tolerance (max(1, 2% of 3) = 1),
        # so this is treated as a real structural mismatch and returns None.
        self.assertIsNone(
            per_iteration_totals({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
        )

    def test_small_boundary_skew_truncates_trailing_partial(self) -> None:
        # Async GPU + profiler window boundary: the combine kernel is captured
        # one extra time (100 vs 99). remainder 1 is within tolerance
        # (max(1, 2% of 99) = 2), so the trailing partial is dropped and a valid
        # 99-iteration total is produced instead of None.
        main = [1.0] * 99
        combine = [2.0] * 100  # one extra async-tail launch
        totals = per_iteration_totals({"main": main, "combine": combine})
        assert totals is not None
        self.assertEqual(len(totals), 99)
        self.assertTrue(all(t == 3.0 for t in totals))  # 1.0 + 2.0 per iteration

    def test_boundary_skew_scales_with_launches_per_iteration(self) -> None:
        main = [1.0] * 99
        sort = [2.0] * 300  # 3x/iteration plus one iteration's capture tail
        totals = per_iteration_totals({"main": main, "sort": sort})
        assert totals is not None
        self.assertEqual(len(totals), 99)
        self.assertTrue(all(t == 7.0 for t in totals))

    def test_high_launch_rate_rejects_too_many_tail_iterations(self) -> None:
        main = [1.0] * 200
        forty_nine_per_iteration_plus_five_tail_iterations = [2.0] * 9999
        self.assertIsNone(
            per_iteration_totals(
                {
                    "main": main,
                    "high_rate": forty_nine_per_iteration_plus_five_tail_iterations,
                }
            )
        )

    def test_total_median_is_outlier_robust(self) -> None:
        # A single huge outlier inflates the mean/stdev but not the median —
        # exactly the property the report's median column surfaces.
        totals = per_iteration_totals({"prim": [10.0, 100.0, 11.0, 12.0, 13.0]})
        ks = KernelStats(name="(total)", durations_us=totals)
        self.assertEqual(ks.count, 5)
        self.assertEqual(ks.median_us, 12.0)
        self.assertGreater(ks.mean_us, ks.median_us)
