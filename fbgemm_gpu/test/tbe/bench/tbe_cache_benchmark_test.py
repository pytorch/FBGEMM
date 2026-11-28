# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import runpy
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import torch

# Benchmark scripts live in the source tree, not in the installed package.
_BENCHMARK = (
    Path(__file__).resolve().parents[3] / "bench" / "tbe" / "tbe_cache_benchmark.py"
)
_CREATE_REQUEST: Callable[[int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = (
    runpy.run_path(str(_BENCHMARK))["create_request"]
)


class CacheBenchmarkRequestTest(unittest.TestCase):
    def _bag_lengths(self, tables: int, batch: int, avg: int) -> list[int]:
        rows = 17
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        with (
            patch.object(torch.Tensor, "cuda", lambda tensor: tensor),
            torch.random.fork_rng(devices=[]),
        ):
            random.seed(0)
            torch.random.default_generator.manual_seed(0)
            indices, offsets = _CREATE_REQUEST(tables, rows, batch, avg)

        total = tables * batch * avg
        self.assertEqual(indices.shape, (total,))
        self.assertEqual(offsets.shape, (tables * batch + 1,))
        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(offsets.dtype, torch.int32)
        ends = offsets.tolist()
        self.assertEqual(ends[0], 0)
        lengths = [end - start for start, end in zip(ends, ends[1:])]
        self.assertTrue(all(length > 0 for length in lengths), lengths)
        self.assertEqual(ends[-1], total)
        self.assertTrue(((indices >= 0) & (indices < rows)).all().item())
        return lengths

    def test_batch_of_one(self) -> None:
        self._bag_lengths(tables=4, batch=1, avg=2)

    def test_unit_pooling(self) -> None:
        self.assertEqual(self._bag_lengths(tables=2, batch=3, avg=1), [1] * 6)

    def test_table_samples_over_budget(self) -> None:
        with patch.object(random, "choices", return_value=[32, 4, 1]):
            lengths = self._bag_lengths(tables=3, batch=3, avg=4)
        self.assertEqual(lengths[6:], [1] * 3)
        self.assertGreater(sum(lengths[:3]), sum(lengths[3:6]))

    def test_table_samples_under_budget(self) -> None:
        with patch.object(random, "choices", return_value=[32] + [1] * 11):
            lengths = self._bag_lengths(tables=12, batch=1, avg=4)
        self.assertEqual(lengths[0], 32)
        self.assertGreater(max(lengths[1:]), 1)

    def test_gaussian_draws_are_bounded(self) -> None:
        with patch.object(random, "gauss", side_effect=[-1e6, 1e6, 1e6]):
            self.assertEqual(self._bag_lengths(tables=1, batch=4, avg=3), [1, 9, 1, 1])

    def test_invalid_offset_budget(self) -> None:
        for avg in (0, 2**31):
            with (
                self.subTest(avg=avg),
                patch.object(
                    torch, "randint", side_effect=AssertionError("allocated indices")
                ),
                self.assertRaises(ValueError),
            ):
                _CREATE_REQUEST(1, 17, 1, avg)
