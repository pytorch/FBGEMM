# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from click.testing import CliRunner

# The CLI lives in the source tree rather than the installed Python package.
_spec = importlib.util.spec_from_file_location(
    "sparse_ops_benchmark",
    Path(__file__).resolve().parents[1] / "bench" / "sparse_ops_benchmark.py",
)
assert _spec is not None and _spec.loader is not None
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


class BlockBucketizeBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)

    def test_default_lengths(self) -> None:
        with (
            patch.object(
                torch.ops.fbgemm, "block_bucketize_sparse_features", create=True
            ) as native,
            patch.object(
                benchmark, "benchmark_torch_function", return_value=(1.0, ())
            ) as timer,
        ):
            result = CliRunner().invoke(
                benchmark.block_bucketize_sparse_features_bench,
                [],
                catch_exceptions=False,
            )
        self.assertEqual(result.exit_code, 0, result.output)
        native.assert_not_called()
        self.assertEqual(timer.call_count, 2)
        for call in timer.call_args_list:
            lengths, indices, _, sequence, block_sizes, buckets, weights, *_ = (
                call.args[1]
            )
            values = lengths.tolist()
            self.assertEqual(len(values), 299008)
            self.assertEqual(sum(values), indices.numel())
            self.assertEqual(indices.numel(), 300000)
            self.assertEqual(lengths.dtype, torch.long)
            self.assertEqual(call.kwargs["device"], "cpu")
            self.assertFalse(sequence)
            self.assertEqual(block_sizes.numel(), 2048)
            self.assertEqual(buckets, 16)
            self.assertEqual(weights.numel(), indices.numel())
            self.assertEqual(min(values), 0)
            self.assertLess(max(values), 16)

    def test_all_draws_clipped_to_zero(self) -> None:
        with patch.object(random, "gauss", return_value=-1.1):
            lengths = benchmark._generate_bucketize_lengths(8, 19)
        # The fallback spreads the IDs across all bags even with zero draws.
        self.assertCountEqual(lengths, [2] * 5 + [3] * 3)

    def test_rescaling_surplus_preserves_empty_bags(self) -> None:
        with patch.object(random, "gauss", side_effect=[3.0, -1.0, 3.0, 3.0]):
            lengths = benchmark._generate_bucketize_lengths(4, 5)
        self.assertEqual(lengths[1], 0)
        self.assertCountEqual(lengths, [0, 1, 2, 2])

    def test_invalid_cli_values(self) -> None:
        cases = (
            (["--batch-size", "0"], "--batch-size"),
            (["--lengths-num", "2047"], "after rounding down"),
            (["--indices-num", "299007"], "after rounding down"),
        )
        for args, message in cases:
            with (
                self.subTest(args=args),
                patch.object(
                    torch,
                    "randint",
                    side_effect=AssertionError("unexpected allocation"),
                ),
            ):
                result = CliRunner().invoke(
                    benchmark.block_bucketize_sparse_features_bench,
                    args,
                    catch_exceptions=False,
                )
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn(message, result.output)
