# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest

import torch
from fbgemm_gpu.bench.bench_utils import benchmark_torch_function


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class BenchUtilsTest(unittest.TestCase):
    def _operands(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Citrine C3: allocate directly on the current accelerator (portable
        # across cuda/rocm/mtia rather than hardcoding "cuda").
        dev = torch.accelerator.current_accelerator()
        a = torch.randn(512, 512, device=dev)
        b = torch.randn(512, 512, device=dev)
        return a, b

    def _assert_sane_per_iter(self, elapsed: float) -> None:
        # Returned value is seconds-per-iteration for both single- and
        # multi-stream paths (the contract callers rely on for batch/elapsed).
        self.assertTrue(math.isfinite(elapsed))
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 1.0)  # a 512x512 mm is well under 1s/iter

    def test_single_thread(self) -> None:
        a, b = self._operands()
        elapsed, _ = benchmark_torch_function(
            _mm, (a, b), iters=20, num_warmups=5, device="cuda", num_threads=1
        )
        self._assert_sane_per_iter(elapsed)

    def test_multi_stream_default(self) -> None:
        # New default: per-stream warmup + wall-clock throughput.
        a, b = self._operands()
        elapsed, _ = benchmark_torch_function(
            _mm, (a, b), iters=40, num_warmups=5, device="cuda", num_threads=2
        )
        self._assert_sane_per_iter(elapsed)

    def test_multi_stream_legacy_still_supported(self) -> None:
        # Legacy event-amortized path preserves the same per-iter contract.
        a, b = self._operands()
        elapsed_default, _ = benchmark_torch_function(
            _mm, (a, b), iters=40, num_warmups=5, device="cuda", num_threads=2
        )
        elapsed_legacy, _ = benchmark_torch_function(
            _mm,
            (a, b),
            iters=40,
            num_warmups=5,
            device="cuda",
            num_threads=2,
            legacy_multi_stream_timing=True,
        )
        self._assert_sane_per_iter(elapsed_legacy)
        # Both measure per-iter time for identical work; they should agree to
        # within an order of magnitude (sanity that neither path is broken).
        self.assertLess(elapsed_default / elapsed_legacy, 10.0)
        self.assertLess(elapsed_legacy / elapsed_default, 10.0)


if __name__ == "__main__":
    unittest.main()
