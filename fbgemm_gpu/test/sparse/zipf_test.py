#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest

import torch

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_memory_lt_gb, gpu_unavailable, optests
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_memory_lt_gb, gpu_unavailable, optests


class ZipfTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # Skip on GPUs with insufficient HBM. The test allocates int64[n] for
    # n = 2**32 + 1 (~32 GiB), so it needs a large-HBM GPU. Gate at 64 GiB to
    # avoid OOM/timeouts on 36-42 GiB CI GPUs (it only needs to run somewhere).
    @unittest.skipIf(*gpu_memory_lt_gb(64))
    # large-grid CUDA-only stress repro (allocates ~32 GiB); the generated
    # opcheck variants add no op-schema coverage and only produce FAILURE/
    # SKIPPING test-health records on CPU/small-GPU runs (T191384137).
    @optests.dontGenerateOpCheckTests(
        "large-grid CUDA-only stress repro; opcheck variants add no coverage (T191384137)"
    )
    def test_zipf_large_n_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in zipf_cuda and validates
        the kernel's output via Tier-C structural invariants (no CPU
        dispatch exists for ``zipf_cuda`` — the op is registered as a
        single CUDA-only function pointer via ``DISPATCH_TO_ALL`` and a
        Python reference cannot match the kernel's draws because the
        kernel uses cuRAND with a per-thread sub-sequence stride).

        With kMaxThreads=1024, the launch grid is
        cuda_calc_xblock_count(n, 1024). For n > 2**32, total threads
        exceed the HIP 2**32 limit, causing FBGEMM_LAUNCH_KERNEL ->
        KernelLauncher::checkThreadCountNotExceeded to TORCH_CHECK-fail
        on ROCm. zipf_kernel already grid-strides over the output index,
        so the post-fix capped grid still covers all n elements.

        Tier-C invariants checked: shape, dtype, plus an O(1) sentinel
        check on the large output (start / middle / end positions are
        non-negative). Heavier invariants (full non-negativity,
        finiteness, seed-determinism, uniqueness) are validated at a
        small scale to keep the opcheck-amplified cost bounded — each
        opcheck variant (test_schema, test_faketensor,
        test_aot_dispatch_dynamic) re-runs this test body, so any
        full-scale reduction is multiplied 4×.

        Note: This test allocates int64[n] (~32 GiB at the chosen n), so
        it is skipped on GPUs without ~36 GiB of free HBM.
        """

        # Choose n so that total threads strictly exceeds 2**32:
        # cuda_calc_xblock_count(n, 1024) * 1024 ~= n; need n > 2**32.
        n = (1 << 32) + 1

        # Large-scale invocation — trips the HIP grid cap pre-fix.
        y = torch.ops.fbgemm.zipf_cuda(1.5, n, 0)

        # Shape and dtype.
        self.assertEqual(y.shape, (n,))
        self.assertEqual(y.dtype, torch.int64)

        # O(1) sentinel non-negativity check at start / middle / end.
        # Avoids a full reduction over the 32 GiB tensor (which under
        # opcheck would run 4× and dominate wall time).
        sentinels = y[torch.tensor([0, n // 2, n - 1], device=y.device)]
        self.assertTrue(torch.all(sentinels >= 0).item())

        # Release the 32 GiB tensor before subsequent allocations.
        del y

        # Small-scale invariants — cheap under opcheck.
        n_small = 1024
        y_small = torch.ops.fbgemm.zipf_cuda(1.5, n_small, 0)

        # Non-negativity (full reduction at small scale is free).
        self.assertGreaterEqual(int(y_small.min().item()), 0)

        # Finiteness: cast to float64 because ``isfinite`` on integer
        # dtypes is not defined on all backends (notably ROCm).
        self.assertTrue(torch.isfinite(y_small.to(torch.float64)).all().item())

        # Seed-determinism: same seed must produce identical output.
        y_small2 = torch.ops.fbgemm.zipf_cuda(1.5, n_small, 0)
        torch.testing.assert_close(y_small, y_small2)

        # Uniqueness sanity: the kernel must produce more than one
        # distinct value at this scale.
        unique_count = torch.unique(y_small).numel()
        self.assertGreaterEqual(unique_count, 2)
        self.assertLessEqual(unique_count, n_small)


extend_test_class(ZipfTest)

if __name__ == "__main__":
    unittest.main()
