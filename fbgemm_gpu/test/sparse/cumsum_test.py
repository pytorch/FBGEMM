#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu, gpu_memory_lt_gb, gpu_unavailable, optests
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import (
        cpu_and_maybe_gpu,
        gpu_memory_lt_gb,
        gpu_unavailable,
        optests,
    )


class CumSumTest(unittest.TestCase):
    @given(
        n=st.integers(min_value=0, max_value=10),
        index_types=st.sampled_from(
            [
                (torch.int64, np.int64),
                (torch.int32, np.int32),
                (torch.float32, np.float32),
            ]
        ),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.normal, max_examples=20, deadline=None)
    def test_cumsum(
        self,
        n: int,
        index_types: tuple[type[object], type[object]],
        device: torch.device,
    ) -> None:
        pt_index_dtype, np_index_dtype = index_types

        # The CPU variants of asynchronous_*_cumsum support floats, since some
        # downstream tests appear to be relying on this behavior.  As such, the
        # test is disabled for GPU + float test cases.
        if (
            device == torch.accelerator.current_accelerator()
            and pt_index_dtype is torch.float32
        ):
            return

        # pyre-ignore-errors[16]
        x = torch.randint(low=0, high=100, size=(n,)).type(pt_index_dtype).to(device)
        ze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
        zi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
        zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)

        torch.testing.assert_close(
            torch.from_numpy(np.cumsum(x.cpu().numpy()).astype(np_index_dtype)),
            zi.cpu(),
        )
        torch.testing.assert_close(
            torch.from_numpy(
                (np.cumsum([0] + x.cpu().numpy().tolist())[:-1]).astype(np_index_dtype)
            ),
            ze.cpu(),
        )
        torch.testing.assert_close(
            torch.from_numpy(
                (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype)
            ),
            zc.cpu(),
        )

        # meta tests
        # pyre-ignore-errors[16]
        mx = torch.randint(low=0, high=100, size=(n,)).type(pt_index_dtype).to("meta")

        mze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(mx)
        self.assertEqual(ze.size(), mze.size())

        mzi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(mx)
        self.assertEqual(zi.size(), mzi.size())

        mzc = torch.ops.fbgemm.asynchronous_complete_cumsum(mx)
        self.assertEqual(zc.size(), mzc.size())

    @given(
        n=st.integers(min_value=0, max_value=60),
        b=st.integers(min_value=0, max_value=10),
        index_types=st.sampled_from(
            [
                (torch.int64, np.int64),
                (torch.int32, np.int32),
                (torch.float32, np.float32),
            ]
        ),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.normal, max_examples=20, deadline=None)
    def test_asynchronous_complete_cumsum_2d(
        self,
        n: int,
        b: int,
        index_types: tuple[type[object], type[object]],
        device: torch.device,
    ) -> None:
        pt_index_dtype, np_index_dtype = index_types

        # The CPU variants of asynchronous_*_cumsum support floats, since some
        # downstream tests appear to be relying on this behavior.  As such, the
        # test is disabled for GPU + float test cases.
        if (
            device == torch.accelerator.current_accelerator()
            and pt_index_dtype is torch.float32
        ):
            return

        # pyre-ignore-errors[16]
        x = torch.randint(low=0, high=100, size=(b, n)).type(pt_index_dtype).to(device)

        zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        zeros = torch.zeros(b, 1)
        torch.testing.assert_close(
            torch.from_numpy(
                np.cumsum(torch.concat([zeros, x.cpu()], dim=1).numpy(), axis=1).astype(
                    np_index_dtype
                )
            ),
            zc.cpu(),
        )

    @given(
        batch_size=st.integers(10, 1000),
        max_len=st.integers(10, 1000),
        dtype=st.sampled_from([torch.int32, torch.int64]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(
        verbosity=Verbosity.normal,
        max_examples=50,
        deadline=None,
    )
    def test_batched_complete_cumsum(
        self,
        batch_size: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        def cumsum_base(values: torch.Tensor) -> torch.Tensor:
            out = [
                torch.ops.fbgemm.asynchronous_complete_cumsum(values[i])
                for i in range(values.shape[0])
            ]
            return torch.stack(out, dim=0)

        values = torch.randint(
            0, 1000, (batch_size, max_len), device=device, dtype=dtype
        )
        out = torch.ops.fbgemm.asynchronous_batched_complete_cumsum(values)
        out2 = cumsum_base(values)
        torch.testing.assert_close(out, out2)

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(4))
    @optests.dontGenerateOpCheckTests(
        "large-grid GPU-memory-gated stress repro; opcheck variants only skip on "
        "CPU samples and add no op coverage (T191384137)"
    )
    def test_asynchronous_batched_complete_cumsum_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in _batched_complete_cumsum_kernel
        and verifies output correctness against the CPU dispatch at the same
        scale.

        With len=1, nthreads_per_block = max(next_power_of_2(1), 64) = 64.
        Total threads = B * 64. For B > 2**26, total threads exceed the HIP
        2**32 limit, causing FBGEMM_LAUNCH_KERNEL ->
        KernelLauncher::checkThreadCountNotExceeded to TORCH_CHECK-fail on
        ROCm pre-fix. With the production fix, the kernel grid-strides
        over batches so the capped grid still covers all B rows.

        ``values`` is sparse: zero everywhere except sentinel non-zero
        entries at start / middle / end of the batch axis. Any "kernel
        addressed wrong row" bug surfaces in the assertion below.
        """

        # B * 64 > 2**32 requires B > 2**26.
        B = (1 << 26) + 1

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # Sparse non-zero values at sentinel positions.
        values_cpu = torch.zeros((B, 1), dtype=torch.int32)
        values_cpu[0, 0] = 1
        values_cpu[B // 2, 0] = 2
        values_cpu[B - 1, 0] = 3

        # CPU reference oracle — same op, different dispatch.
        cumsum_cpu = torch.ops.fbgemm.asynchronous_batched_complete_cumsum(values_cpu)

        # GPU op under test. Pre-fix, this launch trips
        # KernelLauncher::checkThreadCountNotExceeded on ROCm.
        cumsum_gpu = torch.ops.fbgemm.asynchronous_batched_complete_cumsum(
            values_cpu.to(device)
        )

        torch.testing.assert_close(cumsum_gpu.cpu(), cumsum_cpu)


extend_test_class(CumSumTest)

if __name__ == "__main__":
    unittest.main()
