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
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
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

        # Citrine C3: create random input directly on the target device.
        # pyre-ignore-errors[16]
        x = torch.randint(low=0, high=100, size=(n,), device=device).type(
            pt_index_dtype
        )
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
        # Citrine C3: create random input directly on the meta device.
        # pyre-ignore-errors[16]
        mx = torch.randint(low=0, high=100, size=(n,), device="meta").type(
            pt_index_dtype
        )

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
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
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

        # Citrine C3: create random input directly on the target device.
        # pyre-ignore-errors[16]
        x = torch.randint(low=0, high=100, size=(b, n), device=device).type(
            pt_index_dtype
        )

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
        verbosity=Verbosity.verbose,
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
        values = torch.randint(
            0, 1000, (batch_size, max_len), device=device, dtype=dtype
        )
        out = torch.ops.fbgemm.asynchronous_batched_complete_cumsum(values)
        out2 = torch.cat(
            [
                torch.zeros_like(values[:, :1]),
                torch.cumsum(values, dim=1, dtype=values.dtype),
            ],
            dim=1,
        )
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

    @staticmethod
    def _segmented_complete_cumsum_ref(
        x: torch.Tensor, segment_sizes: list[int]
    ) -> torch.Tensor:
        return torch.cat(
            [x.new_zeros(0)]
            + [
                torch.ops.fbgemm.asynchronous_complete_cumsum(segment)
                for segment in x.contiguous().split(segment_sizes)
            ]
        )

    @given(
        segment_sizes=st.lists(st.integers(0, 64), max_size=20),
        dtype=st.sampled_from([torch.int64, torch.int32, torch.float32]),
        strided=st.booleans(),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_segmented_complete_cumsum(
        self,
        segment_sizes: list[int],
        dtype: torch.dtype,
        strided: bool,
        device: torch.device,
    ) -> None:
        # Like asynchronous_complete_cumsum, only the CPU variant supports floats.
        if device == torch.accelerator.current_accelerator() and dtype is torch.float32:
            return

        # Citrine C3: create random input directly on the target device.
        x = torch.randint(0, 100, (2 * sum(segment_sizes),), device=device).to(dtype)
        x = x[::2] if strided else x[: sum(segment_sizes)]
        out = torch.ops.fbgemm.asynchronous_segmented_complete_cumsum(x, segment_sizes)
        torch.testing.assert_close(
            out, self._segmented_complete_cumsum_ref(x, segment_sizes)
        )

        meta_out = torch.ops.fbgemm.asynchronous_segmented_complete_cumsum(
            x.to("meta"), segment_sizes
        )
        self.assertEqual(meta_out.size(), out.size())

    def test_segmented_complete_cumsum_cpu_and_meta(self) -> None:
        for segment_sizes in ([3, 0, 5], []):
            for dtype in (torch.int32, torch.int64, torch.float32):
                x = torch.arange(sum(segment_sizes), dtype=dtype)
                out = torch.ops.fbgemm.asynchronous_segmented_complete_cumsum(
                    x, segment_sizes
                )
                torch.testing.assert_close(
                    out, self._segmented_complete_cumsum_ref(x, segment_sizes)
                )
                meta_out = torch.ops.fbgemm.asynchronous_segmented_complete_cumsum(
                    x.to("meta"), segment_sizes
                )
                self.assertEqual(meta_out.size(), out.size())

    @optests.dontGenerateOpCheckTests("expects the op to raise")
    def test_segmented_complete_cumsum_invalid_inputs(self) -> None:
        devices: list[str | torch.device] = ["cpu", "meta"]
        if torch.cuda.is_available():
            devices.append(torch.accelerator.current_accelerator())
        for device in devices:
            x = torch.arange(6, device=device)
            for t_in, segment_sizes in (
                (x, [2, 3]),
                (x, [7, -1]),
                (x.view(2, 3), [6]),
                # Sums to 2**64, which would wrap to the length of t_in.
                (x[:0], [2**63 - 1, 2**63 - 1, 2]),
            ):
                with self.assertRaises(RuntimeError):
                    torch.ops.fbgemm.asynchronous_segmented_complete_cumsum(
                        t_in, segment_sizes
                    )

    @unittest.skipIf(*gpu_unavailable)
    @optests.dontGenerateOpCheckTests("large GPU inputs")
    def test_segmented_complete_cumsum_long_and_many_segments(self) -> None:
        device = torch.accelerator.current_accelerator()
        # Long and empty segments, many long segments, more segments than one
        # launch takes, and a single segment.
        for segment_sizes in (
            [4095, 4096, 4097, 0, 100_000, 1],
            [70_000] * 50,
            [i % 97 for i in range(1000)],
            [300_000],
        ):
            for dtype in (torch.int32, torch.int64):
                # Large int32 values wrap, which must match the per-segment op.
                x = torch.randint(
                    0, 2**30, (sum(segment_sizes),), device=device, dtype=dtype
                )
                torch.testing.assert_close(
                    torch.ops.fbgemm.asynchronous_segmented_complete_cumsum(
                        x, segment_sizes
                    ),
                    self._segmented_complete_cumsum_ref(x, segment_sizes),
                )

    @unittest.skipIf(*gpu_unavailable)
    @optests.dontGenerateOpCheckTests("CUDA graph capture and replay")
    def test_segmented_complete_cumsum_cuda_graph(self) -> None:
        device = torch.accelerator.current_accelerator()
        segment_sizes = [3, 0, 5000, 7]
        x = torch.randint(0, 100, (sum(segment_sizes),), device=device)

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            torch.ops.fbgemm.asynchronous_segmented_complete_cumsum(x, segment_sizes)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = torch.ops.fbgemm.asynchronous_segmented_complete_cumsum(
                x, segment_sizes
            )

        x.copy_(torch.randint(0, 100, x.shape, device=device))
        graph.replay()
        torch.testing.assert_close(
            out, self._segmented_complete_cumsum_ref(x, segment_sizes)
        )


extend_test_class(CumSumTest)

if __name__ == "__main__":
    unittest.main()
