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
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import (
    additional_decorators,
    generate_jagged_tensor,
    open_source,
    to_padded_dense,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        cpu_and_maybe_gpu,
        gpu_memory_lt_gb,
        gpu_unavailable,
        gradcheck,
        optests,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        cpu_and_maybe_gpu,
        gpu_memory_lt_gb,
        gpu_unavailable,
        gradcheck,
        optests,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class DenseDenseElementwiseAddTest(unittest.TestCase):
    def _test_jagged_dense_dense_elementwise_add_jagged_output(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        (
            output,
            output_offsets,
        ) = torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
            x_values, x_offsets, y_0, y_1
        )
        output = to_padded_dense(output, output_offsets, max_lengths)

        torch.testing.assert_close(output, output_ref)

        # pyre-fixme[2]: Parameter must be annotated.
        def add_jagged_output_func(*args) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                *args
            )[0]

        f = add_jagged_output_func

        gradcheck(
            f,
            (
                x_values.float().requires_grad_(True),
                x_offsets,
                y_0.float().requires_grad_(True),
                y_1.float().requires_grad_(True),
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.normal, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._test_jagged_dense_dense_elementwise_add_jagged_output(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 8),
        inner_dense_size=st.sampled_from([16, 64, 96, 192]),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.normal, max_examples=4, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._test_jagged_dense_dense_elementwise_add_jagged_output(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("meta"),
    )
    @settings(verbosity=Verbosity.normal, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        device = torch.device("cpu")

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        x_values.to(device_type)
        (
            output,
            output_offsets,
        ) = torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
            x_values, x_offsets, y_0, y_1
        )
        output.to("cpu")
        output = to_padded_dense(output, output_offsets, max_lengths)

        assert output.size() == output_ref.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(2, 4),
        inner_dense_size=st.integers(2, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("cpu"),
    )
    @settings(verbosity=Verbosity.normal, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            torch.device(device_type),
            mark_dynamic=True,
        )

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=torch.device(device_type),
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=torch.device(device_type),
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        x_values.to(device_type)
        output, output_offsets = torch.compile(
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output,
            fullgraph=True,
            dynamic=True,
        )(x_values, x_offsets, y_0, y_1)
        output.to("cpu")
        output = to_padded_dense(output, output_offsets, max_lengths)

        assert output.size() == output_ref.size()

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(40))
    def test_jagged_dense_dense_elementwise_add_jagged_output_large_grid(
        self,
    ) -> None:
        """
        Reproduces the HIP grid-overflow bug in
        jagged_dense_dense_elementwise_jagged_output_kernel_ (the
        common.cuh `_opt_` codepath, kernel template at common.cuh:738)
        and verifies output correctness via a downsampled CPU oracle.

        Same block/grid arithmetic as the mul_backward kernel: block <= 1024;
        grid.x = div_round_up(outer * jagged_folded, 32); saturation at
        outer * jagged_folded >= 2**27. With D=8 we keep
        x_values.numel() < 2**31 (the kernel's int32_t accessor limit).

        Verification strategy (downsampled-oracle):
        1. Full-scale forward to verify launch survival at the cap-trip
           scale. Uses zeros so output is zeros; shape and zero-fill are
           asserted.
        2. Small-scale invocation vs CPU dispatch with sentinel non-zero
           values to validate kernel correctness end-to-end.
        """
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # ---- Step 1: full-scale launch survival (cap-trip detection). ----
        B = 1
        max_L = (1 << 27) + 1
        # D=8 keeps numel < 2**31 for int32_t accessor compatibility.
        D = 8
        nnz = max_L
        x_values_large = torch.zeros((nnz, D), dtype=torch.float32, device=device)
        x_offsets_large = [torch.tensor([0, nnz], dtype=torch.int64, device=device)]
        y0_large = torch.zeros((B, max_L, D), dtype=torch.float32, device=device)
        y1_large = torch.zeros((B, max_L, D), dtype=torch.float32, device=device)

        output_large = (
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                x_values_large, x_offsets_large, y0_large, y1_large
            )
        )
        # output[0] is the output jagged values.
        self.assertEqual(output_large[0].shape, (nnz, D))
        self.assertTrue(torch.all(output_large[0] == 0).item())
        del x_values_large, x_offsets_large, y0_large, y1_large, output_large
        torch.cuda.empty_cache()

        # ---- Step 2: downsampled CPU-oracle correctness check. ----
        # Same kernel code path, smaller scale. Distinct non-zero values so
        # any "kernel addressed wrong row" bug surfaces in the output sum.
        small_B = 2
        small_max_L = 4
        small_D = 3
        small_lengths = torch.tensor([1, 3], dtype=torch.int64)
        small_x_offsets_cpu = [torch.zeros(small_B + 1, dtype=torch.int64)]
        small_x_offsets_cpu[0][1:] = torch.cumsum(small_lengths, dim=0)
        small_nnz = int(small_x_offsets_cpu[0][-1].item())
        small_x_values_cpu = torch.arange(
            small_nnz * small_D, dtype=torch.float32
        ).reshape(small_nnz, small_D)
        small_y0_cpu = (
            torch.arange(small_B * small_max_L * small_D, dtype=torch.float32).reshape(
                small_B, small_max_L, small_D
            )
            * 0.1
        )
        small_y1_cpu = (
            torch.arange(small_B * small_max_L * small_D, dtype=torch.float32).reshape(
                small_B, small_max_L, small_D
            )
            * 0.01
        )

        # CPU reference oracle.
        small_output_cpu = (
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                small_x_values_cpu,
                small_x_offsets_cpu,
                small_y0_cpu,
                small_y1_cpu,
            )
        )

        # GPU op under test.
        small_x_offsets_gpu = [t.to(device) for t in small_x_offsets_cpu]
        small_output_gpu = (
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                small_x_values_cpu.to(device),
                small_x_offsets_gpu,
                small_y0_cpu.to(device),
                small_y1_cpu.to(device),
            )
        )

        torch.testing.assert_close(small_output_gpu[0].cpu(), small_output_cpu[0])

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(40))
    def test_jagged_dense_dense_elementwise_add_jagged_output_opt_search_large_grid(
        self,
    ) -> None:
        """
        Retro: regression test for the HIP grid-overflow bug in
        ``jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_``
        (D105203391 / Subplan D Diff #25), which lacked its own test
        method when landed.

        The opt-search path is taken by
        ``jagged_dense_dense_elementwise_add_jagged_output`` when
        ``D = Half`` and ``B == 1``. With kMaxThreads=1024, the
        binary-search kernel launches with grid_x = ceil(nnz / 1024).
        For nnz > 2**22 the launch exceeds 2^32 threads pre-fix; the
        production fix caps grid_x to ``get_max_thread_blocks(stream)``
        and the kernel grid-strides over rows.

        Verification strategy:
        1. Full-scale launch survival to verify the cap-trip path.
        2. Small-scale CPU-oracle correctness to verify kernel logic
           through the opt-search dispatch.
        """
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # ---- Step 1: full-scale launch survival. ----
        B = 1  # opt-search path requires B == 1.
        max_L = (1 << 22) + 1  # nnz > 2**22 trips the cap pre-fix.
        D_half = 16  # half-dtype required; numel < 2**31 satisfied.
        nnz = max_L
        x_values_large = torch.zeros((nnz, D_half), dtype=torch.float16, device=device)
        x_offsets_large = [torch.tensor([0, nnz], dtype=torch.int64, device=device)]
        y0_large = torch.zeros((B, max_L, D_half), dtype=torch.float16, device=device)
        y1_large = torch.zeros((B, max_L, D_half), dtype=torch.float16, device=device)

        output_large = (
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                x_values_large, x_offsets_large, y0_large, y1_large
            )
        )
        self.assertEqual(output_large[0].shape, (nnz, D_half))
        del x_values_large, x_offsets_large, y0_large, y1_large, output_large
        torch.cuda.empty_cache()

        # ---- Step 2: small-scale CPU-oracle correctness. ----
        small_B = 1
        small_max_L = 4
        small_D = 8
        small_lengths = torch.tensor([3], dtype=torch.int64)
        small_x_offsets_cpu = [torch.zeros(small_B + 1, dtype=torch.int64)]
        small_x_offsets_cpu[0][1:] = torch.cumsum(small_lengths, dim=0)
        small_nnz = int(small_x_offsets_cpu[0][-1].item())
        small_x_values_cpu = (
            torch.arange(small_nnz * small_D, dtype=torch.float16).reshape(
                small_nnz, small_D
            )
            * 0.1
        )
        small_y0_cpu = (
            torch.arange(small_B * small_max_L * small_D, dtype=torch.float16).reshape(
                small_B, small_max_L, small_D
            )
            * 0.01
        )
        small_y1_cpu = (
            torch.arange(small_B * small_max_L * small_D, dtype=torch.float16).reshape(
                small_B, small_max_L, small_D
            )
            * 0.001
        )

        small_output_cpu = (
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                small_x_values_cpu,
                small_x_offsets_cpu,
                small_y0_cpu,
                small_y1_cpu,
            )
        )
        small_x_offsets_gpu = [t.to(device) for t in small_x_offsets_cpu]
        small_output_gpu = (
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                small_x_values_cpu.to(device),
                small_x_offsets_gpu,
                small_y0_cpu.to(device),
                small_y1_cpu.to(device),
            )
        )
        torch.testing.assert_close(small_output_gpu[0].cpu(), small_output_cpu[0])


if __name__ == "__main__":
    unittest.main()
