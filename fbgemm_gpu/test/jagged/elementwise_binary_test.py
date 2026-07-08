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
class ElementwiseBinaryTest(unittest.TestCase):
    def _test_jagged_elementwise_binary(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        y = torch.rand(
            outer_dense_size * np.prod(max_lengths) * inner_dense_size,
            dtype=dtype,
            device=device,
        ).reshape((outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,))

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)
        if operation == "add":
            output_ref = x_padded + y
            output = torch.ops.fbgemm.jagged_dense_elementwise_add(
                x_values, x_offsets, y
            )
        elif operation == "add_jagged_output":
            # create a jagged tensor and then densify
            y = to_padded_dense(
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
            output_ref = x_padded + y
            (
                output,
                output_offsets,
            ) = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                x_values, x_offsets, y
            )
            output = to_padded_dense(output, output_offsets, max_lengths)
        elif operation == "mul":
            output_ref = x_padded * y
            output, output_offsets = torch.ops.fbgemm.jagged_dense_elementwise_mul(
                x_values, x_offsets, y
            )
            output = to_padded_dense(output, output_offsets, max_lengths)
        else:
            raise AssertionError(f"Unknown operation {operation}")

        torch.testing.assert_close(output, output_ref)

        if operation == "add":
            f = torch.ops.fbgemm.jagged_dense_elementwise_add
        elif operation == "add_jagged_output":
            # pyre-fixme[2]: Parameter must be annotated.
            def add_jagged_output_func(*args) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                    *args
                )[0]

            f = add_jagged_output_func
        else:
            assert operation == "mul"

            # pyre-fixme[2]: Parameter must be annotated.
            def mul_func(*args) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_dense_elementwise_mul(*args)[0]

            f = mul_func

        gradcheck(
            f,
            (
                x_values.float().requires_grad_(True),
                x_offsets,
                y.float().requires_grad_(True),
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        operation=st.sampled_from(["add", "add_jagged_output", "mul"]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_elementwise_binary(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._test_jagged_elementwise_binary(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            operation,
            dtype,
            device,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 8),
        inner_dense_size=st.sampled_from([16, 64, 96, 192]),
        operation=st.sampled_from(["add_jagged_output", "mul"]),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_jagged_elementwise_binary_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._test_jagged_elementwise_binary(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            operation,
            dtype,
            device,
        )

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(2, 5),
        inner_dense_size=st.integers(2, 5),
        operation=st.sampled_from(["add", "add_jagged_output", "mul"]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_elementwise_binary_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            mark_dynamic=True,
        )
        y = torch.rand(
            outer_dense_size * np.prod(max_lengths) * inner_dense_size,
            dtype=dtype,
            device=device,
        ).reshape((outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,))

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)

        def jagged_dense_elementwise_add(
            x_values: torch.Tensor, x_offsets: list[torch.LongTensor], y: torch.Tensor
        ) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_dense_elementwise_add(x_values, x_offsets, y)

        def jagged_dense_elementwise_add_jagged_output(
            x_values: torch.Tensor, x_offsets: list[torch.LongTensor], y: torch.Tensor
        ) -> tuple[torch.Tensor, list[torch.LongTensor]]:
            return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                x_values, x_offsets, y
            )

        def jagged_dense_elementwise_mul(
            x_values: torch.Tensor, x_offsets: list[torch.LongTensor], y: torch.Tensor
        ) -> tuple[torch.Tensor, list[torch.LongTensor]]:
            return torch.ops.fbgemm.jagged_dense_elementwise_mul(x_values, x_offsets, y)

        if operation == "add":
            output_ref = x_padded + y
            output = jagged_dense_elementwise_add(x_values, x_offsets, y)

        elif operation == "add_jagged_output":
            # create a jagged tensor and then densify
            y = to_padded_dense(
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
            output_ref = x_padded + y
            (
                output,
                output_offsets,
            ) = jagged_dense_elementwise_add_jagged_output(x_values, x_offsets, y)
            output = to_padded_dense(output, output_offsets, max_lengths)

        elif operation == "mul":
            output_ref = x_padded * y
            output, output_offsets = jagged_dense_elementwise_mul(
                x_values, x_offsets, y
            )
            output = to_padded_dense(output, output_offsets, max_lengths)
        else:
            raise AssertionError(f"Unknown operation {operation}")

        assert output.size() == output_ref.size()

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(32))
    def test_jagged_dense_elementwise_mul_backward_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in
        jagged_jagged_elementwise_dense_output_kernel_ (the backward of
        fbgemm.jagged_dense_elementwise_mul) and verifies output
        correctness via a downsampled CPU oracle.

        block = up to 1024 (dim3(threads_x<=32, threads_y=32) from
            check_shape_and_partition_ at common.cuh:187)
        grid.x = div_round_up(outer * jagged_folded, 32)
        Total threads ~= outer * jagged_folded * 32.
        For pre-fix to trip and post-fix to pass:
            pre-fix:  outer * jagged_folded > 2**27 ⇒ trips.
            post-fix: 16384 * 32 * 32 = 2**24 < 2**32 (always passes).

        Verification strategy (downsampled-oracle):
        1. Full-scale forward + backward to verify the backward kernel
           launches successfully under the cap. Uses zero values so per-
           element work is zero; backward grad shape is asserted.
        2. Small-scale forward + backward vs CPU dispatch to validate
           kernel correctness end-to-end.
        """
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # ---- Step 1: full-scale forward + backward launch survival. ----
        B = 1
        max_L = (1 << 27) + 1
        # D=8 keeps x_values.numel() = max_L*D < 2**31 (the kernel's
        # int32_t accessor limit) while still leaving outer*jagged_folded
        # = max_L > 2**27, which trips the cap pre-fix.
        D = 8
        nnz = max_L  # one segment of length max_L
        x_values_large = torch.zeros(
            (nnz, D), dtype=torch.float32, device=device, requires_grad=True
        )
        x_offsets_large = [torch.tensor([0, nnz], dtype=torch.int64, device=device)]
        y_large = torch.zeros(
            (B, max_L, D),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        output_large = torch.ops.fbgemm.jagged_dense_elementwise_mul(
            x_values_large, x_offsets_large, y_large
        )
        # output is a tuple (output_jagged_values, output_offsets). Use the
        # jagged values for backward.
        if isinstance(output_large, (tuple, list)):
            loss_large = output_large[0].sum()
        else:
            loss_large = output_large.sum()
        # Pre-fix this trips KernelLauncher::checkThreadCountNotExceeded on
        # ROCm.
        loss_large.backward()
        # pyre-ignore[16]
        self.assertEqual(x_values_large.grad.shape, x_values_large.shape)
        # pyre-ignore[16]
        self.assertEqual(y_large.grad.shape, y_large.shape)
        del x_values_large, x_offsets_large, y_large, output_large, loss_large
        torch.cuda.empty_cache()

        # ---- Step 2: downsampled CPU-oracle correctness check. ----
        # Same kernel code path, smaller scale to keep the CPU oracle cheap.
        # Distinct non-zero values so any "kernel addressed wrong row" bug
        # surfaces in both the forward output and the backward grads.
        small_B = 2
        small_max_L = 4
        small_D = 3
        # Sparse jagged: segment 0 has 1 row, segment 1 has 3 rows.
        small_lengths = torch.tensor([1, 3], dtype=torch.int64)
        small_offsets_cpu = [torch.zeros(small_B + 1, dtype=torch.int64)]
        small_offsets_cpu[0][1:] = torch.cumsum(small_lengths, dim=0)
        small_nnz = int(small_offsets_cpu[0][-1].item())
        x_values_init = torch.arange(small_nnz * small_D, dtype=torch.float32).reshape(
            small_nnz, small_D
        )
        y_init = (
            torch.arange(small_B * small_max_L * small_D, dtype=torch.float32).reshape(
                small_B, small_max_L, small_D
            )
            * 0.1
        )

        # CPU forward + backward.
        x_values_cpu = x_values_init.detach().clone().requires_grad_(True)
        y_cpu = y_init.detach().clone().requires_grad_(True)
        output_cpu = torch.ops.fbgemm.jagged_dense_elementwise_mul(
            x_values_cpu, small_offsets_cpu, y_cpu
        )
        if isinstance(output_cpu, (tuple, list)):
            loss_cpu = output_cpu[0].sum()
        else:
            loss_cpu = output_cpu.sum()
        loss_cpu.backward()

        # GPU forward + backward.
        x_values_gpu = x_values_init.detach().clone().to(device).requires_grad_(True)
        y_gpu = y_init.detach().clone().to(device).requires_grad_(True)
        small_offsets_gpu = [t.to(device) for t in small_offsets_cpu]
        output_gpu = torch.ops.fbgemm.jagged_dense_elementwise_mul(
            x_values_gpu, small_offsets_gpu, y_gpu
        )
        if isinstance(output_gpu, (tuple, list)):
            loss_gpu = output_gpu[0].sum()
        else:
            loss_gpu = output_gpu.sum()
        loss_gpu.backward()

        # Forward jagged values must match.
        if isinstance(output_cpu, (tuple, list)) and isinstance(
            output_gpu, (tuple, list)
        ):
            torch.testing.assert_close(output_gpu[0].cpu(), output_cpu[0])
        else:
            torch.testing.assert_close(output_gpu.cpu(), output_cpu)
        # pyre-ignore[16]
        torch.testing.assert_close(x_values_gpu.grad.cpu(), x_values_cpu.grad)
        # pyre-ignore[16]
        torch.testing.assert_close(y_gpu.grad.cpu(), y_cpu.grad)


if __name__ == "__main__":
    unittest.main()
