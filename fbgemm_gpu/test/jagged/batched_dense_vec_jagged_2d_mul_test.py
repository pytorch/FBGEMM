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
import torch
import torch._dynamo
from hypothesis import assume, given, settings, Verbosity

from .common import additional_decorators, open_source

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
class BatchedDenseVecJagged2DMulTest(unittest.TestCase):
    @settings(
        verbosity=Verbosity.normal,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(0, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(0, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    def test_batched_dense_vec_jagged_2d_mul(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assume(H == 1 or B != 0)
        # CPU doesn't support bfloat16
        assume(device != torch.device("cpu") or dtype != torch.bfloat16)

        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16] and device == torch.device("cpu"):
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        if dtype in [torch.half, torch.bfloat16] and device == torch.device("cpu"):
            output_ref = output_ref.to(dtype)
        output = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            dense, values, offsets
        )
        torch.testing.assert_close(
            output,
            output_ref,
            rtol=1e-2 if dtype in [torch.half, torch.bfloat16] else None,
            atol=1e-2 if dtype in [torch.half, torch.bfloat16] else None,
        )

        gradcheck(
            torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
            (
                dense.clone().detach().float().requires_grad_(True),
                values.clone().detach().float().requires_grad_(True),
                offsets,
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @settings(
        verbosity=Verbosity.normal,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(0, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(0, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.sampled_from(["meta"]),
    )
    def test_batched_dense_vec_jagged_2d_mul_meta_backend(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        assume(H == 1 or B != 0)

        device = torch.device("cpu")
        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16]:
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        dense.to(device_type)
        values.to(device_type)
        output = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            dense, values, offsets
        )
        assert output.size() == output_ref.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @settings(
        verbosity=Verbosity.normal,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(2, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(2, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("cpu"),
    )
    def test_batched_dense_vec_jagged_2d_mul_dynamic_shape(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        assume(H == 1 or B != 0)

        device = torch.device(device_type)
        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(low=1, high=max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16]:
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        dense.to(device_type)
        values.to(device_type)

        torch._dynamo.mark_dynamic(dense, 0)
        torch._dynamo.mark_dynamic(values, 0)
        torch._dynamo.mark_dynamic(values, 1)
        torch._dynamo.mark_dynamic(offsets, 0)

        output = torch.compile(
            torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
            fullgraph=True,
            dynamic=True,
        )(dense, values, offsets)
        assert output.size() == output_ref.size()

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(8))
    def test_batched_dense_vec_jagged_2d_mul_forward_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in dense_vec_jagged_2d_bmm
        and verifies output correctness via a downsampled CPU oracle.

        Block: dim3(block_dim_x, block_dim_y) where
            block_dim_x = min(div_round_up(D, kWarpSize)*kWarpSize, kMaxThreads),
            block_dim_y = kMaxThreads / block_dim_x.
        Block size in threads = kMaxThreads = 1024.
        Grid: dim3(div_round_up(B*H, block_dim_y)).
        Total threads = ceil(B*H / block_dim_y) * 1024.

        The production cap is `blocks_x_capped = min(blocks_x_uncapped,
        get_max_thread_blocks(stream))` where `get_max_thread_blocks ~=
        16384` on MI300/MI350. For pre-fix to trip and post-fix to pass:
            pre-fix:  ceil(B*H / block_dim_y) * 1024 > 2**32
                      ⇒ B*H > 2**22 * block_dim_y. With D = 4,
                        block_dim_y = 32, so B*H > 2**27.
            post-fix: 16384 * 1024 = 2**24 < 2**32 (always passes
                      regardless of B*H).

        Verification strategy (per master plan's downsampled-oracle
        guidance for ops where the full-scale CPU oracle is impractical
        due to ~2 GiB CPU memory pressure):

        1. Full-scale invocation to verify launch survival at the
           cap-trip scale. Uses an empty jagged tensor (max_L=0) so
           the kernel does no per-segment work; output shape and
           zero-fill are asserted.
        2. Small-scale invocation vs CPU dispatch with sentinel
           non-zero values to validate kernel correctness end-to-end.
        """

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # ---- Step 1: full-scale launch survival (cap-trip detection). ----
        B = (1 << 27) + 1
        H = 1
        D = 4
        # Empty jagged tensor (max_L=0 ⇒ a_values has 0 rows).
        a_values_large = torch.zeros((0, H * D), dtype=torch.float32, device=device)
        a_offsets_large = torch.zeros(B * H + 1, dtype=torch.int64, device=device)
        v_large = torch.zeros((B * H, 0), dtype=torch.float32, device=device)

        output_large = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            v_large, a_values_large, a_offsets_large
        )
        self.assertEqual(output_large.shape, (B * H, D))
        self.assertTrue(torch.all(output_large == 0).item())
        del a_values_large, a_offsets_large, v_large, output_large

        # ---- Step 2: downsampled CPU-oracle correctness check. ----
        # Same kernel code path, smaller scale to keep the CPU oracle cheap.
        # Distinct non-zero values so any "kernel addressed wrong row" bug
        # surfaces in the matrix product output.
        small_B = 2
        small_H = 2
        small_D = 4
        small_max_L = 3
        # Sparse jagged: segment 0 has 1 row, segment 1 has 2 rows,
        # segment 2 has 0 rows, segment 3 has 3 rows. Total = 6 rows.
        small_lengths = torch.tensor([1, 2, 0, 3], dtype=torch.int64)
        small_offsets_cpu = torch.zeros(small_B * small_H + 1, dtype=torch.int64)
        small_offsets_cpu[1:] = torch.cumsum(small_lengths, dim=0)
        total_rows = int(small_offsets_cpu[-1].item())
        small_a_values_cpu = torch.arange(
            total_rows * small_H * small_D, dtype=torch.float32
        ).reshape(total_rows, small_H * small_D)
        small_v_cpu = (
            torch.arange(small_B * small_H * small_max_L, dtype=torch.float32).reshape(
                small_B * small_H, small_max_L
            )
            * 0.1
        )

        # CPU reference oracle.
        small_output_cpu = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            small_v_cpu, small_a_values_cpu, small_offsets_cpu
        )

        # GPU op under test.
        small_output_gpu = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            small_v_cpu.to(device),
            small_a_values_cpu.to(device),
            small_offsets_cpu.to(device),
        )

        torch.testing.assert_close(small_output_gpu.cpu(), small_output_cpu)

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(8))
    def test_batched_dense_vec_jagged_2d_mul_backward_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in the backward kernels of
        fbgemm.batched_dense_vec_jagged_2d_mul:
          - dense_vec_jagged_2d_transposed_bmm: grid =
              ceil(B*H / block_dim_y) * 1024.
          - outer_prod_jagged_2d_output: grid =
              ceil(B*H*max_L / block_dim_y) * 1024.

        The production cap is `blocks_x_capped = min(blocks_x_uncapped,
        get_max_thread_blocks(stream))` where `get_max_thread_blocks ~=
        16384` on MI300/MI350. With max_L=1, D=4, block_dim_y=32 for
        both kernels:
            pre-fix:  ceil(B*H / 32) * 1024 > 2**32 ⇒ B*H > 2**27.
            post-fix: 16384 * 1024 = 2**24 < 2**32 (always passes).

        Verification strategy (downsampled-oracle):
        1. Full-scale forward + backward to verify both backward kernels
           launch successfully under the cap. Uses an empty jagged
           tensor (offsets all zero) so per-segment work is zero;
           backward grad shapes are asserted.
        2. Small-scale forward + backward vs CPU dispatch to validate
           kernel correctness end-to-end.
        """

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # ---- Step 1: full-scale forward + backward launch survival. ----
        B = (1 << 27) + 1
        H = 1
        max_L = 1
        D = 4
        a_values_large = torch.zeros(
            (0, H * D), dtype=torch.float32, device=device, requires_grad=True
        )
        a_offsets_large = torch.zeros(B * H + 1, dtype=torch.int64, device=device)
        v_large = torch.zeros(
            (B * H, max_L),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        output_large = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            v_large, a_values_large, a_offsets_large
        )
        # Pre-fix, this backward call's two kernels both trip
        # KernelLauncher::checkThreadCountNotExceeded on ROCm.
        output_large.sum().backward()
        # pyre-ignore[16]
        self.assertEqual(v_large.grad.shape, v_large.shape)
        # pyre-ignore[16]
        self.assertEqual(a_values_large.grad.shape, a_values_large.shape)
        del v_large, a_values_large, a_offsets_large, output_large
        torch.cuda.empty_cache()

        # ---- Step 2: downsampled CPU-oracle correctness check. ----
        small_B = 2
        small_H = 2
        small_D = 4
        small_max_L = 3
        # Sparse jagged offsets. Total = 6 rows.
        small_lengths = torch.tensor([1, 2, 0, 3], dtype=torch.int64)
        small_offsets_cpu = torch.zeros(small_B * small_H + 1, dtype=torch.int64)
        small_offsets_cpu[1:] = torch.cumsum(small_lengths, dim=0)
        total_rows = int(small_offsets_cpu[-1].item())
        a_values_init = torch.arange(
            total_rows * small_H * small_D, dtype=torch.float32
        ).reshape(total_rows, small_H * small_D)
        v_init = (
            torch.arange(small_B * small_H * small_max_L, dtype=torch.float32).reshape(
                small_B * small_H, small_max_L
            )
            * 0.1
        )

        # CPU forward + backward.
        a_values_cpu = a_values_init.detach().clone().requires_grad_(True)
        v_cpu = v_init.detach().clone().requires_grad_(True)
        output_cpu = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            v_cpu, a_values_cpu, small_offsets_cpu
        )
        output_cpu.sum().backward()

        # GPU forward + backward.
        a_values_gpu = a_values_init.detach().clone().to(device).requires_grad_(True)
        v_gpu = v_init.detach().clone().to(device).requires_grad_(True)
        output_gpu = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            v_gpu, a_values_gpu, small_offsets_cpu.to(device)
        )
        output_gpu.sum().backward()

        torch.testing.assert_close(output_gpu.cpu(), output_cpu)
        # pyre-ignore[16]
        torch.testing.assert_close(v_gpu.grad.cpu(), v_cpu.grad)
        # pyre-ignore[16]
        torch.testing.assert_close(a_values_gpu.grad.cpu(), a_values_cpu.grad)


if __name__ == "__main__":
    unittest.main()
