#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import unittest
from typing import Any

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import given, settings, Verbosity

from ..common import gpu_unavailable

common_st = {
    "D": st.integers(min_value=1, max_value=512),
}

common_settings = {
    "verbosity": Verbosity.normal,
    "max_examples": 4,
    "deadline": None,
}

MAX_INT32 = 2147483647


class ForwardBackwardInt32OverflowTest(unittest.TestCase):
    def _execute_forward_backward_large_emb(
        self,
        weights_precision: SparseType,
        indices_dtype: torch.dtype,
        D: int = 1,
    ) -> None:
        """
        Execute the forward and backward tests for a large embedding table
        (numel >= MAX_INT32)

        The test will fail if a runtime error, such as illegal memory access,
        is caught
        """
        weight_dtype_bytes = weights_precision.bit_rate() // 8

        # Embedding dimension
        D = D * 4
        row_bytes = D * weight_dtype_bytes
        # Hash size
        # Compute the number of rows in the embedding table by
        # div_up(MAX_INT32, D) and add 32 extra bytes to ensure that IMA
        E = math.ceil(MAX_INT32 / D) + math.ceil(32 / row_bytes)

        assert E * D >= MAX_INT32

        # Compute total weight bytes
        weight_bytes = E * D * weight_dtype_bytes
        assert weight_bytes > 0

        # Compute free memory
        total_memory = torch.cuda.get_device_properties().total_memory
        reserved_memory = torch.cuda.memory_reserved()
        free_memory = total_memory - reserved_memory
        if free_memory < weight_bytes:
            self.skipTest(
                f"Skip test_forward_backward_large_emb: Free memory "
                f"({free_memory}) < weight_bytes ({weight_bytes})"
            )

        # Get device
        device = torch.cuda.current_device()

        # Instantiate a TBE op
        op = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[(E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)],
            output_dtype=SparseType.FP32,
            device=device,
        )

        # Generate inputs
        indices = torch.as_tensor([E - 1], dtype=indices_dtype, device=device)
        offsets = torch.as_tensor([0, 1], dtype=indices_dtype, device=device)
        per_sample_weights = torch.as_tensor([0.9], dtype=torch.float, device=device)

        # Test both weighted and unweighted
        for weighted in [False, True]:
            try:
                # Run forward
                out = op(
                    indices=indices,
                    offsets=offsets,
                    per_sample_weights=per_sample_weights if weighted else None,
                )
                torch.cuda.synchronize()
            except RuntimeError as e:
                raise AssertionError(f"Forward error: {weighted=} {e}")

            grad = out.clone().detach()

            try:
                # Run backward
                out.backward(grad)
                torch.cuda.synchronize()
            except RuntimeError as e:
                raise AssertionError(f"Backward error: {weighted=} {e}")

        # Delete the op to save space
        del op

    @unittest.skipIf(*gpu_unavailable)
    def test_forward_nobag_large_grid(self) -> None:
        """
        Repro for the HIP 2**32 threads-per-launch overflow in the sequence
        (nobag) TBE forward kernel.

        The nobag forward launches
        grid = div_round_up(total_B, kForwardMaxThreads / kWarpSize) with
        block = dim3(kWarpSize, kForwardMaxThreads / kWarpSize), so the total
        thread count is ~= total_B * kWarpSize. On ROCm (kWarpSize = 64),
        total_B > 2**32 / 64 (~67.1M) exceeds the HIP limit. total_B = B * T;
        we drive it with a large B and a tiny 1-row table (D = 4) so tensors
        stay small -- the overflow is grid-driven, not memory-driven.

        Pre-fix on ROCm, FBGEMM_LAUNCH_KERNEL ->
        KernelLauncher::checkThreadCountNotExceeded TORCH_CHECK-fails. The fix
        caps the grid on ROCm and grid-strides the small nobag kernel to cover
        the full workload; the tail sentinel below fails if any strided bag is
        dropped.

        On CUDA (kWarpSize = 32) total threads stay under 2**32, so this is a
        ROCm-specific repro.
        """
        if torch.version.hip is None:
            self.skipTest("2**32 grid-overflow repro is ROCm-specific")

        device = torch.cuda.current_device()
        warp = 64
        D = 4
        E = 1
        # total_B = B (T = 1). Require total_B * warp > 2**32.
        B = (2**32) // warp + 2
        self.assertGreater(B * warp, 2**32)

        # offsets (B+1) i32 + indices B i32 + output B*D f32, with headroom.
        needed_bytes = ((B + 1) * 4 + B * 4 + B * D * 4) * 6 // 5
        total_memory = torch.cuda.get_device_properties().total_memory
        free_memory = total_memory - torch.cuda.memory_reserved()
        if free_memory < needed_bytes:
            self.skipTest(
                f"Skip test_forward_nobag_large_grid: free memory "
                f"({free_memory}) < needed ({needed_bytes})"
            )

        op = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[(E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)],
            output_dtype=SparseType.FP32,
            pooling_mode=PoolingMode.NONE,
            device=device,
        )

        # Row 0 of the (single) table = a known sentinel; every bag selects it.
        sentinel = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        with torch.no_grad():
            op.split_embedding_weights()[0].copy_(sentinel.view(E, D))

        indices = torch.zeros(B, dtype=torch.int, device=device)
        offsets = torch.arange(0, B + 1, dtype=torch.int, device=device)

        out = op(indices=indices, offsets=offsets)
        torch.cuda.synchronize()

        self.assertEqual(out.shape[0], B)
        # Head / middle / tail bags must all be produced; the tail only appears
        # if the ROCm grid-stride loop covers the whole (capped) workload.
        torch.testing.assert_close(out[0].cpu(), sentinel.cpu())
        torch.testing.assert_close(out[B // 2].cpu(), sentinel.cpu())
        torch.testing.assert_close(out[-1].cpu(), sentinel.cpu())
        del op, indices, offsets, out

    @unittest.skipIf(*gpu_unavailable)
    @given(**common_st)
    @settings(**common_settings)
    def test_forward_backward_large_fp32_emb_int32_indices(self, **kwargs: Any) -> None:
        """
        Test forward and backward TBE with a large FP32 embedding table and
        INT32 indices and offsets
        """
        self._execute_forward_backward_large_emb(
            weights_precision=SparseType.FP32,
            indices_dtype=torch.int,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(**common_st)
    @settings(**common_settings)
    def test_forward_backward_large_fp16_emb_int32_indices(self, **kwargs: Any) -> None:
        """
        Test forward and backward TBE with a large FP16 embedding table and
        INT32 indices and offsets
        """
        self._execute_forward_backward_large_emb(
            weights_precision=SparseType.FP16,
            indices_dtype=torch.int,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(**common_st)
    @settings(**common_settings)
    def test_forward_backward_large_fp32_emb_int64_indices(self, **kwargs: Any) -> None:
        """
        Test forward and backward TBE with a large FP32 embedding table and
        INT64 indices and offsets
        """
        self._execute_forward_backward_large_emb(
            weights_precision=SparseType.FP32,
            indices_dtype=torch.long,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(**common_st)
    @settings(**common_settings)
    def test_forward_backward_large_fp16_emb_int64_indices(self, **kwargs: Any) -> None:
        """
        Test forward and backward TBE with a large FP16 embedding table and
        INT64 indices and offsets
        """
        self._execute_forward_backward_large_emb(
            weights_precision=SparseType.FP16,
            indices_dtype=torch.long,
            **kwargs,
        )
