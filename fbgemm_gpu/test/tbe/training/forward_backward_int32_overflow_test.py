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
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation
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
    "verbosity": Verbosity.verbose,
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
