#!/usr/bin/env python3

# pyre-unsafe

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import fbgemm_gpu.split_table_batched_embeddings_ops as split_table_batched_embeddings_ops
import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_embedding_inference_converter import SplitEmbInferenceConverter
from fbgemm_gpu.split_table_batched_embeddings_ops import OptimType
from hypothesis import Verbosity, given, settings
from torch import nn


EMB_WEIGHT_UNIFORM_INIT_BOUND = 0.000316
MAX_EXAMPLES = 40


def div_round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def to_device(t: torch.Tensor, use_cpu: bool) -> torch.Tensor:
    return t.cpu() if use_cpu else t.cuda()


def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor, use_cpu: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        to_device(merged_indices.contiguous().view(-1), use_cpu),
        to_device(
            torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long(),
            use_cpu,
        ),
    )


class SparseArch(nn.Module):
    """
    The testing module with split table batched embedding op
    """

    def __init__(
        self,
        emb_dim,
        num_tables,
        num_rows,
        use_cpu,
    ):
        super().__init__()
        pooling_mode = split_table_batched_embeddings_ops.PoolingMode.SUM
        Ds = [emb_dim] * num_tables
        Es = [num_rows] * num_tables

        device = (
            split_table_batched_embeddings_ops.ComputeDevice.CPU
            if use_cpu
            else split_table_batched_embeddings_ops.ComputeDevice.CUDA
        )
        loc = (
            split_table_batched_embeddings_ops.EmbeddingLocation.HOST
            if use_cpu
            else split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
        )

        self.emb_module = (
            split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        E,
                        D,
                        loc,
                        device,
                    )
                    for (E, D) in zip(Es, Ds)
                ],
                weights_precision=SparseType.FP32,
                optimizer=OptimType.EXACT_SGD,
                learning_rate=0.05,
                pooling_mode=pooling_mode,
            )
        )

        self.emb_module.init_embedding_weights_uniform(
            -EMB_WEIGHT_UNIFORM_INIT_BOUND, +EMB_WEIGHT_UNIFORM_INIT_BOUND
        )

    def forward(self, indices, offsets):
        return self.emb_module(indices, offsets)


class QuantizedSplitEmbeddingsTest(unittest.TestCase):
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
            ]
        ),
        quantize_type=st.sampled_from(
            [
                SparseType.INT8,
                SparseType.INT4,
                # TODO: support SparseType.INT2,
                SparseType.FP16,
            ]
        ),
        use_cpu=st.booleans() if torch.cuda.is_available() else st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_quantize_workflow(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        quantize_type: SparseType,
        use_cpu: bool,
    ) -> None:
        E = int(10 ** log_E)
        Es = [E] * T
        D_alignment = 8 if not quantize_type == SparseType.INT2 else 16
        D = div_round_up(D, D_alignment)

        xs = [torch.randint(low=0, high=e, size=(B, L)) for e in Es]
        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        # indices: T, B, L; offsets: T * B + 1
        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=use_cpu)
        sparse_arch = SparseArch(emb_dim=D, num_tables=T, num_rows=E, use_cpu=use_cpu)

        # Fake quantize to make the original weight in FP32 all be exactly
        # representable by INT8 row-wise quantized values
        if quantize_type == quantize_type.INT8:
            for t in range(T):
                sparse_arch.emb_module.split_embedding_weights()[t].data.copy_(
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            sparse_arch.emb_module.split_embedding_weights()[t].data
                        )
                    )
                )
        elif quantize_type == quantize_type.INT4 or quantize_type == quantize_type.INT2:
            for t in range(T):
                sparse_arch.emb_module.split_embedding_weights()[t].data.copy_(
                    torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                        torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                            sparse_arch.emb_module.split_embedding_weights()[t].data,
                            bit_rate=quantize_type.bit_rate(),
                        ),
                        bit_rate=quantize_type.bit_rate(),
                    )
                )

        emb_out = sparse_arch(indices, offsets)  # B, T, D

        # Apply the quantization transformations on the model!
        split_emb_infer_converter = SplitEmbInferenceConverter(
            quantize_type=quantize_type
        )
        split_emb_infer_converter.convert_model(sparse_arch)
        assert (
            type(sparse_arch.emb_module)
            == split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen
        )
        assert sparse_arch.emb_module.use_cpu == use_cpu
        quantized_emb_out = sparse_arch(indices.int(), offsets.int())  # B, T, D

        # Compare FP32 emb module vs. quantize_type (FP16, INT8, INT4, INT2) emb module
        torch.testing.assert_allclose(
            emb_out.float().cpu(),
            quantized_emb_out.float().cpu(),
            atol=1.0e-1,
            rtol=1.0e-1,
        )


if __name__ == "__main__":
    unittest.main()
