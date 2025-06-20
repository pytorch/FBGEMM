# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import skipIf, TestCase

import fbgemm_gpu

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    random_quant_scaled_tensor,
)
from fbgemm_gpu.tbe.cache.kv_embedding_ops_inference import KVEmbeddingInference
from fbgemm_gpu.tbe.utils import generate_requests

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)


@skipIf(open_source, "Not supported in open source yet")
class KVEmbeddingTest(TestCase):
    def test_forward(self) -> None:
        dim = 256
        num_tables = 4
        num_embeddings = 100
        batch_size = 2
        bag_size = 1
        num_requests = 1
        weights_precision = SparseType.INT8
        output_dtype = SparseType.FP16

        dimentions = [dim] * num_tables

        nbit_emb_cpu = IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    "",
                    num_embeddings,
                    d,
                    weights_precision,
                    EmbeddingLocation.HOST,
                )
                for d in dimentions
            ],
            output_dtype=output_dtype,
            device="cpu",
        )
        nbit_emb_cpu.fill_random_weights()
        # fill random scale bias
        nbit_weights = nbit_emb_cpu.split_embedding_weights()
        for dest_weight in nbit_weights:
            scale_bias = dest_weight[1]
            if scale_bias is not None:
                random_quant_scaled_tensor(
                    shape=scale_bias.shape,
                    device=nbit_emb_cpu.current_device,
                    output_tensor=scale_bias,
                )

        kv_emb_cpu = KVEmbeddingInference(
            [
                (
                    "",
                    num_embeddings,
                    d,
                    weights_precision,
                    EmbeddingLocation.HOST,
                )
                for d in dimentions
            ],
            output_dtype=output_dtype,
            device="cpu",
        )
        kv_emb_cpu.initialize_kv_embedding_cache()

        nbit_weights = nbit_emb_cpu.split_embedding_weights(split_scale_shifts=False)
        for i, (nbit_weight, _) in enumerate(nbit_weights):
            indices = torch.arange(0, nbit_weight.shape[0], dtype=torch.int64)
            kv_emb_cpu.embedding_inplace_update_per_table(
                i,
                indices,
                nbit_weight,
            )
        kv_emb_cpu.weight_initialized = True

        requests = generate_requests(
            num_requests,
            batch_size,
            num_tables,
            bag_size,
            num_embeddings,
            use_cpu=True,
        )

        for req in requests:
            indices = req.indices.int().cpu()
            offsets = req.offsets.int().cpu()

            nbit_emb_cpu_output = nbit_emb_cpu.forward(
                indices,
                offsets,
            )
            kv_emb_cpu_output = kv_emb_cpu.forward(
                indices,
                offsets,
            )
            print(f"nbit_emb_cpu_output: {nbit_emb_cpu_output}")
            print(f"kv_emb_cpu_output: {kv_emb_cpu_output}")
            self.assertTrue(
                torch.allclose(
                    input=nbit_emb_cpu_output, other=kv_emb_cpu_output, equal_nan=True
                )
            )
