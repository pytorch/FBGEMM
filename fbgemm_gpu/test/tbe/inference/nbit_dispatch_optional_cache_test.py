#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode

from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class NBitDispatchOptionalCacheTest(unittest.TestCase):
    @staticmethod
    def _lookup(
        embedding: IntNBitTableBatchedEmbeddingBagsCodegen,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        pooling_mode: PoolingMode,
        indice_weights: torch.Tensor | None,
        cache_weights: torch.Tensor | None,
        cache_locations: torch.Tensor | None,
        weights_placements: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
            dev_weights=embedding.weights_dev,
            uvm_weights=embedding.weights_uvm,
            weights_placements=(
                embedding.weights_placements
                if weights_placements is None
                else weights_placements
            ),
            weights_offsets=embedding.weights_offsets,
            weights_tys=embedding.weights_tys,
            D_offsets=embedding.D_offsets,
            total_D=embedding.total_D,
            max_int2_D=embedding.max_int2_D,
            max_int4_D=embedding.max_int4_D,
            max_int8_D=embedding.max_int8_D,
            max_float16_D=embedding.max_float16_D,
            max_float32_D=embedding.max_float32_D,
            indices=indices,
            offsets=offsets,
            pooling_mode=int(pooling_mode),
            indice_weights=indice_weights,
            output_dtype=embedding.output_dtype,
            lxu_cache_weights=cache_weights,
            lxu_cache_locations=cache_locations,
            row_alignment=embedding.row_alignment,
            max_float8_D=embedding.max_float8_D,
            fp8_exponent_bits=embedding.fp8_exponent_bits,
            fp8_exponent_bias=embedding.fp8_exponent_bias,
        )

    @unittest.skipIf(bool(gpu_unavailable[0]), str(gpu_unavailable[1]))
    def test_cache_optionals_are_lazy_for_all_dispatch_paths(self) -> None:
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None:
            self.fail("An accelerator is required")
        device = torch.device(accelerator.type, torch.cuda.current_device())
        embedding = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", 128, 256, SparseType.INT8, EmbeddingLocation.DEVICE)],
            device=device,
            output_dtype=SparseType.FP16,
        )
        embedding.fill_random_weights()

        # Citrine C3: construct test tensors directly on the accelerator.
        indices = torch.arange(12, dtype=torch.int32, device=device)
        offsets = torch.arange(0, 13, 3, dtype=torch.int32, device=device)
        indice_weights = torch.linspace(0.25, 1.0, 12, device=device)
        cache_weights = torch.empty((0, 0), dtype=torch.uint8, device=device)
        cache_locations = torch.empty((0,), dtype=torch.int32, device=device)
        cases = [
            ("nobag", PoolingMode.NONE, None),
            ("pooled_unweighted", PoolingMode.SUM, None),
            ("weighted", PoolingMode.SUM, indice_weights),
        ]

        for _, pooling_mode, weights in cases:
            self._lookup(
                embedding,
                indices,
                offsets,
                pooling_mode,
                weights,
                cache_weights,
                cache_locations,
            )
        torch.cuda.synchronize()

        supplied_outputs = []
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]  # pyre-ignore[16]
        ) as profiler:
            for _, pooling_mode, weights in cases:
                supplied_outputs.append(
                    self._lookup(
                        embedding,
                        indices,
                        offsets,
                        pooling_mode,
                        weights,
                        cache_weights,
                        cache_locations,
                    )
                )
            torch.cuda.synchronize()

        empty_count = sum(
            event.count
            for event in profiler.key_averages()
            if event.key == "aten::empty"
        )
        self.assertEqual(empty_count, len(cases))

        for (name, pooling_mode, weights), supplied_output in zip(
            cases, supplied_outputs
        ):
            with self.subTest(path=name, cache="device_only"):
                nullopt_output = self._lookup(
                    embedding,
                    indices,
                    offsets,
                    pooling_mode,
                    weights,
                    None,
                    None,
                )
                torch.testing.assert_close(supplied_output, nullopt_output)
                self.assertEqual(supplied_output.dtype, nullopt_output.dtype)
                self.assertEqual(supplied_output.device, nullopt_output.device)

        cached_embedding = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", 128, 256, SparseType.INT8, EmbeddingLocation.MANAGED_CACHING)],
            device=device,
            output_dtype=SparseType.FP16,
        )
        cached_embedding.fill_random_weights()
        uncached_placements = torch.full_like(
            cached_embedding.weights_placements, EmbeddingLocation.MANAGED.value
        )
        for name, pooling_mode, weights in cases:
            with self.subTest(path=name, cache="enabled"):
                cached_embedding.prefetch(indices, offsets)
                cached_locations = cached_embedding.lxu_cache_locations_list.pop()
                cached_embedding.timestep_prefetch_size.decrement()
                cached_output = self._lookup(
                    cached_embedding,
                    indices,
                    offsets,
                    pooling_mode,
                    weights,
                    cached_embedding.lxu_cache_weights,
                    cached_locations,
                )
                nullopt_output = self._lookup(
                    cached_embedding,
                    indices,
                    offsets,
                    pooling_mode,
                    weights,
                    None,
                    None,
                    weights_placements=uncached_placements,
                )
                torch.testing.assert_close(cached_output, nullopt_output)
                self.assertEqual(cached_output.dtype, nullopt_output.dtype)
                self.assertEqual(cached_output.device, nullopt_output.device)


if __name__ == "__main__":
    unittest.main()
