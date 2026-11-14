#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode

from .. import common  # noqa F401


class LookupModule(torch.nn.Module):
    def __init__(
        self,
        weights_tys: tuple[SparseType, ...] = (SparseType.INT4,),
        output_dtype: SparseType = SparseType.FP32,
        pooling_mode: PoolingMode = PoolingMode.SUM,
    ) -> None:
        super().__init__()
        self.output_dtype = output_dtype.as_int()
        self.pooling_mode = int(pooling_mode)
        self.total_D = 16 * len(weights_tys)
        tables = []
        weights_offsets = []
        for ty in weights_tys:
            weights_offsets.append(sum(table.numel() for table in tables))
            if ty == SparseType.INT4:
                row = torch.cat(
                    (
                        torch.tensor([1.5, 0.25], dtype=torch.float16).view(
                            torch.uint8
                        ),
                        torch.full((8,), 0x22, dtype=torch.uint8),
                        torch.zeros(4, dtype=torch.uint8),
                    )
                )
            else:
                row = torch.ones(16, dtype=torch.float16).view(torch.uint8)
            tables.append(row.repeat(2))
        self.register_buffer("dev_weights", torch.cat(tables))
        self.register_buffer("uvm_weights", torch.empty(0, dtype=torch.uint8))
        self.register_buffer(
            "weights_placements",
            torch.full((len(tables),), int(EmbeddingLocation.HOST), dtype=torch.int32),
        )
        self.register_buffer(
            "weights_offsets", torch.tensor(weights_offsets, dtype=torch.int64)
        )
        self.register_buffer(
            "weights_tys",
            torch.tensor([ty.as_int() for ty in weights_tys], dtype=torch.uint8),
        )
        self.register_buffer(
            "D_offsets", torch.arange(len(tables) + 1, dtype=torch.int32) * 16
        )

    def lookup_args(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        indice_weights: torch.Tensor | None = None,
    ) -> tuple[Any, ...]:
        return (
            self.dev_weights,
            self.uvm_weights,
            self.weights_placements,
            self.weights_offsets,
            self.weights_tys,
            self.D_offsets,
            self.total_D,
            0,
            16,
            0,
            16,
            0,
            indices,
            offsets,
            self.pooling_mode,
            indice_weights,
            self.output_dtype,
            None,
            None,
            16,
        )

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        indice_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function_epo_gpu_compatible_cpu(
            *self.lookup_args(indices, offsets, indice_weights)
        )


class GpuCompatibleCpuTest(unittest.TestCase):
    def test_mixed_tables(self) -> None:
        indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
        for output_dtype in (SparseType.FP16, SparseType.FP32):
            for pooling_mode in (PoolingMode.SUM, PoolingMode.MEAN):
                with self.subTest(output_dtype=output_dtype, pooling_mode=pooling_mode):
                    module = LookupModule(
                        (SparseType.FP16, SparseType.INT4), output_dtype, pooling_mode
                    )
                    expected = torch.tensor(
                        [[2.0] * 16 + [6.5] * 16], dtype=output_dtype.as_dtype()
                    )
                    if pooling_mode == PoolingMode.MEAN:
                        expected /= 2
                    torch.testing.assert_close(
                        module(indices, offsets), expected, rtol=0, atol=0
                    )

    def test_rejects_unsupported_int4_modes(self) -> None:
        indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
        types = (SparseType.FP16, SparseType.INT4)
        with self.assertRaisesRegex(RuntimeError, "requires unweighted tables"):
            LookupModule(types)(indices, offsets, torch.ones(4))
        with self.assertRaisesRegex(RuntimeError, "does not support no-bag lookup"):
            LookupModule(types, pooling_mode=PoolingMode.NONE)(indices, offsets)
        with self.assertRaisesRegex(RuntimeError, "requires FP16 or FP32 output"):
            LookupModule(types, output_dtype=SparseType.BF16)(indices, offsets)

    def test_non_int4_preserves_legacy_modes(self) -> None:
        indices = torch.tensor([0, 1], dtype=torch.int64)
        offsets = torch.tensor([0, 2], dtype=torch.int64)
        for pooling_mode in (PoolingMode.SUM, PoolingMode.NONE):
            for indice_weights in (None, torch.tensor([0.5, 2.0])):
                with self.subTest(
                    pooling_mode=pooling_mode, weighted=indice_weights is not None
                ):
                    module = LookupModule(
                        (SparseType.FP16,), SparseType.BF16, pooling_mode
                    )
                    expected = torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
                        *module.lookup_args(indices, offsets, indice_weights)
                    )
                    torch.testing.assert_close(
                        module(indices, offsets, indice_weights),
                        expected,
                        rtol=0,
                        atol=0,
                    )

    def test_fake_tensor(self) -> None:
        op = (
            torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function_epo_gpu_compatible_cpu.default
        )
        for output_dtype in (SparseType.FP16, SparseType.FP32):
            for batch_size in (0, 2):
                with self.subTest(output_dtype=output_dtype, batch_size=batch_size):
                    module = LookupModule(output_dtype=output_dtype)
                    indices = torch.tensor([0, 1] * batch_size, dtype=torch.int64)
                    offsets = torch.arange(batch_size + 1, dtype=torch.int64) * 2
                    result = torch.library.opcheck(
                        op,
                        module.lookup_args(indices, offsets),
                        test_utils=("test_schema", "test_faketensor"),
                    )
                    self.assertEqual(
                        result,
                        {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"},
                    )

    def test_export_dynamic_batch(self) -> None:
        module = LookupModule()
        indices = torch.tensor([0, 1] * 2, dtype=torch.int64)
        offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
        batch = torch.export.Dim("batch", min=2, max=8)
        exported = torch.export.export(
            module,
            (indices, offsets),
            dynamic_shapes=({0: 2 * batch}, {0: batch + 1}),
        ).module()
        for batch_size in (2, 3, 4):
            indices = torch.tensor([0, 1] * batch_size, dtype=torch.int64)
            offsets = torch.arange(batch_size + 1, dtype=torch.int64) * 2
            torch.testing.assert_close(
                exported(indices, offsets),
                torch.full((batch_size, 16), 6.5),
                rtol=0,
                atol=0,
            )

    def test_compile(self) -> None:
        for output_dtype in (SparseType.FP16, SparseType.FP32):
            module = LookupModule(output_dtype=output_dtype)
            compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
            indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
            offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
            torch.testing.assert_close(
                compiled(indices, offsets), module(indices, offsets), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
