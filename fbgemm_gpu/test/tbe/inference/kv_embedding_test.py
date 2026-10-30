# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
from typing import cast
from unittest import skipIf, TestCase
from unittest.mock import patch

import fbgemm_gpu
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    random_quant_scaled_tensor,
)
from fbgemm_gpu.tbe.cache import kv_embedding_ops_inference
from fbgemm_gpu.tbe.cache.kv_embedding_ops_inference import (
    _supports_packed_int4_rows,
    KVEmbeddingInference,
)
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode
from fbgemm_gpu.tbe.utils import generate_requests

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)


@skipIf(open_source, "Not supported in open source yet")
class KVEmbeddingTest(TestCase):
    def test_heterogeneous_int4_uses_legacy_rows(self) -> None:
        kv = KVEmbeddingInference(
            [
                ("", 4, 8, SparseType.INT4, EmbeddingLocation.HOST),
                ("", 4, 10, SparseType.INT4, EmbeddingLocation.HOST),
            ],
            pooling_mode=PoolingMode.SUM,
            output_dtype=SparseType.FP16,
            device="cpu",
        )

        self.assertFalse(kv.use_packed_int4_rows)
        kv.initialize_kv_embedding_cache()

        self.assertEqual(kv.row_alignment, 8)
        self.assertEqual(kv.kv_embedding_cache.get_lookup_row_bytes(), 16)

    def test_int4_module_uses_legacy_schema_before_rollout(self) -> None:
        kv = KVEmbeddingInference(
            [("", 4, 100, SparseType.INT4, EmbeddingLocation.HOST)],
            pooling_mode=PoolingMode.NONE,
            output_dtype=SparseType.INT4,
            device="cpu",
        )
        kv.initialize_weights()

        scripted = torch.jit.script(kv)
        self.assertFalse(scripted.use_packed_int4_rows)
        self.assertNotIn(
            "init_with_row_alignments",
            scripted.initialize_kv_embedding_cache.code,
        )

        archive = io.BytesIO()
        torch.jit.save(scripted, archive)
        archive.seek(0)
        loaded = torch.jit.load(archive)
        loaded.initialize_kv_embedding_cache()

        self.assertEqual(loaded.row_alignment, 8)

    def test_packed_int4_native_capability_probe(self) -> None:
        class LegacyNative:
            def get_lookup_row_bytes(self) -> int:
                return 54

        class CurrentNative:
            def init_with_row_alignments(self) -> None:
                return None

        self.assertFalse(_supports_packed_int4_rows(LegacyNative()))
        self.assertTrue(_supports_packed_int4_rows(CurrentNative()))

    def test_int4_forward_unaligned_logical_row(self) -> None:
        dim = 100
        num_embeddings = 16
        specs = [("", num_embeddings, dim, SparseType.INT4, EmbeddingLocation.HOST)]
        baseline = IntNBitTableBatchedEmbeddingBagsCodegen(
            specs,
            pooling_mode=PoolingMode.NONE,
            output_dtype=SparseType.INT4,
            device="cpu",
        )
        baseline.fill_random_weights()
        with patch.object(
            kv_embedding_ops_inference,
            "KV_INT4_PACKED_ROWS_ROLLOUT_ENABLED",
            True,
        ):
            kv = KVEmbeddingInference(
                specs,
                pooling_mode=PoolingMode.NONE,
                output_dtype=SparseType.INT4,
                device="cpu",
            )
        kv.initialize_kv_embedding_cache()
        fused_weights = baseline.split_embedding_weights(split_scale_shifts=False)[0][0]
        rows = torch.arange(num_embeddings, dtype=torch.int64)
        kv.embedding_inplace_update_per_table(0, rows, fused_weights)
        kv.weight_initialized = True
        indices = torch.tensor([3, 1, 7], dtype=torch.int32)
        offsets = torch.arange(indices.numel() + 1, dtype=torch.int32)

        expected = baseline(indices, offsets)
        actual = kv(indices, offsets)

        logical_bytes = dim // 2 + 4
        self.assertEqual(
            tuple(kv.kv_embedding_cache.get_embeddings(indices).shape),
            (3, logical_bytes),
        )
        self.assertEqual(
            actual.untyped_storage().nbytes(), indices.numel() * logical_bytes
        )
        self.assertTrue(torch.equal(actual.int_repr(), expected.int_repr()))

    def _grouped_int4_module(
        self, pooling_mode: PoolingMode, num_embeddings: int, dim: int, group_size: int
    ) -> tuple[KVEmbeddingInference, torch.Tensor]:
        """A grouped INT4 KV module loaded with random rows; also returns the
        decoded FP16 rows it serves."""
        kv = KVEmbeddingInference(
            [("", num_embeddings, dim, SparseType.INT4, EmbeddingLocation.HOST)],
            pooling_mode=pooling_mode,
            output_dtype=SparseType.FP16,
            device="cpu",
            grouped_int4_group_size=group_size,
        )
        kv.initialize_kv_embedding_cache()
        rows = torch.randn(
            num_embeddings, dim, generator=torch.Generator().manual_seed(0)
        )
        packed = torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
            (rows * 0.05).half(), group_size
        )
        kv.embedding_inplace_update_per_table(
            0, torch.arange(num_embeddings, dtype=torch.int64), packed
        )
        kv.weight_initialized = True
        decoded = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, group_size, dim)
        return kv, decoded

    def test_grouped_int4_lookup_serves_decoded_rows(self) -> None:
        dim, group_size, num_embeddings = 128, 8, 16
        kv, decoded = self._grouped_int4_module(
            PoolingMode.NONE, num_embeddings, dim, group_size
        )
        self.assertEqual(
            kv.kv_embedding_cache.get_lookup_row_bytes(), dim // 2 + dim // group_size
        )
        indices = torch.tensor([3, 1, 7, 3], dtype=torch.int32)
        offsets = torch.arange(indices.numel() + 1, dtype=torch.int32)

        actual = kv(indices, offsets)

        self.assertTrue(
            torch.equal(
                actual.view(torch.int16), decoded[indices.long()].view(torch.int16)
            )
        )

    def test_grouped_int4_pooled_lookup_matches_fp16_table(self) -> None:
        dim, group_size, num_embeddings = 128, 8, 16
        kv, decoded = self._grouped_int4_module(
            PoolingMode.SUM, num_embeddings, dim, group_size
        )
        fp16_table = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", num_embeddings, dim, SparseType.FP16, EmbeddingLocation.HOST)],
            pooling_mode=PoolingMode.SUM,
            output_dtype=SparseType.FP16,
            device="cpu",
        )
        fp16_table.fill_random_weights()
        fp16_table.split_embedding_weights()[0][0].copy_(decoded.view(torch.uint8))
        indices = torch.tensor([3, 1, 7, 0, 5, 5], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 3, 6], dtype=torch.int32)

        self.assertTrue(torch.equal(kv(indices, offsets), fp16_table(indices, offsets)))

    def test_grouped_int4_module_scripts_and_serves(self) -> None:
        dim, group_size, num_embeddings = 128, 8, 16
        # Scripted before the KV store is initialized, as a published module is.
        kv = KVEmbeddingInference(
            [("", num_embeddings, dim, SparseType.INT4, EmbeddingLocation.HOST)],
            pooling_mode=PoolingMode.NONE,
            output_dtype=SparseType.FP16,
            device="cpu",
            grouped_int4_group_size=group_size,
        )
        kv.initialize_weights()
        archive = io.BytesIO()
        torch.jit.save(torch.jit.script(kv), archive)
        archive.seek(0)
        loaded = torch.jit.load(archive)
        loaded.initialize_kv_embedding_cache()
        packed = torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
            (torch.randn(num_embeddings, dim) * 0.05).half(), group_size
        )
        decoded = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, group_size, dim)
        loaded.embedding_inplace_update_per_table(
            0, torch.arange(num_embeddings, dtype=torch.int64), packed, None
        )
        loaded.weight_initialized = True
        indices = torch.tensor([2, 9], dtype=torch.int32)
        offsets = torch.arange(indices.numel() + 1, dtype=torch.int32)

        actual = loaded(indices, offsets)

        self.assertTrue(
            torch.equal(
                actual.view(torch.int16), decoded[indices.long()].view(torch.int16)
            )
        )

    def test_grouped_int4_decode_is_compiled_out_without_groups(self) -> None:
        for group_size, expect_decode in ((0, False), (8, True)):
            with self.subTest(group_size=group_size):
                kv = KVEmbeddingInference(
                    [("", 16, 128, SparseType.INT4, EmbeddingLocation.HOST)],
                    pooling_mode=PoolingMode.NONE,
                    output_dtype=SparseType.FP16,
                    device="cpu",
                    grouped_int4_group_size=group_size,
                )
                kv.initialize_weights()

                forward_code = cast(
                    torch.jit.RecursiveScriptModule, torch.jit.script(kv)
                ).code

                self.assertEqual("GroupedInt4E4M3ToHalf" in forward_code, expect_decode)

    def test_grouped_int4_module_built_on_meta_serves_after_serving_init(
        self,
    ) -> None:
        dim, group_size, num_embeddings = 128, 8, 16
        # Publish builds the module on meta; serving loads it, moves it to the
        # host and rebuilds its buffers before loading rows.
        kv = KVEmbeddingInference(
            [("", num_embeddings, dim, SparseType.INT4, EmbeddingLocation.HOST)],
            pooling_mode=PoolingMode.NONE,
            output_dtype=SparseType.FP16,
            device="meta",
            embedding_cache_mode=True,
            grouped_int4_group_size=group_size,
        )
        kv.initialize_weights()
        archive = io.BytesIO()
        torch.jit.save(torch.jit.script(kv), archive)
        archive.seek(0)
        loaded = torch.jit.load(archive)
        loaded.reset_weights_placements_and_offsets(
            torch.device("cpu"), EmbeddingLocation.HOST.value
        )
        loaded.recompute_module_buffers()
        loaded.initialize_kv_embedding_cache()
        packed = torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
            (torch.randn(num_embeddings, dim) * 0.05).half(), group_size
        )
        decoded = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, group_size, dim)
        loaded.embedding_inplace_update_per_table(
            0, torch.arange(num_embeddings - 1, dtype=torch.int64), packed[:-1], None
        )
        loaded.weight_initialized = True
        indices = torch.tensor([2, 9, num_embeddings - 1], dtype=torch.int32)
        offsets = torch.arange(indices.numel() + 1, dtype=torch.int32)

        actual = loaded(indices, offsets)

        self.assertTrue(
            torch.equal(actual[:2].view(torch.int16), decoded[[2, 9]].view(torch.int16))
        )
        # A missing id is a zero row, as in training's embedding cache mode.
        self.assertTrue(torch.equal(actual[2], torch.zeros(dim, dtype=torch.float16)))

    def test_grouped_int4_rejects_rows_of_another_width(self) -> None:
        dim, group_size, num_embeddings = 128, 8, 4
        kv, _ = self._grouped_int4_module(
            PoolingMode.NONE, num_embeddings, dim, group_size
        )
        rowwise = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
            torch.randn(num_embeddings, dim).half(), 4
        )
        # Rowwise rows as quantized (68 bytes) and padded to 8 bytes (72 bytes).
        padded = torch.zeros(num_embeddings, 72, dtype=torch.uint8)
        wide = torch.zeros(num_embeddings, 96, dtype=torch.uint8)
        for rows in (rowwise, padded, wide):
            with self.subTest(width=rows.size(1)), self.assertRaises(ValueError):
                kv.embedding_inplace_update_per_table(
                    0, torch.arange(num_embeddings, dtype=torch.int64), rows
                )

    def test_grouped_int4_rejects_other_group_sizes_and_quantized_outputs(
        self,
    ) -> None:
        specs = [("", 4, 128, SparseType.INT4, EmbeddingLocation.HOST)]
        cases = [
            # Group size 16 rows are 72 bytes, as wide as padded rowwise rows.
            (16, SparseType.FP16),
            (4, SparseType.FP16),
            (8, SparseType.INT4),
            (8, SparseType.INT8),
        ]
        for group_size, output_dtype in cases:
            with self.subTest(
                group_size=group_size, output_dtype=output_dtype
            ), self.assertRaises(AssertionError):
                KVEmbeddingInference(
                    specs,
                    pooling_mode=PoolingMode.NONE,
                    output_dtype=output_dtype,
                    device="cpu",
                    grouped_int4_group_size=group_size,
                )

    def test_grouped_int4_rejects_unsupported_tables(self) -> None:
        bad_specs = [
            [("", 4, 128, SparseType.INT8, EmbeddingLocation.HOST)],
            [
                ("", 4, 128, SparseType.INT4, EmbeddingLocation.HOST),
                ("", 4, 64, SparseType.INT4, EmbeddingLocation.HOST),
            ],
            [("", 4, 12, SparseType.INT4, EmbeddingLocation.HOST)],
            # dim / group_size == 4: as wide as a rowwise INT4 row.
            [("", 4, 32, SparseType.INT4, EmbeddingLocation.HOST)],
        ]
        for specs in bad_specs:
            with self.subTest(specs=specs), self.assertRaises(AssertionError):
                KVEmbeddingInference(
                    specs,
                    pooling_mode=PoolingMode.NONE,
                    output_dtype=SparseType.FP16,
                    device="cpu",
                    grouped_int4_group_size=8,
                )

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
            # pyre-fixme[6]: Type-identity mismatch on EmbeddingLocation between shell
            # and canonical package; resolves once D103477971 unifies the classes via re-export.
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
