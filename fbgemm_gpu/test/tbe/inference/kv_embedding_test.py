# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
from typing import cast
from unittest import skipIf, skipUnless, TestCase
from unittest.mock import patch

import fbgemm_gpu
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    random_quant_scaled_tensor,
)
from fbgemm_gpu.tbe.cache import (  # @manual=//deeplearning/fbgemm/fbgemm_gpu:kv_zch_gpu_bucket_wise_embedding  # noqa: B950
    kv_embedding_ops_inference,
    kv_zch_gpu_bucket_wise_embedding,
)
from fbgemm_gpu.tbe.cache.kv_embedding_ops_inference import (
    _supports_packed_int4_rows,
    KVEmbeddingInference,
)
from fbgemm_gpu.tbe.cache.kv_zch_gpu_bucket_wise_embedding import (  # @manual=//deeplearning/fbgemm/fbgemm_gpu:kv_zch_gpu_bucket_wise_embedding  # noqa: B950
    KV_SHARD_DELTA_REFUSAL_MARKER,
    KVZCHGPUBucketWiseEmbeddingInference,
)
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode
from fbgemm_gpu.tbe.utils import generate_requests
from fbgemm_gpu.utils.torchscript import has_torchscript

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

    @skipUnless(
        has_torchscript(), "torch.jit.script is unavailable in this PyTorch build"
    )
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

    def test_shard_refuses_delta_ids_outside_its_rows(self) -> None:
        dim, shard_rows, shard_offset = 16, 8, 8
        specs = [("", shard_rows, dim, SparseType.INT8, EmbeddingLocation.HOST)]
        baseline = IntNBitTableBatchedEmbeddingBagsCodegen(
            specs, pooling_mode=PoolingMode.NONE, device="cpu"
        )
        baseline.fill_random_weights()
        rows = baseline.split_embedding_weights(split_scale_shifts=False)[0][0]
        for module_fn in ("eager", "scripted"):
            with self.subTest(module_fn=module_fn):
                kv = KVZCHGPUBucketWiseEmbeddingInference(
                    specs,
                    pooling_mode=PoolingMode.NONE,
                    output_dtype=SparseType.FP16,
                    device="cpu",
                    embedding_cache_mode=True,
                )
                if module_fn == "scripted":
                    kv.initialize_weights()
                    kv = torch.jit.script(kv)
                kv.init_tbe_config([shard_offset])
                kv.initialize_kv_embedding_cache()
                stored = kv.kv_embedding_cache.get_embeddings

                # Shard 1 of a table whose delta names it with shard 0's ids.
                with self.assertRaises((RuntimeError, torch.jit.Error)) as refusal:
                    kv.embedding_inplace_update_per_table(
                        0, torch.tensor([7, 9]), rows[:2]
                    )
                # The predictor counts refusals by the module's marker.
                self.assertEqual(kv.delta_refusal_marker, KV_SHARD_DELTA_REFUSAL_MARKER)
                self.assertIn(kv.delta_refusal_marker, str(refusal.exception))
                self.assertTrue(torch.all(stored(torch.arange(8)) == 0))

                kv.embedding_inplace_update_per_table(
                    0, torch.tensor([shard_offset + 1]), rows[1:2]
                )
                self.assertTrue(
                    torch.equal(stored(torch.tensor([1]))[:, : rows.size(1)], rows[1:2])
                )

    def test_host_module_keeps_trunk_script_api(self) -> None:
        kv = KVEmbeddingInference(
            [("cu", 16, 128, SparseType.INT8, EmbeddingLocation.HOST)],
            pooling_mode=PoolingMode.NONE,
            output_dtype=SparseType.FP16,
            device="cpu",
        )
        for name in (
            "zeros_before_kv_init",
            "kv_cuda_row_alignment",
            "fill_new_embeddings_supported",
        ):
            self.assertFalse(hasattr(kv, name))
        kv.initialize_weights()
        scripted = torch.jit.script(kv)
        self.assertNotIn(
            "embedding_fill_new_rows_per_table", scripted._c._method_names()
        )

    def test_update_needs_one_row_per_index(self) -> None:
        kv = KVZCHGPUBucketWiseEmbeddingInference(
            [("", 16, 16, SparseType.INT8, EmbeddingLocation.HOST)],
            pooling_mode=PoolingMode.NONE,
            device="cpu",
        )
        kv.initialize_kv_embedding_cache()
        for rows in (1, 3):
            with self.assertRaisesRegex(RuntimeError, "one row per index"):
                kv.embedding_inplace_update_per_table(
                    0, torch.tensor([1, 2]), torch.zeros(rows, 24, dtype=torch.uint8)
                )

    def test_fill_only_native_capability_is_compiled_out_unless_enabled(self) -> None:
        # Calls to the native method reach the artifact only for a table
        # published with fill_new_embeddings_enabled, and only when the native
        # class has the method (an older predictor's does not).
        def without_native_fill(obj: object, name: str) -> bool:
            return name != "fill_new_embeddings" and hasattr(obj, name)

        for enabled, older_native in ((False, False), (True, True), (True, False)):
            with self.subTest(enabled=enabled, older_native=older_native):
                with patch.object(
                    kv_zch_gpu_bucket_wise_embedding,
                    "hasattr",
                    without_native_fill if older_native else hasattr,
                    create=True,
                ):
                    kv = KVZCHGPUBucketWiseEmbeddingInference(
                        [("cu", 16, 16, SparseType.INT8, EmbeddingLocation.HOST)],
                        pooling_mode=PoolingMode.NONE,
                        device="cpu",
                        fill_new_embeddings_enabled=enabled,
                    )
                supported = enabled and not older_native
                self.assertEqual(kv.fill_new_embeddings_supported, supported)
                kv.initialize_weights()
                scripted = torch.jit.script(kv)
                archive = io.BytesIO()
                torch.jit.save(scripted, archive)
                archive.seek(0)
                loaded = torch.jit.load(archive)
                for module in (scripted, loaded):
                    self.assertTrue(module._c.hasattr("fill_new_embeddings_supported"))
                    self.assertEqual(module.fill_new_embeddings_supported, supported)
                    self.assertEqual(
                        ".fill_new_embeddings("
                        in module.embedding_fill_new_rows_per_table.code,
                        supported,
                    )
                    module.initialize_kv_embedding_cache()
                    module.log_inplace_update_stats()
                if supported:
                    continue
                for module in (kv, scripted, loaded):
                    with self.assertRaisesRegex(
                        (RuntimeError, torch.jit.Error),
                        "require native fill_new_embeddings",
                    ):
                        module.embedding_fill_new_rows_per_table(
                            0, torch.tensor([1]), torch.zeros(1, 20, dtype=torch.uint8)
                        )

    def test_set_device_is_exported(self) -> None:
        kv = KVZCHGPUBucketWiseEmbeddingInference(
            [("", 16, 128, SparseType.INT4, EmbeddingLocation.HOST)],
            device="cpu",
        )

        kv.initialize_weights()
        scripted = torch.jit.script(kv)

        # Not the device the module was built on, so a no-op fails.
        scripted.set_device("meta")

        self.assertEqual(scripted.current_device, torch.device("meta"))

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

    def _fill_only_shard(
        self, shard_offset: int, module_fn: str = "eager"
    ) -> tuple[KVZCHGPUBucketWiseEmbeddingInference, torch.Tensor]:
        """A row-wise shard of 8 INT8 rows starting at global row
        shard_offset, with zero rows for absent ids, and 8 random rows."""
        dim, shard_rows = 16, 8
        specs = [("", shard_rows, dim, SparseType.INT8, EmbeddingLocation.HOST)]
        baseline = IntNBitTableBatchedEmbeddingBagsCodegen(
            specs, pooling_mode=PoolingMode.NONE, device="cpu"
        )
        baseline.fill_random_weights()
        rows = baseline.split_embedding_weights(split_scale_shifts=False)[0][0]
        kv = KVZCHGPUBucketWiseEmbeddingInference(
            specs,
            pooling_mode=PoolingMode.NONE,
            output_dtype=SparseType.FP16,
            device="cpu",
            embedding_cache_mode=True,
            fill_new_embeddings_enabled=True,
        )
        if module_fn == "scripted":
            kv.initialize_weights()
            # The scripted module exposes the same exported methods.
            kv = cast(KVZCHGPUBucketWiseEmbeddingInference, torch.jit.script(kv))
        kv.init_tbe_config([shard_offset])
        kv.initialize_kv_embedding_cache()
        return kv, rows

    def _stored(self, kv: KVEmbeddingInference, width: int) -> torch.Tensor:
        return kv.kv_embedding_cache.get_embeddings(torch.arange(8, dtype=torch.int64))[
            :, :width
        ]

    def test_fill_only_delta_writes_only_new_in_shard_rows(self) -> None:
        for shard_offset in (0, 8):
            for module_fn in ("eager", "scripted"):
                with self.subTest(shard_offset=shard_offset, module_fn=module_fn):
                    kv, rows = self._fill_only_shard(shard_offset, module_fn)
                    width = rows.size(1)
                    # The snapshot load already wrote the shard's row 1.
                    kv.embedding_inplace_update_per_table(
                        0, torch.tensor([shard_offset + 1]), rows[5:6]
                    )
                    # Two rows on each side of the shard, and the whole shard.
                    ids = torch.arange(shard_offset - 2, shard_offset + 10)
                    deltas = torch.cat([rows[:2], rows, rows[:2]])

                    kv.embedding_fill_new_rows_per_table(0, ids, deltas)

                    expected = rows.clone()
                    expected[1] = rows[5]
                    self.assertTrue(torch.equal(self._stored(kv, width), expected))
                    self.assertEqual(kv.kv_embedding_cache.get_fill_stats(), [7, 1])
                    kv.log_inplace_update_stats()

    def test_fill_only_delta_outside_the_shard_writes_nothing(self) -> None:
        kv, rows = self._fill_only_shard(8)
        kv.embedding_fill_new_rows_per_table(0, torch.tensor([0, 16]), rows[:2])
        self.assertEqual(kv.kv_embedding_cache.get_fill_stats(), [0, 0])
        self.assertTrue(torch.all(self._stored(kv, rows.size(1)) == 0))

    def test_fill_only_delta_writes_zero_rows_and_one_row_per_id(self) -> None:
        kv, rows = self._fill_only_shard(0)
        width = rows.size(1)
        # A snapshot that carried id 2 unfilled stored an all-zero row for it.
        kv.embedding_inplace_update_per_table(
            0, torch.tensor([2]), torch.zeros_like(rows[:1])
        )
        kv.embedding_fill_new_rows_per_table(
            0, torch.tensor([2, 3, 3]), torch.stack([rows[2], rows[3], rows[3]])
        )
        self.assertTrue(torch.equal(self._stored(kv, width)[2:4], rows[2:4]))
        self.assertEqual(kv.kv_embedding_cache.get_fill_stats(), [2, 1])

    def test_fill_only_delta_takes_cuda_padded_rows_of_a_gpu_shard(self) -> None:
        # SparseDelta pads rows to 16 bytes for CUDA modules.
        kv, _ = self._fill_only_shard(0)
        cache = kv.kv_embedding_cache
        width = cache.get_max_row_bytes()
        padded_width = (width + 15) // 16 * 16
        self.assertGreater(padded_width, width + 1)
        rows = torch.randint(1, 255, (2, width), dtype=torch.uint8)
        padded = torch.zeros(2, padded_width, dtype=torch.uint8)
        padded[:, :width] = rows
        # Only a planned KV-ZCH GPU serving shard takes padded rows.
        with self.assertRaisesRegex(RuntimeError, "wider than KV storage"):
            cache.fill_new_embeddings(torch.tensor([1]), padded[:1])
        cache.set_hbm_cache_rows(0)
        for wider in (width + 1, padded_width - 1, padded_width + 1):
            with self.assertRaisesRegex(RuntimeError, "wider than KV storage"):
                cache.fill_new_embeddings(
                    torch.tensor([1]), torch.zeros(1, wider, dtype=torch.uint8)
                )
        tail = padded[:1].clone()
        tail[0, -1] = 1
        with self.assertRaisesRegex(RuntimeError, "non-zero bytes past"):
            cache.fill_new_embeddings(torch.tensor([1]), tail)
        self.assertEqual(cache.fill_new_embeddings(torch.tensor([1, 4]), padded), 2)
        self.assertTrue(torch.equal(self._stored(kv, width)[[1, 4]], rows))

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

    @skipUnless(
        has_torchscript(), "torch.jit.script is unavailable in this PyTorch build"
    )
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

    @skipUnless(
        has_torchscript(), "torch.jit.script is unavailable in this PyTorch build"
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

    @skipUnless(
        has_torchscript(), "torch.jit.script is unavailable in this PyTorch build"
    )
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

    def test_cpu_forward_matches_parent(self) -> None:
        # The CPU forward is a copy of KVEmbeddingInference's.
        for pooling_mode, weight_ty in (
            (PoolingMode.SUM, SparseType.INT8),
            (PoolingMode.SUM, SparseType.INT4),
            (PoolingMode.NONE, SparseType.INT4),
        ):
            with self.subTest(pooling_mode=pooling_mode, weight_ty=weight_ty):
                specs = [
                    ("", 32, 64, weight_ty, EmbeddingLocation.HOST) for _ in range(2)
                ]
                reference = IntNBitTableBatchedEmbeddingBagsCodegen(
                    specs, pooling_mode=pooling_mode, device="cpu"
                )
                _fill_finite_weights(reference)
                outputs = []
                for cls in (KVEmbeddingInference, KVZCHGPUBucketWiseEmbeddingInference):
                    kv = cls(
                        specs,
                        pooling_mode=pooling_mode,
                        output_dtype=SparseType.FP16,
                        device="cpu",
                        embedding_cache_mode=True,
                    )
                    kv.initialize_kv_embedding_cache()
                    _load_kv_weights(reference, kv)
                    kv.weight_initialized = True
                    generator = torch.Generator().manual_seed(0)
                    indices = torch.randint(32, (24,), generator=generator)
                    offsets = torch.arange(0, 25, 3)
                    weights = (
                        torch.rand(24, generator=generator)
                        if pooling_mode == PoolingMode.SUM
                        else None
                    )
                    outputs.append(kv(indices, offsets, weights))
                self.assertTrue(torch.isfinite(outputs[0]).all())
                self.assertTrue(torch.equal(outputs[0], outputs[1]))


def _fill_random_weights(reference: IntNBitTableBatchedEmbeddingBagsCodegen) -> None:
    reference.fill_random_weights()
    for _, scale_bias in reference.split_embedding_weights():
        if scale_bias is not None:
            random_quant_scaled_tensor(
                shape=scale_bias.shape,
                device=reference.current_device,
                output_tensor=scale_bias,
            )


def _fill_finite_weights(reference: IntNBitTableBatchedEmbeddingBagsCodegen) -> None:
    """Seeded random rows with bounded fp16 scales and shifts, so every output
    is finite."""
    torch.manual_seed(0)
    reference.fill_random_weights()
    generator = torch.Generator().manual_seed(0)
    for _, scale_shift in reference.split_embedding_weights():
        if scale_shift is not None:
            rows = scale_shift.shape[0]
            qparams = torch.stack(
                [
                    torch.rand(rows, generator=generator) * 0.01 + 0.001,
                    torch.rand(rows, generator=generator) * 0.2 - 0.1,
                ],
                dim=1,
            ).half()
            scale_shift.copy_(qparams.view(torch.uint8).to(scale_shift.device))


def _load_kv_weights(
    reference: IntNBitTableBatchedEmbeddingBagsCodegen, kv: KVEmbeddingInference
) -> None:
    # CUDA reference rows carry 16-byte alignment padding beyond KV storage.
    for t, (weight, _) in enumerate(
        reference.split_embedding_weights(split_scale_shifts=False)
    ):
        kv.embedding_inplace_update_per_table(
            t, torch.arange(weight.shape[0], dtype=torch.int64), weight.cpu()
        )


def _cuda_kv_with_reference(
    num_tables: int,
    num_embeddings: int,
    dim: int,
    weight_ty: SparseType = SparseType.INT8,
    pooling_mode: PoolingMode = PoolingMode.SUM,
    embedding_cache_mode: bool = True,
    fill_new_embeddings_enabled: bool = False,
) -> tuple[
    IntNBitTableBatchedEmbeddingBagsCodegen, KVZCHGPUBucketWiseEmbeddingInference
]:
    specs = [
        ("", num_embeddings, dim, weight_ty, EmbeddingLocation.DEVICE)
        for _ in range(num_tables)
    ]
    reference = IntNBitTableBatchedEmbeddingBagsCodegen(
        specs, pooling_mode=pooling_mode, output_dtype=SparseType.FP16, device="cuda"
    )
    _fill_random_weights(reference)
    kv = KVZCHGPUBucketWiseEmbeddingInference(
        specs,
        pooling_mode=pooling_mode,
        output_dtype=SparseType.FP16,
        device="cuda",
        embedding_cache_mode=embedding_cache_mode,
        fill_new_embeddings_enabled=fill_new_embeddings_enabled,
    )
    kv.initialize_kv_embedding_cache()
    # As the predictor loader does, plan the shard before loading its weights:
    # only a planned KV-ZCH GPU serving shard takes CUDA-padded rows.
    kv.kv_embedding_cache.set_hbm_cache_rows(0)
    _load_kv_weights(reference, kv)
    kv.weight_initialized = True
    return reference, kv


@skipIf(open_source, "Not supported in open source yet")
@skipIf(not torch.cuda.is_available(), "Requires CUDA")
class KVEmbeddingCudaTest(TestCase):
    num_tables = 4
    num_embeddings = 100
    dim = 128

    def test_int8_unpooled_cache_rows_and_outputs_are_byte_exact(self) -> None:
        reference, kv = _cuda_kv_with_reference(
            1, 128, 128, weight_ty=SparseType.INT8, pooling_mode=PoolingMode.NONE
        )
        kv.kv_embedding_cache.set_hbm_cache_rows(1024)
        kv.kv_embedding_cache.set_hbm_cache_fused_lookup(True)
        kv.kv_embedding_cache.set_hbm_cache_fused_admission(True)
        kv.kv_embedding_cache.set_hbm_cache_direct_output(False)
        keys = torch.cat((torch.arange(64, dtype=torch.int64), torch.tensor([7, 1, 7])))
        device = torch.accelerator.current_accelerator()
        indices = keys.to(device=device, dtype=torch.int32)
        # Citrine C3: allocate offsets on the active device.
        offsets = torch.tensor(
            [0, keys.numel()],
            device=device,
            dtype=torch.int32,
        )
        expected = reference(indices, offsets)
        storage_rows = kv.kv_embedding_cache.get_embeddings(keys)
        self.assertEqual(storage_rows.shape[1], 136)
        expected_rows = torch.zeros(keys.numel(), 144, dtype=torch.uint8)
        expected_rows[:, :136] = storage_rows
        for _ in range(2):
            rows = kv.kv_embedding_cache.get_embeddings(indices)
            self.assertTrue(torch.equal(rows.cpu(), expected_rows))
            output = kv(indices, offsets)
            self.assertTrue(
                torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            )
        hits, misses, admitted, _, _ = kv.kv_embedding_cache.get_hbm_cache_stats()
        self.assertGreater(hits, 0)
        self.assertGreater(misses, 0)
        self.assertGreater(admitted, 0)

    @staticmethod
    def _cache_sets(keys: torch.Tensor, num_sets: int) -> torch.Tensor:
        mixed = (keys ^ (keys >> 31)) * -7046029254386353131
        return (mixed ^ (mixed >> 29)).remainder(num_sets)

    @staticmethod
    def _aten_admit(
        keys: torch.Tensor,
        input_rows: torch.Tensor,
        tags: torch.Tensor,
        cache_rows: torch.Tensor,
        lru: torch.Tensor,
        tick: int,
    ) -> tuple[int, int]:
        assoc = tags.shape[1]
        sets = KVEmbeddingCudaTest._cache_sets(keys, tags.shape[0])
        # Key -1 is the empty-way tag: it never hits and is never admitted.
        cacheable = keys.ne(-1)
        hit = tags.index_select(0, sets).eq(keys.unsqueeze(1)).any(1) & cacheable
        absent = (hit.logical_not() & cacheable).nonzero().squeeze(1)
        if absent.numel() == 0:
            return 0, 0
        sorted_sets, order = sets.index_select(0, absent).sort()
        rank = torch.arange(
            sorted_sets.numel(), device=keys.device
        ) - torch.searchsorted(sorted_sets, sorted_sets)
        admitted = rank.lt(assoc).nonzero().squeeze(1)
        source = absent.index_select(0, order.index_select(0, admitted))
        sets = sorted_sets.index_select(0, admitted)
        ways = (
            lru.index_select(0, sets)
            .argsort(stable=False, dim=1, descending=False)
            .gather(1, rank.index_select(0, admitted).unsqueeze(1))
            .squeeze(1)
        )
        slots = sets * assoc + ways
        cache_rows.index_copy_(0, slots, input_rows.index_select(0, source))
        tags.view(-1).index_copy_(0, slots, keys.index_select(0, source))
        lru.view(-1).index_fill_(0, slots, tick)
        return admitted.numel(), absent.numel() - admitted.numel()

    def _assert_admission_matches_aten(
        self,
        keys: torch.Tensor,
        input_rows: torch.Tensor,
        tags: torch.Tensor,
        cache_rows: torch.Tensor,
        lru: torch.Tensor,
        generation: int = 1,
        fallback: tuple[bool, bool, bool, bool] = (False, False, False, False),
        scratch: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            | None
        ) = None,
    ) -> None:
        before = (tags.clone(), cache_rows.clone(), lru.clone())
        expected = (tags.clone(), cache_rows.clone(), lru.clone())
        expected_counts = self._aten_admit(keys, input_rows, *expected, tick=97)
        if scratch is None:
            scratch = (
                torch.zeros(tags.shape[0], dtype=torch.int64, device=keys.device),
                torch.empty(tags.shape[0], dtype=torch.int32, device=keys.device),
                torch.empty(tags.shape[0], dtype=torch.int32, device=keys.device),
                torch.empty(keys.numel(), dtype=torch.int32, device=keys.device),
                torch.empty(keys.numel(), dtype=torch.int32, device=keys.device),
            )

        result = torch.ops.fbgemm.kv_hbm_cache_admit(
            keys, input_rows, tags, cache_rows, lru, *scratch, 97, generation
        )

        self.assertEqual(tuple(result[2:]), fallback)
        if any(fallback):
            for actual, original in zip((tags, cache_rows, lru), before):
                self.assertTrue(torch.equal(actual, original))
            self.assertEqual(result[:2], (0, 0))
            counts = self._aten_admit(keys, input_rows, tags, cache_rows, lru, 97)
        else:
            counts = result[:2]
        self.assertEqual(counts, expected_counts)
        for actual, reference in zip((tags, cache_rows, lru), expected):
            self.assertTrue(torch.equal(actual, reference))
        if tags.shape[1] > 32:
            return
        sets = self._cache_sets(keys, tags.shape[0])
        matches = expected[0][sets].eq(keys.unsqueeze(1))
        expected_hits = matches.any(1)
        slots = sets * tags.shape[1] + matches.byte().argmax(1)
        rows, hits = torch.ops.fbgemm.kv_hbm_cache_lookup(
            keys, tags, cache_rows, lru.clone(), 98
        )
        self.assertTrue(torch.equal(hits, expected_hits))
        self.assertTrue(torch.equal(rows[hits], expected[1][slots[expected_hits]]))

    def test_fused_admission_matches_aten_eviction_and_full_rows(self) -> None:
        device = torch.accelerator.current_accelerator()
        for num_keys in (33, 64, 128, 257, 1024, 4096, 4097):
            num_sets = 17 if num_keys == 1024 else 257
            for assoc in (1, 4, 7, 32):
                for row_bytes, storage_offset in ((73, 1), (80, 0), (144, 1)):
                    for tied_lru in (False, True):
                        with self.subTest(
                            num_keys=num_keys,
                            assoc=assoc,
                            row_bytes=row_bytes,
                            storage_offset=storage_offset,
                            tied_lru=tied_lru,
                        ):
                            keys = (
                                torch.arange(num_keys, device=device).long() * 104729
                                - 137
                            )
                            keys[0] = -(1 << 63)
                            keys[-1] = (1 << 63) - 1
                            keys = keys.flip(0)
                            tags = (
                                torch.arange(num_sets * assoc, device=device).long()
                                + 1_000_000_000
                            ).view(num_sets, assoc)
                            cache_rows = (
                                torch.arange(
                                    num_sets * assoc * row_bytes, device=device
                                )
                                .byte()
                                .view(-1, row_bytes)
                            )
                            storage = torch.arange(
                                num_keys * row_bytes + storage_offset, device=device
                            ).byte()
                            input_rows = storage.narrow(
                                0, storage_offset, num_keys * row_bytes
                            ).view(num_keys, row_bytes)
                            lru = (
                                tags.remainder(11) if tied_lru else tags.flip(1).clone()
                            )
                            self._assert_admission_matches_aten(
                                keys, input_rows, tags, cache_rows, lru
                            )

    @staticmethod
    def _admission_scratch(
        keys: torch.Tensor, num_sets: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(num_sets, dtype=torch.int64, device=keys.device),
            torch.empty(num_sets, dtype=torch.int32, device=keys.device),
            torch.empty(num_sets, dtype=torch.int32, device=keys.device),
            torch.empty(keys.numel(), dtype=torch.int32, device=keys.device),
            torch.empty(keys.numel(), dtype=torch.int32, device=keys.device),
        )

    def _assert_direct_matches_fused(
        self,
        keys: torch.Tensor,
        tags: torch.Tensor,
        cache_rows: torch.Tensor,
        lru: torch.Tensor,
        generation: int = 1,
        readmitted: int = 0,
        storage_offset: int = 0,
        fallback: tuple[bool, bool, bool, bool] | None = None,
    ) -> None:
        baseline = (tags.clone(), cache_rows.clone(), lru.clone())
        unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        unique_rows, hit = torch.ops.fbgemm.kv_hbm_cache_lookup(unique, *baseline, 97)
        miss_positions = hit.logical_not().nonzero().squeeze(1)
        miss_keys = unique.index_select(0, miss_positions)
        packed = (
            torch.empty(
                miss_keys.numel() * 72 + storage_offset,
                dtype=torch.uint8,
                device=keys.device,
            )
            .narrow(0, storage_offset, miss_keys.numel() * 72)
            .view(-1, 72)
        )
        packed.copy_(
            (
                miss_keys.remainder(251).unsqueeze(1)
                + torch.arange(72, device=keys.device)
            ).byte()
        )
        padded = torch.zeros(
            (miss_keys.numel(), 80), dtype=torch.uint8, device=keys.device
        )
        padded[:, :72] = packed
        unique_rows.index_copy_(0, miss_positions, padded)
        expected_output = unique_rows.index_select(0, inverse)

        output, direct_misses, slots, rank, hits = (
            torch.ops.fbgemm.kv_hbm_cache_lookup_direct(keys, tags, cache_rows, lru, 97)
        )
        self.assertEqual(hits, int(hit.sum()))
        self.assertTrue(torch.equal(direct_misses, miss_keys))
        self.assertTrue(torch.equal(lru, baseline[2]))
        tick = 97
        if readmitted:
            tick = 98
            other_rows = padded[:readmitted].bitwise_xor(13)
            for state in (baseline, (tags, cache_rows, lru)):
                self._aten_admit(miss_keys[:readmitted], other_rows, *state, tick)
        before = (tags.clone(), cache_rows.clone(), lru.clone())
        expected_state = (baseline[0].clone(), baseline[1].clone(), baseline[2].clone())
        expected_counts = self._aten_admit(miss_keys, padded, *expected_state, tick)
        baseline_result = torch.ops.fbgemm.kv_hbm_cache_admit(
            miss_keys,
            padded,
            *baseline,
            *self._admission_scratch(miss_keys, tags.shape[0]),
            tick,
            generation,
        )
        direct_result = torch.ops.fbgemm.kv_hbm_cache_admit_direct(
            direct_misses,
            packed,
            tags,
            cache_rows,
            lru,
            *self._admission_scratch(miss_keys, tags.shape[0]),
            slots,
            rank,
            output,
            tick,
            generation,
        )
        self.assertEqual(direct_result, baseline_result)
        if fallback is not None:
            self.assertEqual(tuple(direct_result[2:]), fallback)
        if any(direct_result[2:]):
            for actual, original in zip((tags, cache_rows, lru), before):
                self.assertTrue(torch.equal(actual, original))
            actual_padding = torch.ops.fbgemm.kv_hbm_cache_pad_output(
                packed, slots, rank, output
            )
            self.assertTrue(torch.equal(actual_padding, padded))
            direct_counts = self._aten_admit(
                miss_keys, actual_padding, tags, cache_rows, lru, tick
            )
            self._aten_admit(miss_keys, padded, *baseline, tick)
        else:
            direct_counts = direct_result[:2]
        self.assertEqual(direct_counts, expected_counts)
        self.assertTrue(torch.equal(output, expected_output))
        for actual, reference, fused in zip(
            (tags, cache_rows, lru), expected_state, baseline
        ):
            self.assertTrue(torch.equal(actual, reference))
            self.assertTrue(torch.equal(actual, fused))
        cache_rows.fill_(255)
        self.assertTrue(torch.equal(output, expected_output))

    def test_direct_output_matches_fused_cache_and_signed_miss_order(self) -> None:
        device = torch.accelerator.current_accelerator()
        for num_keys in (0, 1, 32, 33, 129, 257, 4097):
            for assoc in (1, 7, 32):
                for seed_fraction in (0, 2):
                    with self.subTest(
                        num_keys=num_keys, assoc=assoc, seed_fraction=seed_fraction
                    ):
                        keys = torch.arange(num_keys, dtype=torch.int64, device=device)
                        keys = keys * 104729 - 137
                        if num_keys >= 3:
                            keys[0] = -(1 << 63)
                            keys[1] = (1 << 63) - 1
                            keys[2] = -1
                        tags = torch.full(
                            (257, assoc), -1, dtype=torch.int64, device=device
                        )
                        rows = (
                            torch.arange(257 * assoc * 80, device=device)
                            .byte()
                            .view(-1, 80)
                        )
                        lru = tags.remainder(11)
                        if seed_fraction:
                            seed = keys[::seed_fraction].sort()[0]
                            seed_rows = (
                                torch.arange(seed.numel() * 80, device=device)
                                .byte()
                                .view(-1, 80)
                            )
                            self._aten_admit(seed, seed_rows, tags, rows, lru, 11)
                        keys = torch.cat((keys.flip(0), keys[::3], keys[:2]))
                        self._assert_direct_matches_fused(
                            keys, tags, rows, lru, storage_offset=num_keys % 2
                        )

    def test_direct_output_fallback_and_interleaved_admission(self) -> None:
        device = torch.accelerator.current_accelerator()
        cases = (
            (32, 17, 1, 0, (True, False, False, False)),
            (64, 17, 1, 32, (True, False, False, False)),
            (64, 17, 1, 64, (False, False, False, False)),
            (129, 1, 1, 0, (False, True, False, False)),
            (64, 17, 0, 0, (False, False, True, False)),
            (64, 17, (1 << 63) - 1, 0, (False, False, True, False)),
        )
        for num_keys, num_sets, generation, readmitted, fallback in cases:
            with self.subTest(
                num_keys=num_keys,
                num_sets=num_sets,
                generation=generation,
                readmitted=readmitted,
            ):
                keys = torch.arange(num_keys, device=device).long() * 104729 - 137
                keys = torch.cat((keys.flip(0), keys[::3]))
                tags = torch.full((num_sets, 32), -1, dtype=torch.int64, device=device)
                rows = (
                    torch.arange(num_sets * 32 * 80, device=device).byte().view(-1, 80)
                )
                self._assert_direct_matches_fused(
                    keys,
                    tags,
                    rows,
                    torch.zeros_like(tags),
                    generation,
                    readmitted,
                    storage_offset=1,
                    fallback=fallback,
                )

    def test_direct_output_large_all_hit_and_preflight(self) -> None:
        device = torch.accelerator.current_accelerator()
        for num_keys in (7522, 175808, 304306, 512 * 1024):
            with self.subTest(num_keys=num_keys):
                tags = torch.full((17, 32), -1, dtype=torch.int64, device=device)
                rows = torch.arange(17 * 32 * 80, device=device).byte().view(-1, 80)
                seed = torch.arange(20, device=device).long() * 104729 + 13
                self._aten_admit(
                    seed, rows[:20].clone(), tags, rows, torch.zeros_like(tags), 11
                )
                keys = seed[
                    torch.arange(num_keys, device=device).remainder(seed.numel())
                ]
                self._assert_direct_matches_fused(
                    keys, tags, rows, torch.zeros_like(tags)
                )
        tags = torch.full((17, 32), -1, dtype=torch.int64, device=device)
        rows = torch.zeros((17 * 32, 80), dtype=torch.uint8, device=device)
        lru = torch.zeros_like(tags)
        with self.assertRaisesRegex(RuntimeError, "at most 512K"):
            torch.ops.fbgemm.kv_hbm_cache_lookup_direct(
                torch.zeros(512 * 1024 + 1, dtype=torch.int64, device=device),
                tags,
                rows,
                lru,
                97,
            )
        with self.assertRaisesRegex(RuntimeError, "80-byte rows"):
            torch.ops.fbgemm.kv_hbm_cache_lookup_direct(
                torch.arange(64, device=device),
                tags,
                torch.zeros((17 * 32, 144), dtype=torch.uint8, device=device),
                lru,
                97,
            )
        self.assertTrue(tags.eq(-1).all())
        self.assertFalse(rows.any())
        self.assertFalse(lru.any())

    def test_fused_admission_fallbacks_leave_cache_unchanged(self) -> None:
        device = torch.accelerator.current_accelerator()
        cases = (
            (0, 17, 32, 1, 0, (False, False, False, False)),
            (32, 17, 32, 1, 0, (True, False, False, False)),
            (64, 17, 32, 1, 32, (True, False, False, False)),
            (64, 17, 32, 1, 64, (False, False, False, False)),
            (33, 1, 32, 1, 0, (False, False, False, False)),
            (128, 1, 32, 1, 0, (False, False, False, False)),
            (129, 1, 32, 1, 0, (False, True, False, False)),
            (64, 17, 32, 0, 0, (False, False, True, False)),
            (64, 17, 32, (1 << 63) - 1, 0, (False, False, True, False)),
            (64, 17, 64, 1, 0, (False, False, False, True)),
        )
        for num_keys, num_sets, assoc, generation, num_hits, fallback in cases:
            with self.subTest(
                num_keys=num_keys,
                num_sets=num_sets,
                assoc=assoc,
                generation=generation,
                num_hits=num_hits,
            ):
                keys = torch.arange(num_keys, device=device).long() * 104729 - 137
                tags = torch.full(
                    (num_sets, assoc), -1, dtype=torch.int64, device=device
                )
                sets = self._cache_sets(keys, num_sets).cpu()
                counts = [0] * num_sets
                for index in range(num_hits):
                    cache_set = int(sets[index])
                    tags[cache_set, counts[cache_set]] = keys[index]
                    counts[cache_set] += 1
                cache_rows = (
                    torch.arange(num_sets * assoc * 80, device=device)
                    .byte()
                    .view(-1, 80)
                )
                input_rows = (
                    torch.arange(num_keys * 80, device=device).byte().view(num_keys, 80)
                )
                lru = torch.zeros_like(tags)
                self._assert_admission_matches_aten(
                    keys, input_rows, tags, cache_rows, lru, generation, fallback
                )

    def test_fused_admission_reuses_scratch_across_generations(self) -> None:
        device = torch.accelerator.current_accelerator()
        tags = torch.full((3, 32), -1, dtype=torch.int64, device=device)
        cache_rows = torch.zeros((96, 80), dtype=torch.uint8, device=device)
        lru = torch.zeros_like(tags)
        scratch = (
            torch.zeros(3, dtype=torch.int64, device=device),
            torch.empty(3, dtype=torch.int32, device=device),
            torch.empty(3, dtype=torch.int32, device=device),
            torch.empty(64, dtype=torch.int32, device=device),
            torch.empty(64, dtype=torch.int32, device=device),
        )
        for generation in (1, 2, 3):
            keys = torch.arange(64, device=device).long() + generation * 1000
            input_rows = (
                (torch.arange(64 * 80, device=device) + generation).byte().view(64, 80)
            )
            self._assert_admission_matches_aten(
                keys, input_rows, tags, cache_rows, lru, generation, scratch=scratch
            )

    def test_fused_cache_flags_preserve_padded_rows_and_eviction_results(self) -> None:
        reference, _ = _cuda_kv_with_reference(
            self.num_tables, self.num_embeddings, self.dim
        )
        baseline_rows: list[torch.Tensor] = []
        baseline_stats: list[list[int]] = []
        requests = (
            torch.arange(16),
            torch.arange(400),
            torch.arange(400).flip(0),
            torch.arange(0, 400, 3).repeat(2),
            torch.arange(80, 320),
        )
        for fused_lookup, fused_admission in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                fused_lookup=fused_lookup, fused_admission=fused_admission
            ):
                _, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                _load_kv_weights(reference, kv)
                cache = kv.kv_embedding_cache
                cache.set_hbm_cache_rows(256)
                cache.set_hbm_cache_fused_lookup(fused_lookup)
                cache.set_hbm_cache_fused_admission(fused_admission)
                for index, keys in enumerate(requests):
                    rows = cache.get_embeddings(
                        keys.to(torch.accelerator.current_accelerator())
                    )
                    self._assert_encoded_rows_match_cpu(kv, keys, rows)
                    stats = cache.get_hbm_cache_stats()
                    if not fused_lookup and not fused_admission:
                        baseline_rows.append(rows.cpu())
                        baseline_stats.append(stats)
                    else:
                        self.assertTrue(torch.equal(rows.cpu(), baseline_rows[index]))
                        self.assertEqual(stats, baseline_stats[index])
                attempts, fused, admitted, small, overflow, generation, unsupported = (
                    cache.get_hbm_cache_fused_admission_stats()
                )
                if fused_admission:
                    self.assertGreater(attempts, 0)
                    self.assertGreater(fused, 0)
                    self.assertGreater(admitted, 0)
                    self.assertGreater(small, 0)
                    self.assertEqual(overflow + generation + unsupported, 0)
                    self.assertEqual(attempts, fused + small)
                else:
                    self.assertEqual(
                        cache.get_hbm_cache_fused_admission_stats(), [0] * 7
                    )

    def test_fused_cache_orders_different_cuda_streams(self) -> None:
        device = torch.accelerator.current_accelerator()
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                _, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                cache = kv.kv_embedding_cache
                cache.set_hbm_cache_rows(512)
                cache.set_hbm_cache_fused_lookup(enabled)
                cache.set_hbm_cache_fused_admission(enabled)
                streams = (torch.cuda.Stream(), torch.cuda.Stream())
                for index in range(4):
                    with torch.cuda.stream(streams[index % 2]):
                        keys = torch.arange(64, dtype=torch.int64, device=device)
                        rows = cache.get_embeddings(keys)
                        self._assert_encoded_rows_match_cpu(kv, keys, rows)
                hits, misses, admitted, rejected, _ = cache.get_hbm_cache_stats()
                self.assertEqual((hits, misses, admitted, rejected), (192, 64, 64, 0))
                if enabled:
                    self.assertEqual(
                        cache.get_hbm_cache_fused_admission_stats()[:3], [1, 1, 64]
                    )

    def test_direct_output_wrapper_flags_match_fused_rows_and_stats(self) -> None:
        reference, _ = _cuda_kv_with_reference(1, 400, 128, SparseType.INT4)
        requests = (
            torch.arange(16),
            torch.arange(128),
            torch.arange(400).flip(0),
            torch.arange(0, 400, 3).repeat(2),
            torch.arange(80, 320),
            torch.empty(0, dtype=torch.int64),
        )
        for fused_lookup, fused_admission in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            baseline_rows: list[torch.Tensor] = []
            baseline_stats: list[list[int]] = []
            baseline_fused_stats: list[list[int]] = []
            for direct in (False, True):
                with self.subTest(
                    fused_lookup=fused_lookup,
                    fused_admission=fused_admission,
                    direct=direct,
                ):
                    _, kv = _cuda_kv_with_reference(1, 400, 128, SparseType.INT4)
                    _load_kv_weights(reference, kv)
                    cache = kv.kv_embedding_cache
                    cache.set_hbm_cache_rows(256)
                    cache.set_hbm_cache_fused_lookup(fused_lookup)
                    cache.set_hbm_cache_fused_admission(fused_admission)
                    cache.set_hbm_cache_direct_output(direct)
                    for index, keys in enumerate(requests):
                        rows = cache.get_embeddings(
                            keys.to(torch.accelerator.current_accelerator())
                        )
                        self._assert_encoded_rows_match_cpu(kv, keys, rows)
                        stats = cache.get_hbm_cache_stats()
                        fused_stats = cache.get_hbm_cache_fused_admission_stats()
                        if not direct:
                            baseline_rows.append(rows.cpu())
                            baseline_stats.append(stats)
                            baseline_fused_stats.append(fused_stats)
                        else:
                            self.assertTrue(
                                torch.equal(rows.cpu(), baseline_rows[index])
                            )
                            self.assertEqual(stats, baseline_stats[index])
                            self.assertEqual(fused_stats, baseline_fused_stats[index])
                    calls, sorted_misses, unsupported, capacity = (
                        cache.get_hbm_cache_direct_output_stats()
                    )
                    if direct and fused_lookup and fused_admission:
                        self.assertEqual(calls, len(requests) - 1)
                        self.assertEqual(sorted_misses, cache.get_hbm_cache_stats()[1])
                        self.assertEqual(unsupported, 1)
                    else:
                        self.assertEqual(calls + sorted_misses, 0)
                    self.assertEqual(capacity, 0)

    @skipIf(torch.cuda.device_count() < 2, "Requires two CUDA devices")
    def test_direct_output_shared_backend_streams_and_invalidation(self) -> None:
        print(
            "KV HBM direct-output CUDA devices:",
            {index: torch.cuda.get_device_name(index) for index in range(2)},
        )
        reference, primary = _cuda_kv_with_reference(1, 100, 128, SparseType.INT4)
        held_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        accelerator = torch.accelerator.current_accelerator()
        assert accelerator is not None
        for card in range(2):
            device = torch.device(accelerator.type, card)
            with torch.cuda.device(device):
                bindings: list[KVZCHGPUBucketWiseEmbeddingInference] = []
                for _ in range(2):
                    replica = KVZCHGPUBucketWiseEmbeddingInference(
                        [("", 100, 128, SparseType.INT4, EmbeddingLocation.DEVICE)],
                        output_dtype=SparseType.FP16,
                        device=str(device),
                    )
                    replica.set_device(str(device))
                    replica.initialize_kv_embedding_cache()
                    replica.kv_embedding_cache.share_kv_backend(
                        primary.kv_embedding_cache
                    )
                    replica.weight_initialized = True
                    cache = replica.kv_embedding_cache
                    cache.set_hbm_cache_rows(512)
                    cache.set_hbm_cache_fused_lookup(True)
                    cache.set_hbm_cache_fused_admission(True)
                    cache.set_hbm_cache_direct_output(True)
                    bindings.append(replica)
                streams = (torch.cuda.Stream(), torch.cuda.Stream())
                keys = torch.arange(64, dtype=torch.int64, device=device).repeat(2)
                for index in range(4):
                    with torch.cuda.stream(streams[index % 2]):
                        replica = bindings[index % 2]
                        rows = replica.kv_embedding_cache.get_embeddings(keys)
                        self._assert_encoded_rows_match_cpu(replica, keys, rows)
                        held_rows.append((rows, rows.clone()))
                cache = bindings[0].kv_embedding_cache
                self.assertEqual(cache.get_hbm_cache_stats()[:4], [192, 64, 64, 0])
                self.assertEqual(
                    cache.get_hbm_cache_fused_admission_stats()[:3], [1, 1, 64]
                )
                self.assertEqual(
                    cache.get_hbm_cache_direct_output_stats(), [4, 64, 0, 0]
                )
                weights = reference.split_embedding_weights(split_scale_shifts=False)[
                    0
                ][0]
                update = weights[:1].cpu().clone()
                update[0, 0] ^= 1
                primary.embedding_inplace_update_per_table(0, torch.tensor([0]), update)
                self.assertGreater(cache.get_hbm_cache_stats()[4], 0)
                for replica in bindings:
                    rows = replica.kv_embedding_cache.get_embeddings(keys[:1])
                    self._assert_encoded_rows_match_cpu(replica, keys[:1], rows)
                torch.cuda.synchronize(device)
        for actual, original in held_rows:
            self.assertTrue(torch.equal(actual, original))

    def test_fused_lookup_matches_aten_probe(self) -> None:
        # Citrine C7: use .to(device) instead of deprecated .cuda()
        device = torch.accelerator.current_accelerator()
        keys = torch.tensor([-(1 << 63), -1, 0, 7, (1 << 63) - 1], dtype=torch.int64)
        mixed = (keys ^ (keys >> 31)) * -7046029254386353131
        mixed ^= mixed >> 29
        sets = mixed.remainder(17)
        for assoc in (1, 4, 32):
            for row_bytes in (16, 73, 80, 144):
                for storage_offset in (0, 1):
                    with self.subTest(
                        assoc=assoc,
                        row_bytes=row_bytes,
                        storage_offset=storage_offset,
                    ):
                        tags_cpu = torch.full((17, assoc), -1, dtype=torch.int64)
                        rows_cpu = torch.zeros(
                            (17 * assoc, row_bytes), dtype=torch.uint8
                        )
                        for index in (0, 2, 3):
                            cache_set = int(sets[index])
                            way = index % assoc
                            tags_cpu[cache_set, way] = keys[index]
                            rows_cpu[cache_set * assoc + way] = (
                                torch.arange(row_bytes) + index
                            ).byte()
                        matches = tags_cpu[sets].eq(keys.unsqueeze(1))
                        expected_hits = matches.any(1) & keys.ne(-1)
                        slots = sets * assoc + matches.long().argmax(1)
                        tags = tags_cpu.to(device)
                        row_storage = torch.empty(
                            rows_cpu.numel() + storage_offset,
                            dtype=torch.uint8,
                            device=device,
                        )
                        rows = row_storage.narrow(
                            0, storage_offset, rows_cpu.numel()
                        ).view_as(rows_cpu)
                        rows.copy_(rows_cpu)
                        lru = torch.zeros_like(tags)
                        empty_rows, empty_hits = torch.ops.fbgemm.kv_hbm_cache_lookup(
                            keys[:0].to(device), tags, rows, lru, 17
                        )
                        self.assertEqual(empty_rows.shape, (0, row_bytes))
                        self.assertEqual(empty_hits.numel(), 0)
                        self.assertFalse(lru.any())

                        actual_rows, actual_hits = torch.ops.fbgemm.kv_hbm_cache_lookup(
                            keys.to(device), tags, rows, lru, 17
                        )
                        torch.testing.assert_close(actual_hits.cpu(), expected_hits)
                        torch.testing.assert_close(
                            actual_rows[actual_hits].cpu(),
                            rows_cpu[slots[expected_hits]],
                            rtol=0,
                            atol=0,
                        )
                        expected_lru = torch.zeros_like(tags_cpu)
                        expected_lru.view(-1).index_fill_(0, slots[expected_hits], 17)
                        torch.testing.assert_close(lru.cpu(), expected_lru)
                        torch.testing.assert_close(tags.cpu(), tags_cpu)
                        for selected_hits in (False, True):
                            selected = expected_hits.eq(selected_hits)
                            lru.zero_()
                            selected_rows, selected_mask = (
                                torch.ops.fbgemm.kv_hbm_cache_lookup(
                                    keys[selected].to(device), tags, rows, lru, 19
                                )
                            )
                            self.assertGreater(selected_mask.numel(), 0)
                            self.assertTrue(selected_mask.eq(selected_hits).all())
                            expected_lru.zero_()
                            if selected_hits:
                                self.assertTrue(
                                    torch.equal(
                                        selected_rows.cpu(), rows_cpu[slots[selected]]
                                    )
                                )
                                expected_lru.view(-1).index_fill_(
                                    0, slots[selected], 19
                                )
                            self.assertTrue(torch.equal(lru.cpu(), expected_lru))

    def _assert_matches_reference(
        self,
        reference: IntNBitTableBatchedEmbeddingBagsCodegen,
        kv: KVZCHGPUBucketWiseEmbeddingInference,
        num_requests: int,
    ) -> None:
        for req in generate_requests(
            num_requests,
            8,
            self.num_tables,
            4,
            self.num_embeddings,
            use_cpu=True,
        ):
            indices = req.indices.int().to(torch.accelerator.current_accelerator())
            offsets = req.offsets.int().to(torch.accelerator.current_accelerator())
            torch.testing.assert_close(
                kv(indices, offsets), reference(indices, offsets), equal_nan=True
            )

    def test_cuda_rows_are_aligned_and_match_cpu_rows(self) -> None:
        reference, _ = _cuda_kv_with_reference(
            self.num_tables, self.num_embeddings, self.dim
        )
        for fused_lookup in (False, True):
            with self.subTest(fused_lookup=fused_lookup):
                _, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                _load_kv_weights(reference, kv)
                keys = torch.tensor([5, 150, 5, 399], dtype=torch.int64)
                for rows in (0, 64):
                    kv.kv_embedding_cache.set_hbm_cache_rows(rows)
                    kv.kv_embedding_cache.set_hbm_cache_fused_lookup(fused_lookup)
                    kv.kv_embedding_cache.set_hbm_cache_fused_admission(fused_lookup)
                    for _ in range(2):
                        cuda_rows = kv.kv_embedding_cache.get_embeddings(
                            keys.to(torch.accelerator.current_accelerator())
                        )
                        self._assert_encoded_rows_match_cpu(kv, keys, cuda_rows)
                    before = kv.kv_embedding_cache.get_hbm_cache_stats()
                    empty = kv.kv_embedding_cache.get_embeddings(
                        keys[:0].to(torch.accelerator.current_accelerator())
                    )
                    self.assertEqual(empty.shape, (0, cuda_rows.shape[1]))
                    self.assertEqual(
                        kv.kv_embedding_cache.get_hbm_cache_stats(), before
                    )

    def _assert_encoded_rows_match_cpu(
        self,
        kv: KVZCHGPUBucketWiseEmbeddingInference,
        keys: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        cpu_rows = kv.kv_embedding_cache.get_embeddings(keys.cpu())
        expected = torch.zeros(rows.shape, dtype=torch.uint8)
        expected[:, : cpu_rows.shape[1]] = cpu_rows
        self.assertEqual(rows.shape[1] % 16, 0)
        self.assertGreater(rows.shape[1], cpu_rows.shape[1])
        self.assertTrue(torch.equal(rows.cpu(), expected))

    def test_forward_without_hbm_cache(self) -> None:
        reference, kv = _cuda_kv_with_reference(
            self.num_tables, self.num_embeddings, self.dim
        )
        kv.kv_embedding_cache.set_hbm_cache_rows(0)

        self._assert_matches_reference(reference, kv, num_requests=2)

        self.assertEqual(kv.kv_embedding_cache.get_hbm_cache_stats(), [0] * 5)

    def test_forward_with_hbm_cache_hits_and_evictions(self) -> None:
        for fused_lookup in (False, True):
            with self.subTest(fused_lookup=fused_lookup):
                reference, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                # Two 32-way sets hold far fewer than the 400 rows, forcing both
                # evictions and rejections within one batch.
                kv.kv_embedding_cache.set_hbm_cache_rows(64)
                kv.kv_embedding_cache.set_hbm_cache_fused_lookup(fused_lookup)
                kv.kv_embedding_cache.set_hbm_cache_fused_admission(fused_lookup)

                self._assert_matches_reference(reference, kv, num_requests=6)

                hits, misses, admitted, rejected, _ = (
                    kv.kv_embedding_cache.get_hbm_cache_stats()
                )
                self.assertGreater(hits, 0)
                self.assertGreater(misses, 0)
                self.assertGreater(admitted, 0)
                self.assertGreater(rejected, 0)

    def test_random_init_backend_bypasses_hbm_cache(self) -> None:
        reference, kv = _cuda_kv_with_reference(
            self.num_tables, self.num_embeddings, self.dim, embedding_cache_mode=False
        )
        kv.kv_embedding_cache.set_hbm_cache_rows(64)

        self._assert_matches_reference(reference, kv, num_requests=2)

        self.assertEqual(kv.kv_embedding_cache.get_hbm_cache_stats(), [0] * 5)
        bypassed, _ = kv.kv_embedding_cache.get_hbm_cache_admission_policy_stats()
        self.assertGreater(bypassed, 0)

    def test_module_delta_update_replaces_cached_rows(self) -> None:
        for fused in (False, True):
            with self.subTest(fused=fused):
                _, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                cache = kv.kv_embedding_cache
                cache.set_hbm_cache_rows(1024)
                cache.set_hbm_cache_fused_lookup(fused)
                cache.set_hbm_cache_fused_admission(fused)
                ids = torch.arange(40, dtype=torch.int64)
                keys = ids.to(torch.accelerator.current_accelerator())
                for _ in range(2):
                    cache.get_embeddings(keys)
                updated = (cache.get_embeddings(ids).to(torch.int16) + 1).to(
                    torch.uint8
                )

                kv.embedding_inplace_update_per_table(0, ids, updated)
                rows = cache.get_embeddings(keys)

                self.assertTrue(torch.equal(cache.get_embeddings(ids), updated))
                self._assert_encoded_rows_match_cpu(kv, ids, rows)
                hits, _, _, _, invalidated = cache.get_hbm_cache_stats()
                self.assertGreater(invalidated, 0)
                cache.get_embeddings(keys)
                self.assertEqual(cache.get_hbm_cache_stats()[0], hits + 40)

    def test_filled_id_is_served_after_an_absent_lookup(self) -> None:
        # A lookup of an absent id must not leave anything in HBM that would
        # shadow the row a later fill-only delta writes to the store.
        for fused in (False, True):
            with self.subTest(fused=fused):
                reference, kv = _cuda_kv_with_reference(
                    1, 64, self.dim, fill_new_embeddings_enabled=True
                )
                absent_id = 40
                rows = reference.split_embedding_weights(split_scale_shifts=False)[0][
                    0
                ].cpu()
                # The snapshot did not carry absent_id.
                kv.kv_embedding_cache.set_embeddings(
                    torch.tensor([absent_id]), torch.zeros_like(rows[:1])
                )
                cache = kv.kv_embedding_cache
                cache.set_hbm_cache_rows(1024)
                cache.set_hbm_cache_fused_lookup(fused)
                cache.set_hbm_cache_fused_admission(fused)
                key = torch.tensor([absent_id]).to(
                    torch.accelerator.current_accelerator()
                )
                self.assertTrue(torch.all(cache.get_embeddings(key) == 0))

                kv.embedding_fill_new_rows_per_table(
                    0, torch.tensor([absent_id]), rows[absent_id : absent_id + 1]
                )

                self.assertEqual(kv.kv_embedding_cache.get_fill_stats()[1], 0)
                served = cache.get_embeddings(key).cpu()
                self.assertTrue(
                    torch.equal(
                        served[:, : rows.size(1)], rows[absent_id : absent_id + 1]
                    )
                )

    def test_absent_ids_are_not_admitted(self) -> None:
        for fused in (False, True):
            with self.subTest(fused=fused):
                _, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                cache = kv.kv_embedding_cache
                cache.set_hbm_cache_rows(1024)
                cache.set_hbm_cache_fused_lookup(fused)
                cache.set_hbm_cache_fused_admission(fused)
                present = torch.arange(40, dtype=torch.int64)
                absent = torch.arange(10**6, 10**6 + 40, dtype=torch.int64)
                keys = torch.cat([present, absent]).to(
                    torch.accelerator.current_accelerator()
                )

                for _ in range(2):
                    rows = cache.get_embeddings(keys)
                    self.assertTrue(torch.all(rows[40:] == 0))
                    self._assert_encoded_rows_match_cpu(kv, keys.cpu(), rows)

                hits, misses, admitted, _, _ = cache.get_hbm_cache_stats()
                self.assertEqual((hits, misses, admitted), (40, 120, 40))
                self.assertEqual(cache.get_hbm_cache_admission_policy_stats(), [0, 80])

    def test_fused_lookup_preserves_lru_eviction_order(self) -> None:
        cases = (
            (1, 1, False),
            (1, 2, True),
            (2, 1, False),
            (2, 2, False),
            (2, 0, True),
            (2, 32, True),
            (2, 33, True),
        )
        for fused_lookup in (False, True):
            for num_evictions, probe_key, expected_hit in cases:
                with self.subTest(
                    fused_lookup=fused_lookup,
                    num_evictions=num_evictions,
                    probe_key=probe_key,
                ):
                    _, kv = _cuda_kv_with_reference(1, 64, 64)
                    cache = kv.kv_embedding_cache
                    cache.set_hbm_cache_rows(32)
                    cache.set_hbm_cache_fused_lookup(fused_lookup)
                    cache.set_hbm_cache_fused_admission(fused_lookup)
                    device = torch.accelerator.current_accelerator()
                    cache.get_embeddings(torch.arange(32, device=device))
                    for key in (*range(32), 0, *range(32, 32 + num_evictions)):
                        cache.get_embeddings(
                            torch.tensor([key], dtype=torch.int64, device=device)
                        )
                    before = cache.get_hbm_cache_stats()
                    keys = torch.tensor([probe_key], dtype=torch.int64, device=device)

                    rows = cache.get_embeddings(keys)

                    after = cache.get_hbm_cache_stats()
                    self.assertEqual(after[0] - before[0], int(expected_hit))
                    self.assertEqual(after[1] - before[1], int(not expected_hit))
                    self._assert_encoded_rows_match_cpu(kv, keys, rows)

    def test_forward_with_hbm_cache_holding_all_rows(self) -> None:
        for fused_lookup in (False, True):
            with self.subTest(fused_lookup=fused_lookup):
                reference, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                kv.kv_embedding_cache.set_hbm_cache_rows(4096)
                kv.kv_embedding_cache.set_hbm_cache_fused_lookup(fused_lookup)
                kv.kv_embedding_cache.set_hbm_cache_fused_admission(fused_lookup)
                keys = torch.arange(
                    self.num_tables * self.num_embeddings,
                    device=torch.accelerator.current_accelerator(),
                )
                kv.kv_embedding_cache.get_embeddings(keys)

                self._assert_matches_reference(reference, kv, num_requests=2)

                hits, misses, _, rejected, _ = (
                    kv.kv_embedding_cache.get_hbm_cache_stats()
                )
                self.assertEqual(misses, keys.numel())
                self.assertEqual(rejected, 0)
                self.assertGreater(hits, 0)

    def test_load_with_cuda_map_location(self) -> None:
        kv = KVZCHGPUBucketWiseEmbeddingInference(
            [("", 16, 128, SparseType.INT4, EmbeddingLocation.HOST)],
            device="cpu",
        )
        kv.initialize_weights()
        archive = io.BytesIO()
        torch.jit.save(torch.jit.script(kv), archive)
        archive.seek(0)

        loaded = torch.jit.load(
            archive, map_location=torch.accelerator.current_accelerator()
        )
        loaded.initialize_kv_embedding_cache()

        self.assertEqual(loaded.kv_embedding_cache.serialize()[0].item(), 32)

    def test_scripted_module_loaded_to_cuda_serves_int64_ids(self) -> None:
        # The serving path: publish scripts and saves a module built on the CPU;
        # the predictor loads it onto the GPU, sets its device, initializes the
        # KV cache, loads CUDA-padded rows and looks up int64 ids.
        device = torch.accelerator.current_accelerator()
        assert device is not None
        num_embeddings, dim = 64, 128
        for weight_ty, pooling_mode, weighted in (
            (SparseType.INT8, PoolingMode.SUM, True),
            (SparseType.INT8, PoolingMode.SUM, False),
            (SparseType.INT4, PoolingMode.NONE, False),
        ):
            with self.subTest(
                weight_ty=weight_ty, pooling_mode=pooling_mode, weighted=weighted
            ):
                reference = IntNBitTableBatchedEmbeddingBagsCodegen(
                    [
                        ("", num_embeddings, dim, weight_ty, EmbeddingLocation.DEVICE)
                        for _ in range(2)
                    ],
                    pooling_mode=pooling_mode,
                    output_dtype=SparseType.FP16,
                    device="cuda",
                )
                _fill_finite_weights(reference)
                kv = KVZCHGPUBucketWiseEmbeddingInference(
                    [
                        ("", num_embeddings, dim, weight_ty, EmbeddingLocation.HOST)
                        for _ in range(2)
                    ],
                    pooling_mode=pooling_mode,
                    output_dtype=SparseType.FP16,
                    device="cpu",
                    embedding_cache_mode=True,
                )
                kv.initialize_weights()
                archive = io.BytesIO()
                torch.jit.save(torch.jit.script(kv), archive)
                archive.seek(0)
                loaded = torch.jit.load(archive, map_location=device)
                loaded.set_device(str(device))
                loaded.initialize_kv_embedding_cache()
                loaded.kv_embedding_cache.set_hbm_cache_rows(256)
                _load_kv_weights(reference, loaded)
                loaded.weight_initialized = True
                # Citrine C3: create the inputs on the device.
                generator = torch.Generator(device=device).manual_seed(0)
                indices = torch.randint(
                    num_embeddings,
                    (32,),
                    dtype=torch.int64,
                    device=device,
                    generator=generator,
                )
                offsets = torch.arange(0, 33, 2, dtype=torch.int64, device=device)
                weights = (
                    torch.rand(
                        32, device=device, dtype=torch.float32, generator=generator
                    )
                    if weighted
                    else None
                )
                expected = reference(indices, offsets, weights)
                self.assertTrue(torch.isfinite(expected).all())
                # Served from the backend, then from the HBM cache.
                for _ in range(2):
                    actual = loaded(indices, offsets, weights)
                    self.assertEqual(actual.device, indices.device)
                    torch.testing.assert_close(actual, expected)
                self.assertGreater(
                    loaded.kv_embedding_cache.get_hbm_cache_stats()[0], 0
                )

    def test_cuda_forward_needs_initialized_cache_and_uniform_rows(self) -> None:
        device = torch.accelerator.current_accelerator()
        assert device is not None
        # Citrine C3: create the inputs on the device.
        indices = torch.tensor([1, 2], dtype=torch.int64, device=device)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
        for dims, message in (
            ((128, 128), "initialize_kv_embedding_cache first"),
            ((128, 64), "one CUDA-aligned row width"),
        ):
            with self.subTest(dims=dims):
                kv = KVZCHGPUBucketWiseEmbeddingInference(
                    [
                        ("", 16, d, SparseType.INT8, EmbeddingLocation.DEVICE)
                        for d in dims
                    ],
                    output_dtype=SparseType.FP16,
                    device="cuda",
                )
                if dims[0] != dims[1]:
                    kv.initialize_kv_embedding_cache()
                kv.weight_initialized = True
                with self.assertRaisesRegex(AssertionError, message):
                    kv(indices, offsets)

    def test_unpooled_lookup_spreads_rows_over_warps(self) -> None:
        num_tables, num_indices = 40, 200_000
        reference, kv = _cuda_kv_with_reference(
            num_tables,
            self.num_embeddings,
            self.dim,
            weight_ty=SparseType.INT4,
            pooling_mode=PoolingMode.NONE,
        )
        device = torch.accelerator.current_accelerator()
        indices = torch.randint(
            self.num_embeddings, (num_indices,), dtype=torch.int32, device=device
        )
        # One bag per feature, so the per-feature kernel gets 40 long bags.
        offsets = torch.linspace(0, num_indices, num_tables + 1, device=device).to(
            torch.int32
        )
        rows = kv.kv_embedding_cache.get_embeddings(
            torch.ops.fbgemm.linearize_cache_indices(
                kv.feature_hash_size_cumsum, indices, offsets
            )
        )

        def timed_lookup() -> tuple[torch.Tensor, float]:
            kv.lookup_kv_rows(rows, indices, offsets, None)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = kv.lookup_kv_rows(rows, indices, offsets, None)
            end.record()
            torch.cuda.synchronize()
            return output, start.elapsed_time(end)

        self.assertTrue(kv.kv_cuda_single_spec)
        per_index, per_index_ms = timed_lookup()
        kv.kv_cuda_single_spec = False
        per_feature, per_feature_ms = timed_lookup()

        print(
            f"per-feature bags {per_feature_ms:.3f} ms, per-index {per_index_ms:.3f} ms"
        )
        # Random scales include NaNs, so compare bitwise with NaN equal to NaN.
        torch.testing.assert_close(
            per_index, per_feature, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            per_index, reference(indices, offsets), equal_nan=True
        )

    def _unpooled_int4_kv(self) -> KVZCHGPUBucketWiseEmbeddingInference:
        _, kv = _cuda_kv_with_reference(
            4,
            self.num_embeddings,
            self.dim,
            weight_ty=SparseType.INT4,
            pooling_mode=PoolingMode.NONE,
        )
        return kv

    def _lookup_both_ways(
        self,
        kv: KVZCHGPUBucketWiseEmbeddingInference,
        rows: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.zeros(rows.shape[0], dtype=torch.int32, device=rows.device)
        kv.kv_cuda_single_spec = True
        per_index = kv.lookup_kv_rows(rows, indices, offsets, None)
        kv.kv_cuda_single_spec = False
        return per_index, kv.lookup_kv_rows(rows, indices, offsets, None)

    def test_unpooled_lookup_edge_cases_match_per_feature_bags(self) -> None:
        kv = self._unpooled_int4_kv()
        device = torch.accelerator.current_accelerator()
        row_bytes = kv.kv_cuda_table_row_bytes[0]
        cases = {
            # The same key in every position yields the same row each time.
            "duplicates": kv.kv_embedding_cache.get_embeddings(
                torch.full((12,), 3, dtype=torch.int64, device=device)
            ),
            # Keys the cache lacks come back as zero rows.
            "missing": torch.zeros(12, row_bytes, dtype=torch.uint8, device=device),
            "empty": torch.zeros(0, row_bytes, dtype=torch.uint8, device=device),
        }
        for name, rows in cases.items():
            with self.subTest(name):
                offsets = torch.tensor(
                    [
                        0,
                        rows.shape[0] // 2,
                        rows.shape[0],
                        rows.shape[0],
                        rows.shape[0],
                    ],
                    dtype=torch.int32,
                    device=device,
                )

                per_index, per_feature = self._lookup_both_ways(kv, rows, offsets)

                self.assertEqual(per_index.shape, (rows.shape[0], self.dim))
                torch.testing.assert_close(
                    per_index, per_feature, rtol=0, atol=0, equal_nan=True
                )
                if name == "missing":
                    self.assertFalse(per_index.any())

    @skipIf(torch.cuda.device_count() < 2, "Requires two CUDA devices")
    def test_shared_backend_keeps_hbm_cache_per_device(self) -> None:
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        print(
            "KV HBM replica CUDA devices:",
            {index: torch.cuda.get_device_name(index) for index in range(2)},
        )
        for fused_lookup in (False, True):
            with self.subTest(fused_lookup=fused_lookup):
                _, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                devices = [torch.device(f"cuda:{i}") for i in range(2)]
                for device in devices:
                    with self.subTest(device=str(device)):
                        replica = KVZCHGPUBucketWiseEmbeddingInference(
                            [
                                (
                                    "",
                                    self.num_embeddings,
                                    self.dim,
                                    SparseType.INT8,
                                    EmbeddingLocation.DEVICE,
                                )
                                for _ in range(self.num_tables)
                            ],
                            output_dtype=SparseType.FP16,
                            device=str(device),
                        )
                        replica.set_device(str(device))
                        replica.initialize_kv_embedding_cache()
                        replica.kv_embedding_cache.share_kv_backend(
                            kv.kv_embedding_cache
                        )
                        replica.weight_initialized = True
                        replica.kv_embedding_cache.set_hbm_cache_rows(128)
                        replica.kv_embedding_cache.set_hbm_cache_fused_lookup(
                            fused_lookup
                        )
                        replica.kv_embedding_cache.set_hbm_cache_fused_admission(
                            fused_lookup
                        )
                        keys = torch.arange(48, dtype=torch.int64, device=device)

                        for _ in range(2):
                            rows = replica.kv_embedding_cache.get_embeddings(keys)
                            self.assertEqual(
                                replica.kv_embedding_cache.get_hbm_cache_device(),
                                str(device),
                            )
                            self.assertEqual(rows.device, device)
                            self._assert_encoded_rows_match_cpu(replica, keys, rows)
                        hits, misses, _, rejected, _ = (
                            replica.kv_embedding_cache.get_hbm_cache_stats()
                        )
                        self.assertEqual(hits, keys.numel())
                        self.assertEqual(misses, keys.numel())
                        self.assertEqual(rejected, 0)
                        if fused_lookup:
                            self.assertEqual(
                                replica.kv_embedding_cache.get_hbm_cache_fused_admission_stats()[
                                    :3
                                ],
                                [1, 1, keys.numel()],
                            )

    def test_sharing_bound_backend_again_keeps_hbm_cache(self) -> None:
        for fused_lookup in (False, True):
            with self.subTest(fused_lookup=fused_lookup):
                _, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                _, replica = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                replica.kv_embedding_cache.share_kv_backend(kv.kv_embedding_cache)
                replica.kv_embedding_cache.set_hbm_cache_rows(128)
                replica.kv_embedding_cache.set_hbm_cache_fused_lookup(fused_lookup)
                replica.kv_embedding_cache.set_hbm_cache_fused_admission(fused_lookup)
                keys = torch.arange(
                    48,
                    dtype=torch.int64,
                    device=torch.accelerator.current_accelerator(),
                )
                cold_rows = replica.kv_embedding_cache.get_embeddings(keys)
                data_ptr = replica.kv_embedding_cache.get_hbm_cache_data_ptr()
                self.assertNotEqual(data_ptr, 0)
                for _ in range(3):
                    before = replica.kv_embedding_cache.get_hbm_cache_stats()
                    fused_before = (
                        replica.kv_embedding_cache.get_hbm_cache_fused_admission_stats()
                    )
                    replica.kv_embedding_cache.share_kv_backend(kv.kv_embedding_cache)
                    self.assertEqual(
                        replica.kv_embedding_cache.get_hbm_cache_stats(), before
                    )
                    self.assertEqual(
                        replica.kv_embedding_cache.get_hbm_cache_data_ptr(), data_ptr
                    )
                    warm_rows = replica.kv_embedding_cache.get_embeddings(keys)
                    self.assertTrue(torch.equal(warm_rows, cold_rows))
                    after = replica.kv_embedding_cache.get_hbm_cache_stats()
                    self.assertEqual(after[0] - before[0], keys.numel())
                    self.assertEqual(after[1], before[1])
                    self.assertEqual(
                        replica.kv_embedding_cache.get_hbm_cache_fused_admission_stats(),
                        fused_before,
                    )
                for enabled in (not fused_lookup, fused_lookup):
                    before = replica.kv_embedding_cache.get_hbm_cache_stats()
                    fused_before = (
                        replica.kv_embedding_cache.get_hbm_cache_fused_admission_stats()
                    )
                    replica.kv_embedding_cache.set_hbm_cache_fused_lookup(enabled)
                    replica.kv_embedding_cache.set_hbm_cache_fused_admission(enabled)
                    self.assertEqual(
                        replica.kv_embedding_cache.get_hbm_cache_stats(), before
                    )
                    self.assertEqual(
                        replica.kv_embedding_cache.get_hbm_cache_data_ptr(), data_ptr
                    )
                    self.assertTrue(
                        torch.equal(
                            replica.kv_embedding_cache.get_embeddings(keys), cold_rows
                        )
                    )
                    after = replica.kv_embedding_cache.get_hbm_cache_stats()
                    self.assertEqual(after[0] - before[0], keys.numel())
                    self.assertEqual(after[1], before[1])
                    self.assertEqual(
                        replica.kv_embedding_cache.get_hbm_cache_fused_admission_stats(),
                        fused_before,
                    )

    def test_update_invalidates_cached_rows(self) -> None:
        for fused_lookup in (False, True):
            with self.subTest(fused_lookup=fused_lookup):
                reference, kv = _cuda_kv_with_reference(
                    self.num_tables, self.num_embeddings, self.dim
                )
                kv.kv_embedding_cache.set_hbm_cache_rows(4096)
                kv.kv_embedding_cache.set_hbm_cache_fused_lookup(fused_lookup)
                kv.kv_embedding_cache.set_hbm_cache_fused_admission(fused_lookup)
                self._assert_matches_reference(reference, kv, num_requests=1)

                _fill_random_weights(reference)
                _load_kv_weights(reference, kv)

                self._assert_matches_reference(reference, kv, num_requests=2)
                self.assertGreater(kv.kv_embedding_cache.get_hbm_cache_stats()[4], 0)

    @staticmethod
    def _grouped_int4_rows(
        num_embeddings: int, dim: int, group_size: int, seed: int = 0
    ) -> torch.Tensor:
        rows = torch.randn(
            num_embeddings, dim, generator=torch.Generator().manual_seed(seed)
        )
        return torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
            (rows * 0.05).half(), group_size
        )

    def _cuda_grouped_int4_kv(
        self,
        pooling_mode: PoolingMode = PoolingMode.NONE,
        num_embeddings: int = 128,
        fused: bool = True,
    ) -> tuple[KVZCHGPUBucketWiseEmbeddingInference, torch.Tensor]:
        """A grouped INT4 KV module on CUDA with an HBM row cache, holding its
        first 128 ids; also returns their packed rows."""
        kv = KVZCHGPUBucketWiseEmbeddingInference(
            [("", num_embeddings, 128, SparseType.INT4, EmbeddingLocation.DEVICE)],
            pooling_mode=pooling_mode,
            output_dtype=SparseType.FP16,
            device="cuda",
            embedding_cache_mode=True,
            grouped_int4_group_size=8,
        )
        kv.initialize_kv_embedding_cache()
        packed = self._grouped_int4_rows(128, 128, 8)
        kv.embedding_inplace_update_per_table(
            0, torch.arange(128, dtype=torch.int64), packed
        )
        kv.weight_initialized = True
        cache = kv.kv_embedding_cache
        cache.set_hbm_cache_rows(1024)
        cache.set_hbm_cache_fused_lookup(fused)
        cache.set_hbm_cache_fused_admission(fused)
        cache.set_hbm_cache_direct_output(False)
        return kv, packed

    def test_grouped_int4_cache_serves_packed_rows_and_decoded_outputs(self) -> None:
        device = torch.accelerator.current_accelerator()
        keys = torch.cat((torch.arange(64, dtype=torch.int64), torch.tensor([7, 1, 7])))
        indices = keys.to(device=device, dtype=torch.int32)
        offsets = torch.arange(keys.numel() + 1, device=device, dtype=torch.int32)
        for fused in (False, True):
            with self.subTest(fused=fused):
                kv, packed = self._cuda_grouped_int4_kv(fused=fused)
                self.assertTrue(kv.kv_cuda_uniform_rows)
                decoded = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, 8, 128)
                for _ in range(2):
                    rows = kv.kv_embedding_cache.get_embeddings(indices)
                    self.assertEqual(rows.shape, (keys.numel(), 80))
                    self.assertTrue(torch.equal(rows.cpu(), packed[keys]))
                    output = kv(indices, offsets)
                    self.assertEqual(output.device, indices.device)
                    self.assertTrue(
                        torch.equal(
                            output.cpu().view(torch.int16),
                            decoded[keys].view(torch.int16),
                        )
                    )
                hits, misses, admitted, _, _ = (
                    kv.kv_embedding_cache.get_hbm_cache_stats()
                )
                self.assertGreater(hits, 0)
                self.assertGreater(misses, 0)
                self.assertGreater(admitted, 0)

    def test_grouped_int4_unpooled_lookup_spreads_rows_over_warps(self) -> None:
        kv, packed = self._cuda_grouped_int4_kv()
        self.assertTrue(kv.kv_cuda_single_spec)
        device = torch.accelerator.current_accelerator()
        keys = torch.randint(128, (20_000,), generator=torch.Generator().manual_seed(0))
        indices = keys.to(device=device, dtype=torch.int32)
        # Two long bags, as a batch of two requests gives the per-bag kernel.
        offsets = torch.tensor([0, 7_000, 20_000], dtype=torch.int32, device=device)
        rows = kv.decode_grouped_int4_rows(
            kv.kv_embedding_cache.get_embeddings(indices.to(torch.int64))
        )

        per_index = kv.lookup_kv_rows(
            rows, indices, offsets, None, decoded_grouped_int4=True
        )
        kv.kv_cuda_single_spec = False
        per_bag = kv.lookup_kv_rows(
            rows, indices, offsets, None, decoded_grouped_int4=True
        )

        decoded = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, 8, 128)
        self.assertTrue(
            torch.equal(per_index.view(torch.int16), per_bag.view(torch.int16))
        )
        self.assertTrue(
            torch.equal(
                per_index.cpu().view(torch.int16), decoded[keys].view(torch.int16)
            )
        )

    def test_grouped_int4_cuda_lookup_matches_the_cpu_module(self) -> None:
        device = torch.accelerator.current_accelerator()
        indices = torch.tensor([3, 1, 7, 0, 5, 5, 90, 3], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 3, 6, 8], dtype=torch.int32)
        for pooling_mode in (PoolingMode.NONE, PoolingMode.SUM):
            with self.subTest(pooling_mode=pooling_mode):
                kv, packed = self._cuda_grouped_int4_kv(pooling_mode)
                cpu_kv = KVZCHGPUBucketWiseEmbeddingInference(
                    [("", 128, 128, SparseType.INT4, EmbeddingLocation.HOST)],
                    pooling_mode=pooling_mode,
                    output_dtype=SparseType.FP16,
                    device="cpu",
                    grouped_int4_group_size=8,
                )
                cpu_kv.initialize_kv_embedding_cache()
                cpu_kv.embedding_inplace_update_per_table(
                    0, torch.arange(128, dtype=torch.int64), packed
                )
                cpu_kv.weight_initialized = True
                expected = cpu_kv(indices, offsets)
                for _ in range(2):
                    torch.testing.assert_close(
                        kv(indices.to(device), offsets.to(device)).cpu(),
                        expected,
                        rtol=0 if pooling_mode == PoolingMode.NONE else 1e-3,
                        atol=0 if pooling_mode == PoolingMode.NONE else 1e-3,
                    )

    def test_grouped_int4_delta_replaces_cached_rows(self) -> None:
        for fused in (False, True):
            with self.subTest(fused=fused):
                kv, _ = self._cuda_grouped_int4_kv(fused=fused)
                cache = kv.kv_embedding_cache
                ids = torch.arange(40, dtype=torch.int64)
                keys = ids.to(torch.accelerator.current_accelerator())
                for _ in range(2):
                    cache.get_embeddings(keys)
                updated = self._grouped_int4_rows(40, 128, 8, seed=1)

                kv.embedding_inplace_update_per_table(0, ids, updated)

                self.assertTrue(torch.equal(cache.get_embeddings(keys).cpu(), updated))
                self.assertGreater(cache.get_hbm_cache_stats()[4], 0)

    def test_grouped_int4_bound_backend_serves_from_the_new_module(self) -> None:
        device = torch.accelerator.current_accelerator()
        keys = torch.arange(48, dtype=torch.int64, device=device)
        previous, packed = self._cuda_grouped_int4_kv()
        previous.kv_embedding_cache.get_embeddings(keys)
        current, _ = self._cuda_grouped_int4_kv()
        # In-place model update: the new module serves the previous backend.
        current.kv_embedding_cache.share_kv_backend(previous.kv_embedding_cache)
        cache = current.kv_embedding_cache
        for _ in range(2):
            self.assertTrue(torch.equal(cache.get_embeddings(keys).cpu(), packed[:48]))
        self.assertGreater(cache.get_hbm_cache_stats()[0], 0)
        updated = self._grouped_int4_rows(8, 128, 8, seed=3)

        current.embedding_inplace_update_per_table(
            0, torch.arange(8, dtype=torch.int64), updated
        )

        expected = torch.cat([updated, packed[8:48]])
        self.assertTrue(torch.equal(cache.get_embeddings(keys).cpu(), expected))
        self.assertGreater(cache.get_hbm_cache_stats()[4], 0)
