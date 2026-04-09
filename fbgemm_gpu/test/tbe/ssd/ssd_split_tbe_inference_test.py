# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import random
import time
import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    rounded_row_size_in_bytes,
    unpadded_row_size_in_bytes,
)
from fbgemm_gpu.tbe.ssd import SSDIntNBitTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.ssd.common import ASSOC
from fbgemm_gpu.tbe.utils import (
    b_indices,
    fake_quantize_embs,
    get_table_batched_offsets_from_dense,
    round_up,
)
from hypothesis import given, settings, Verbosity

from .. import common  # noqa E402
from ..common import gpu_unavailable, running_in_oss

MAX_EXAMPLES = 40


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
@unittest.skipIf(True, "Test is broken.")
class SSDIntNBitTableBatchedEmbeddingsTest(unittest.TestCase):
    def test_nbit_ssd(self) -> None:
        import tempfile

        E = int(1e4)
        D = 128
        N = 100
        embedding_specs = [("", E, D, SparseType.FP32)]
        weight_dim = rounded_row_size_in_bytes(
            embedding_specs[0][2],
            embedding_specs[0][3],
            16,
            DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
        )
        indices = torch.as_tensor(np.random.choice(E, replace=False, size=(N,)))
        weights = torch.empty(N, weight_dim, dtype=torch.uint8)
        output_weights = torch.empty_like(weights)
        count = torch.tensor([N])

        feature_table_map = list(range(1))
        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=embedding_specs,
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=1,
        )
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()

        emb.ssd_db.set_cuda(indices, weights, count, 1)
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        torch.testing.assert_close(weights, output_weights)

    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        # FIXME: Disable positional weight due to numerical issues.
        weighted=st.just(False),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        mixed_weights_ty=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_ssd_forward(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        weights_ty: SparseType,
        mixed_weights_ty: bool,
    ) -> None:
        import tempfile

        if not mixed_weights_ty:
            weights_ty_list = [weights_ty] * T
        else:
            weights_ty_list = [
                random.choice(
                    [
                        SparseType.FP32,
                        SparseType.FP16,
                        SparseType.INT8,
                        SparseType.INT4,
                        SparseType.INT2,
                    ]
                )
                for _ in range(T)
            ]

        D_alignment = max(
            1 if ty.bit_rate() % 8 == 0 else int(8 / ty.bit_rate())
            for ty in weights_ty_list
        )
        D = round_up(D, D_alignment)

        E = int(10**log_E)

        Ds = [D] * T
        Es = [E] * T

        row_alignment = 16

        feature_table_map = list(range(T))
        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[
                ("", E, D, W_TY) for (E, D, W_TY) in zip(Es, Ds, weights_ty_list)
            ],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=max(T * B * L, 1),
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
        ).cuda()

        bs = [
            torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
            for (E, D) in zip(Es, Ds)
        ]
        torch.manual_seed(42)
        xs = [torch.randint(low=0, high=e, size=(B, L)).cuda() for e in Es]
        xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]

        for t in range(T):
            weights, scale_shift = emb.split_embedding_weights()[t]

            if scale_shift is not None:
                E, R = scale_shift.shape
                self.assertEqual(R, 4)
                scales = np.random.uniform(0.1, 1, size=(E,)).astype(np.float16)
                shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)
                scale_shift[:, :] = torch.tensor(
                    np.stack([scales, shifts], axis=1).astype(np.float16).view(np.uint8)
                )

            D_bytes = rounded_row_size_in_bytes(
                Ds[t], weights_ty_list[t], row_alignment
            )
            copy_byte_tensor = torch.empty([E, D_bytes], dtype=torch.uint8)

            fake_quantize_embs(
                weights,
                scale_shift,
                bs[t].weight.detach(),
                weights_ty_list[t],
                use_cpu=False,
            )

            if weights_ty_list[t] in [SparseType.FP32, SparseType.FP16, SparseType.FP8]:
                copy_byte_tensor[
                    :,
                    : unpadded_row_size_in_bytes(Ds[t], weights_ty_list[t]),
                ] = weights  # q_weights
            else:
                copy_byte_tensor[
                    :,
                    emb.scale_bias_size_in_bytes : unpadded_row_size_in_bytes(
                        Ds[t], weights_ty_list[t]
                    ),
                ] = weights  # q_weights
                # fmt: off
                copy_byte_tensor[:, : emb.scale_bias_size_in_bytes] = (
                    scale_shift  # q_scale_shift
                )
                # fmt: on

            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                copy_byte_tensor,
                torch.as_tensor([E]),
                t,
            )
        torch.cuda.synchronize()

        fs = (
            [b_indices(b, x) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1))
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        f = torch.cat([f.view(B, -1) for f in fs], dim=1)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws], dim=0)
        indices, offsets = get_table_batched_offsets_from_dense(x)
        fc2 = (
            emb(indices.cuda().int(), offsets.cuda().int())
            if not weighted
            else emb(
                indices.cuda().int(),
                offsets.cuda().int(),
                xw.contiguous().view(-1).cuda(),
            )
        )
        torch.testing.assert_close(
            fc2.float(),
            f.float(),
            atol=1.0e-2,
            rtol=1.0e-2,
            equal_nan=True,
        )
        time.sleep(0.1)

    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_ssd_cache(
        self, T: int, D: int, B: int, log_E: int, L: int, weighted: bool
    ) -> None:
        import tempfile

        weights_ty = random.choice(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        )

        D_alignment = (
            1 if weights_ty.bit_rate() % 8 == 0 else int(8 / weights_ty.bit_rate())
        )
        D = round_up(D, D_alignment)

        E = int(10**log_E)
        Ds = [D] * T
        Es = [E] * T
        weights_ty_list = [weights_ty] * T
        C = max(T * B * L, 1)

        row_alignment = 16

        feature_table_map = list(range(T))
        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[
                ("", E, D, W_TY) for (E, D, W_TY) in zip(Es, Ds, weights_ty_list)
            ],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=C,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            ssd_shards=2,
            pooling_mode=PoolingMode.SUM,
        ).cuda()

        bs = [
            torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
            for (E, D) in zip(Es, Ds)
        ]
        torch.manual_seed(42)

        for t in range(T):
            weights, scale_shift = emb.split_embedding_weights()[t]

            if scale_shift is not None:
                E, R = scale_shift.shape
                self.assertEqual(R, 4)
                if weights_ty_list[t] == SparseType.INT2:
                    scales = np.random.uniform(0.1, 1, size=(E,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)
                if weights_ty_list[t] == SparseType.INT4:
                    scales = np.random.uniform(0.01, 0.1, size=(E,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)
                if weights_ty_list[t] == SparseType.INT8:
                    scales = np.random.uniform(0.001, 0.01, size=(E,)).astype(
                        np.float16
                    )
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)

                scale_shift[:, :] = torch.tensor(
                    # pyre-fixme[61]: `scales` is undefined, or not always defined.
                    # pyre-fixme[61]: `shifts` is undefined, or not always defined.
                    np.stack([scales, shifts], axis=1)
                    .astype(np.float16)
                    .view(np.uint8)
                )

            D_bytes = rounded_row_size_in_bytes(
                Ds[t], weights_ty_list[t], row_alignment
            )
            copy_byte_tensor = torch.empty([E, D_bytes], dtype=torch.uint8)

            fake_quantize_embs(
                weights,
                scale_shift,
                bs[t].weight.detach(),
                weights_ty_list[t],
                use_cpu=False,
            )

            if weights_ty_list[t] in [SparseType.FP32, SparseType.FP16, SparseType.FP8]:
                copy_byte_tensor[
                    :,
                    : unpadded_row_size_in_bytes(Ds[t], weights_ty_list[t]),
                ] = weights  # q_weights
            else:
                copy_byte_tensor[
                    :,
                    emb.scale_bias_size_in_bytes : unpadded_row_size_in_bytes(
                        Ds[t], weights_ty_list[t]
                    ),
                ] = weights  # q_weights
                # fmt: off
                copy_byte_tensor[:, : emb.scale_bias_size_in_bytes] = (
                    scale_shift  # q_scale_shift
                )
                # fmt: on

            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                copy_byte_tensor,
                torch.as_tensor([E]),
                t,
            )
        torch.cuda.synchronize()

        for i in range(10):
            xs = [torch.randint(low=0, high=e, size=(B, L)).cuda() for e in Es]
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]
            xw = torch.cat([xw.view(1, B, L) for xw in xws], dim=0)

            indices, offsets = get_table_batched_offsets_from_dense(x)
            indices, offsets = indices.cuda(), offsets.cuda()
            assert emb.timestep_counter.get() == i

            emb.prefetch(indices, offsets)

            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                emb.hash_size_cumsum,
                indices,
                offsets,
            )

            # Verify that prefetching twice avoids any actions.
            (
                _,
                _,
                _,
                actions_count_gpu,
                _,
                _,
                _,
                _,
            ) = torch.ops.fbgemm.ssd_cache_populate_actions(  # noqa
                linear_cache_indices,
                emb.total_hash_size,
                emb.lxu_cache_state,
                emb.timestep_counter.get(),
                0,  # prefetch_dist
                emb.lru_state,
            )
            assert actions_count_gpu.item() == 0

            lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                linear_cache_indices,
                emb.lxu_cache_state,
                emb.hash_size_cumsum[-1],
            )
            lru_state_cpu = emb.lru_state.cpu()
            lxu_cache_state_cpu = emb.lxu_cache_state.cpu()

            NOT_FOUND = np.iinfo(np.int32).max
            ASSOC = 32

            for loc, linear_idx in zip(
                lxu_cache_locations.cpu().numpy().tolist(),
                linear_cache_indices.cpu().numpy().tolist(),
            ):
                assert loc != NOT_FOUND
                # if we have a hit, check the cache is consistent
                loc_set = loc // ASSOC
                loc_slot = loc % ASSOC
                assert lru_state_cpu[loc_set, loc_slot] == emb.timestep_counter.get()
                assert lxu_cache_state_cpu[loc_set, loc_slot] == linear_idx
            fs = (
                [b_indices(b, x) for (b, x) in zip(bs, xs)]
                if not weighted
                else [
                    b_indices(b, x, per_sample_weights=xw.view(-1))
                    for (b, x, xw) in zip(bs, xs, xws)
                ]
            )
            f = torch.cat([f.view(B, -1) for f in fs], dim=1)
            fc2 = (
                emb(indices.cuda().int(), offsets.cuda().int())
                if not weighted
                else emb(
                    indices.cuda().int(),
                    offsets.cuda().int(),
                    xw.contiguous().view(-1).cuda(),
                )
            )
            torch.testing.assert_close(
                fc2.float(),
                f.float(),
                atol=1.0e-2,
                rtol=1.0e-2,
            )
        time.sleep(0.1)


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class SSDInferenceCacheLockingTest(unittest.TestCase):
    """
    Tests for D96632292: Cache locking and memcpy stream in inference path.
    Separated from the broken test class above so these actually run.
    """

    def test_lxu_cache_locking_counter_registered(self) -> None:
        """
        Test D96632292: Verify lxu_cache_locking_counter buffer is registered
        with correct shape (cache_sets, ASSOC) and dtype int32.
        """
        import tempfile

        E = int(1e4)
        D = 128
        cache_sets = 64
        ASSOC = 32  # hardcoded in FBGEMM

        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[("", E, D, SparseType.FP32)],
            feature_table_map=[0],
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=cache_sets,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
        ).cuda()

        self.assertTrue(
            hasattr(emb, "lxu_cache_locking_counter"),
            "lxu_cache_locking_counter should be registered as a buffer",
        )
        self.assertEqual(
            emb.lxu_cache_locking_counter.shape,
            (cache_sets, ASSOC),
            f"Expected shape ({cache_sets}, {ASSOC}), "
            f"got {emb.lxu_cache_locking_counter.shape}",
        )
        self.assertEqual(
            emb.lxu_cache_locking_counter.dtype,
            torch.int32,
            "lxu_cache_locking_counter should be int32",
        )
        # Initially all zeros
        self.assertTrue(
            (emb.lxu_cache_locking_counter == 0).all().item(),
            "lxu_cache_locking_counter should be initialized to zeros",
        )

    def test_ssd_memcpy_stream_created(self) -> None:
        """
        Test D96632292: Verify ssd_memcpy_stream is created as a CUDA stream.
        """
        import tempfile

        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[("", int(1e4), 128, SparseType.FP32)],
            feature_table_map=[0],
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=64,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
        ).cuda()

        self.assertTrue(
            hasattr(emb, "ssd_memcpy_stream"),
            "ssd_memcpy_stream should exist",
        )
        self.assertIsInstance(
            emb.ssd_memcpy_stream,
            torch.cuda.Stream,
            "ssd_memcpy_stream should be a CUDA stream",
        )

    def test_cache_locking_prefetch_forward_cycle(self) -> None:
        """
        Test D96632292: Verify cache locking counter increments during prefetch
        and decrements during forward, with no negative values after completion.
        """
        import tempfile

        E = int(1e4)
        D = 128
        T = 2
        B = 8
        L = 5

        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[("", E, D, weights_ty)] * T,
            feature_table_map=list(range(T)),
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=max(T * B * L, 1),
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
            enable_cache_locking=True,
        ).cuda()

        # Initialize embeddings in SSD
        for t in range(T):
            copy_byte_tensor = torch.empty([E, D_bytes], dtype=torch.uint8)
            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                copy_byte_tensor,
                torch.as_tensor([E]),
                t,
            )
        torch.cuda.synchronize()

        # Run prefetch + forward cycles
        for _ in range(10):
            xs = [torch.randint(low=0, high=E, size=(B, L)).cuda() for _ in range(T)]
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            indices, offsets = get_table_batched_offsets_from_dense(x)
            indices, offsets = indices.cuda(), offsets.cuda()

            emb.prefetch(indices, offsets)
            emb(indices.int(), offsets.int())

        torch.cuda.synchronize()

        # After all cycles complete, counter should be non-negative
        counter = emb.lxu_cache_locking_counter.cpu()
        self.assertTrue(
            (counter >= 0).all().item(),
            f"Cache locking counter has negatives after cycles: "
            f"min={counter.min().item()}",
        )

    def test_cache_locking_disabled_by_default(self) -> None:
        """
        Test that with enable_cache_locking=False (default), the locking
        counter stays at zero and prefetch/forward still work correctly.
        """
        import tempfile

        E = int(1e4)
        D = 128
        T = 2
        B = 8
        L = 5

        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        # Default: enable_cache_locking=False
        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[("", E, D, weights_ty)] * T,
            feature_table_map=list(range(T)),
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=max(T * B * L, 1),
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
        ).cuda()

        self.assertFalse(
            emb.enable_cache_locking,
            "Cache locking should be disabled by default",
        )

        # Initialize embeddings in SSD
        for t in range(T):
            copy_byte_tensor = torch.empty([E, D_bytes], dtype=torch.uint8)
            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                copy_byte_tensor,
                torch.as_tensor([E]),
                t,
            )
        torch.cuda.synchronize()

        # Run prefetch + forward cycles — should work without locking
        for _ in range(5):
            xs = [torch.randint(low=0, high=E, size=(B, L)).cuda() for _ in range(T)]
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            indices, offsets = get_table_batched_offsets_from_dense(x)
            indices, offsets = indices.cuda(), offsets.cuda()

            emb.prefetch(indices, offsets)
            emb(indices.int(), offsets.int())

        torch.cuda.synchronize()

        # Counter should remain all zeros since locking is disabled
        counter = emb.lxu_cache_locking_counter.cpu()
        self.assertTrue(
            (counter == 0).all().item(),
            f"Cache locking counter should be all zeros when disabled, "
            f"but max={counter.max().item()}",
        )


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class SSDInferenceStreamingTest(unittest.TestCase):
    """
    Tests for streaming_update() and load_snapshot() in SSD TBE inference.
    """

    def _create_emb(
        self,
        E: int = 10000,
        D: int = 128,
        T: int = 1,
        cache_sets: int = 64,
        weights_ty: SparseType = SparseType.FP32,
    ) -> "SSDIntNBitTableBatchedEmbeddingBags":
        import tempfile

        return SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[("", E, D, weights_ty)] * T,
            feature_table_map=list(range(T)),
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=cache_sets,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
        ).cuda()

    def _init_table(
        self,
        emb: "SSDIntNBitTableBatchedEmbeddingBags",
        E: int,
        D_bytes: int,
        T: int = 1,
    ) -> torch.Tensor:
        """Initialize all tables with random data and return the weight tensor."""
        weights = torch.randint(0, 255, (T * E, D_bytes), dtype=torch.uint8)
        for t in range(T):
            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E, dtype=torch.int64),
                weights[t * E : (t + 1) * E],
                torch.tensor([E]),
                t,
            )
        torch.cuda.synchronize()
        return weights

    # ─── Basic streaming_update tests ───────────────────────────────────

    def test_streaming_update_writes_to_ssd(self) -> None:
        """Verify streaming_update() writes to RocksDB."""
        E = 1000
        D = 128
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=128, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        update_indices = torch.tensor([0, 10, 100, 999], dtype=torch.int64)
        N = update_indices.shape[0]
        new_weights = torch.randint(0, 255, (N, D_bytes), dtype=torch.uint8)
        emb.streaming_update(update_indices, new_weights)

        output = torch.empty(N, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(update_indices.cpu(), output, torch.tensor([N]))
        torch.cuda.synchronize()
        torch.testing.assert_close(output, new_weights)

    def test_streaming_update_invalidates_cache(self) -> None:
        """Verify streaming_update() invalidates HBM cache entries."""
        E = 1000
        D = 128
        T = 1
        B = 4
        L = 5
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(
            E=E,
            D=D,
            T=T,
            cache_sets=max(T * B * L, 1),
            weights_ty=weights_ty,
        )
        self._init_table(emb, E, D_bytes)

        # Prefetch to populate cache.
        xs = torch.randint(low=0, high=E, size=(B, L)).cuda()
        x = xs.view(1, B, L)
        indices, offsets = get_table_batched_offsets_from_dense(x)
        indices, offsets = indices.cuda(), offsets.cuda()
        emb.prefetch(indices, offsets)

        # Find which indices are cached.
        linear_indices = torch.ops.fbgemm.linearize_cache_indices(
            emb.hash_size_cumsum, indices, offsets
        )
        cache_state = emb.lxu_cache_state.cpu()
        cached_indices = set(cache_state[cache_state >= 0].tolist())
        requested_indices = set(linear_indices.cpu().tolist())
        cached_from_request = cached_indices & requested_indices
        self.assertGreater(len(cached_from_request), 0)

        # streaming_update those cached indices.
        update_list = sorted(cached_from_request)[:4]
        update_indices = torch.tensor(update_list, dtype=torch.int64)
        N = update_indices.shape[0]
        new_weights = torch.randint(0, 255, (N, D_bytes), dtype=torch.uint8)
        emb.streaming_update(update_indices, new_weights)

        # Verify invalidation.
        cache_state_after = emb.lxu_cache_state.cpu()
        for idx in update_list:
            cache_set = idx % cache_state_after.shape[0]
            slots = cache_state_after[cache_set]
            self.assertFalse(
                (slots == idx).any().item(),
                f"Index {idx} should have been invalidated from cache",
            )

    def test_streaming_update_empty(self) -> None:
        """Empty update is a no-op."""
        emb = self._create_emb()
        emb.streaming_update(
            torch.tensor([], dtype=torch.int64),
            torch.empty(0, emb.max_D_cache, dtype=torch.uint8),
        )

    # ─── Edge cases and corner cases ────────────────────────────────────

    def test_streaming_update_single_row(self) -> None:
        """Update a single row."""
        E = 100
        D = 64
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=32, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        idx = torch.tensor([42], dtype=torch.int64)
        new_w = torch.randint(0, 255, (1, D_bytes), dtype=torch.uint8)
        emb.streaming_update(idx, new_w)

        out = torch.empty(1, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(idx, out, torch.tensor([1]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, new_w)

    def test_streaming_update_first_and_last_index(self) -> None:
        """Update the first (0) and last (E-1) indices."""
        E = 500
        D = 128
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=32, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        indices = torch.tensor([0, E - 1], dtype=torch.int64)
        new_w = torch.randint(0, 255, (2, D_bytes), dtype=torch.uint8)
        emb.streaming_update(indices, new_w)

        out = torch.empty(2, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(indices, out, torch.tensor([2]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, new_w)

    def test_streaming_update_duplicate_indices(self) -> None:
        """
        Duplicate indices: last write wins for RocksDB, both cache entries
        should be invalidated.
        """
        E = 1000
        D = 128
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=64, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        # Same index appears twice with different values.
        indices = torch.tensor([42, 42], dtype=torch.int64)
        w1 = torch.randint(0, 255, (1, D_bytes), dtype=torch.uint8)
        w2 = torch.randint(0, 255, (1, D_bytes), dtype=torch.uint8)
        both = torch.cat([w1, w2], dim=0)
        emb.streaming_update(indices, both)

        # Read back — should get w2 (last write).
        out = torch.empty(1, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(
            torch.tensor([42], dtype=torch.int64), out, torch.tensor([1])
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, w2)

    def test_streaming_update_large_batch(self) -> None:
        """Update 1000 rows at once."""
        E = 5000
        D = 128
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=128, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        N = 1000
        indices = torch.randperm(E)[:N].to(torch.int64)
        new_w = torch.randint(0, 255, (N, D_bytes), dtype=torch.uint8)
        emb.streaming_update(indices, new_w)

        out = torch.empty(N, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(indices, out, torch.tensor([N]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, new_w)

    def test_streaming_update_preserves_non_updated_rows(self) -> None:
        """Rows not included in the update remain unchanged."""
        E = 200
        D = 64
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=32, weights_ty=weights_ty)
        original = self._init_table(emb, E, D_bytes)

        # Update only index 0.
        emb.streaming_update(
            torch.tensor([0], dtype=torch.int64),
            torch.randint(0, 255, (1, D_bytes), dtype=torch.uint8),
        )

        # Check that index 1 still has original data.
        out = torch.empty(1, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(
            torch.tensor([1], dtype=torch.int64), out, torch.tensor([1])
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, original[1:2])

    def test_streaming_update_sequential_updates(self) -> None:
        """Multiple sequential updates to the same index; last one wins."""
        E = 100
        D = 64
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=32, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        idx = torch.tensor([5], dtype=torch.int64)
        final_w = None
        for _ in range(10):
            final_w = torch.randint(0, 255, (1, D_bytes), dtype=torch.uint8)
            emb.streaming_update(idx, final_w)

        out = torch.empty(1, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(idx, out, torch.tensor([1]))
        torch.cuda.synchronize()
        assert final_w is not None
        torch.testing.assert_close(out, final_w)

    def test_streaming_update_preserves_non_updated_cache(self) -> None:
        """
        Updating index A should NOT invalidate unrelated index B in the cache,
        even if they map to the same cache set.
        """
        E = 1000
        D = 128
        T = 1
        B = 8
        L = 10
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)
        cache_sets = max(T * B * L, 1)

        emb = self._create_emb(
            E=E,
            D=D,
            T=T,
            cache_sets=cache_sets,
            weights_ty=weights_ty,
        )
        self._init_table(emb, E, D_bytes)

        # Prefetch to populate cache.
        xs = torch.randint(low=0, high=E, size=(B, L)).cuda()
        x = xs.view(1, B, L)
        indices, offsets = get_table_batched_offsets_from_dense(x)
        indices, offsets = indices.cuda(), offsets.cuda()
        emb.prefetch(indices, offsets)

        cache_state_before = emb.lxu_cache_state.cpu()
        all_cached = set(cache_state_before[cache_state_before >= 0].tolist())

        if len(all_cached) < 2:
            return  # Not enough cached entries to test.

        # Pick one index to update, count the rest.
        update_idx = sorted(all_cached)[0]
        remaining = all_cached - {update_idx}

        emb.streaming_update(
            torch.tensor([update_idx], dtype=torch.int64),
            torch.randint(0, 255, (1, D_bytes), dtype=torch.uint8),
        )

        # Verify all non-updated cached indices are still in the cache.
        cache_state_after = emb.lxu_cache_state.cpu()
        still_cached = set(cache_state_after[cache_state_after >= 0].tolist())
        for idx in remaining:
            self.assertIn(
                idx,
                still_cached,
                f"Non-updated index {idx} should still be in cache",
            )

    # ─── Different sparse types ─────────────────────────────────────────

    def test_streaming_update_int8(self) -> None:
        """streaming_update with INT8 embeddings."""
        E = 500
        D = 128
        weights_ty = SparseType.INT8
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=32, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        indices = torch.tensor([0, 50, 499], dtype=torch.int64)
        new_w = torch.randint(0, 255, (3, D_bytes), dtype=torch.uint8)
        emb.streaming_update(indices, new_w)

        out = torch.empty(3, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(indices, out, torch.tensor([3]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, new_w)

    def test_streaming_update_int4(self) -> None:
        """streaming_update with INT4 embeddings."""
        E = 500
        D = 128
        weights_ty = SparseType.INT4
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=32, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        indices = torch.tensor([0, 250, 499], dtype=torch.int64)
        new_w = torch.randint(0, 255, (3, D_bytes), dtype=torch.uint8)
        emb.streaming_update(indices, new_w)

        out = torch.empty(3, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(indices, out, torch.tensor([3]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, new_w)

    def test_streaming_update_fp16(self) -> None:
        """streaming_update with FP16 embeddings."""
        E = 500
        D = 128
        weights_ty = SparseType.FP16
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=32, weights_ty=weights_ty)
        self._init_table(emb, E, D_bytes)

        indices = torch.tensor([0, 100, 499], dtype=torch.int64)
        new_w = torch.randint(0, 255, (3, D_bytes), dtype=torch.uint8)
        emb.streaming_update(indices, new_w)

        out = torch.empty(3, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(indices, out, torch.tensor([3]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, new_w)

    # ─── End-to-end forward correctness after streaming_update ──────────

    def test_streaming_update_forward_correctness(self) -> None:
        """
        End-to-end: update some rows via streaming_update, then verify
        that prefetch + forward produces correct output using the updated
        weights.
        """
        import tempfile

        E = 1000
        D = 128
        T = 1
        B = 4
        L = 5
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[("", E, D, weights_ty)],
            feature_table_map=[0],
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=max(T * B * L, 1),
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
        ).cuda()

        # Create reference FP32 EmbeddingBag.
        ref = torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
        torch.manual_seed(42)

        # Write quantized weights to SSD TBE.
        weights, scale_shift = emb.split_embedding_weights()[0]
        fake_quantize_embs(
            weights,
            scale_shift,
            ref.weight.detach(),
            weights_ty,
            use_cpu=False,
        )
        copy_byte_tensor = torch.empty([E, D_bytes], dtype=torch.uint8)
        copy_byte_tensor[:, : unpadded_row_size_in_bytes(D, weights_ty)] = weights
        emb.ssd_db.set_cuda(
            torch.arange(E, dtype=torch.int64),
            copy_byte_tensor,
            torch.tensor([E]),
            0,
        )
        torch.cuda.synchronize()

        # Update some rows in the reference, then streaming_update TBE.
        update_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        new_ref_weights = torch.randn(5, D).cuda()
        ref.weight.data[update_indices] = new_ref_weights

        # Re-quantize updated rows.
        updated_weights, updated_ss = emb.split_embedding_weights()[0]
        fake_quantize_embs(
            updated_weights,
            updated_ss,
            ref.weight.detach(),
            weights_ty,
            use_cpu=False,
        )
        updated_copy = torch.empty([E, D_bytes], dtype=torch.uint8)
        updated_copy[:, : unpadded_row_size_in_bytes(D, weights_ty)] = updated_weights
        # Only streaming_update the changed rows.
        emb.streaming_update(update_indices, updated_copy[update_indices])

        # Now query indices that include updated rows.
        xs = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long).cuda()  # [1, B=1, L=5]
        x = xs.view(1, 1, 5)
        indices, offsets = get_table_batched_offsets_from_dense(x)
        indices, offsets = indices.cuda(), offsets.cuda()
        emb.prefetch(indices, offsets)
        result = emb(indices.int(), offsets.int())

        expected = b_indices(ref, xs.view(1, 5))
        torch.testing.assert_close(
            result.float(),
            expected.float(),
            atol=1e-2,
            rtol=1e-2,
        )

    # ─── Multi-table tests ──────────────────────────────────────────────

    def test_streaming_update_multi_table(self) -> None:
        """streaming_update on a multi-table TBE with linear index offsets."""
        import tempfile

        E = 500
        D = 64
        T = 3
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[("", E, D, weights_ty)] * T,
            feature_table_map=list(range(T)),
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=64,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
        ).cuda()
        self._init_table(emb, E, D_bytes, T=T)

        # Update row 0 of table 2 (linear index = 2*E + 0 = 1000).
        linear_idx = 2 * E
        idx = torch.tensor([linear_idx], dtype=torch.int64)
        new_w = torch.randint(0, 255, (1, D_bytes), dtype=torch.uint8)
        emb.streaming_update(idx, new_w)

        out = torch.empty(1, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(idx, out, torch.tensor([1]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, new_w)

    # ─── load_snapshot tests ────────────────────────────────────────────

    def test_load_snapshot(self) -> None:
        """load_snapshot() swaps RocksDB and clears HBM cache."""
        import tempfile

        E = 1000
        D = 128
        T = 1
        B = 4
        L = 5
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(
            E=E,
            D=D,
            T=T,
            cache_sets=max(T * B * L, 1),
            weights_ty=weights_ty,
        )
        initial_weights = self._init_table(emb, E, D_bytes)

        # Prefetch to populate cache.
        xs = torch.randint(low=0, high=E, size=(B, L)).cuda()
        x = xs.view(1, B, L)
        indices, offsets = get_table_batched_offsets_from_dense(x)
        indices, offsets = indices.cuda(), offsets.cuda()
        emb.prefetch(indices, offsets)

        cache_state_before = emb.lxu_cache_state.cpu()
        self.assertTrue((cache_state_before >= 0).any().item())

        # Load new snapshot.
        new_snapshot_dir = tempfile.mkdtemp()
        emb.load_snapshot(new_snapshot_dir)

        # Cache fully invalidated.
        cache_state_after = emb.lxu_cache_state.cpu()
        self.assertTrue((cache_state_after == -1).all().item())

        # Old data gone.
        check_indices = torch.tensor([0, 10, 100, 999], dtype=torch.int64)
        N = check_indices.shape[0]
        output = torch.empty(N, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(check_indices, output, torch.tensor([N]))
        torch.cuda.synchronize()
        old_data = initial_weights[check_indices]
        self.assertFalse(torch.equal(output, old_data))

        # Writes to new DB work.
        new_weights = torch.randint(0, 255, (N, D_bytes), dtype=torch.uint8)
        emb.streaming_update(check_indices, new_weights)
        output2 = torch.empty(N, D_bytes, dtype=torch.uint8)
        emb.ssd_db.get_cuda(check_indices, output2, torch.tensor([N]))
        torch.cuda.synchronize()
        torch.testing.assert_close(output2, new_weights)

    def test_load_snapshot_then_forward(self) -> None:
        """
        After load_snapshot + streaming_update, prefetch + forward returns
        correct results from the new data.
        """
        import tempfile

        E = 500
        D = 64
        T = 1
        B = 2
        L = 3
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=[("", E, D, weights_ty)],
            feature_table_map=[0],
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=max(T * B * L, 1),
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            pooling_mode=PoolingMode.SUM,
        ).cuda()

        # Initial data + forward.
        ref = torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
        torch.manual_seed(99)

        weights, scale_shift = emb.split_embedding_weights()[0]
        fake_quantize_embs(
            weights,
            scale_shift,
            ref.weight.detach(),
            weights_ty,
            use_cpu=False,
        )
        copy_bytes = torch.empty([E, D_bytes], dtype=torch.uint8)
        copy_bytes[:, : unpadded_row_size_in_bytes(D, weights_ty)] = weights
        emb.ssd_db.set_cuda(
            torch.arange(E, dtype=torch.int64),
            copy_bytes,
            torch.tensor([E]),
            0,
        )
        torch.cuda.synchronize()

        # Swap to new snapshot.
        new_dir = tempfile.mkdtemp()
        emb.load_snapshot(new_dir)

        # Write new reference data.
        ref2 = torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
        torch.manual_seed(123)
        ref2.weight.data = torch.randn_like(ref2.weight.data)

        weights2, ss2 = emb.split_embedding_weights()[0]
        fake_quantize_embs(
            weights2,
            ss2,
            ref2.weight.detach(),
            weights_ty,
            use_cpu=False,
        )
        copy_bytes2 = torch.empty([E, D_bytes], dtype=torch.uint8)
        copy_bytes2[:, : unpadded_row_size_in_bytes(D, weights_ty)] = weights2
        # Populate new snapshot via streaming_update in batches.
        batch_size = 100
        for start in range(0, E, batch_size):
            end = min(start + batch_size, E)
            idx = torch.arange(start, end, dtype=torch.int64)
            emb.streaming_update(idx, copy_bytes2[start:end])

        # Forward and verify against new reference.
        xs = torch.randint(0, E, (B, L)).cuda()
        x = xs.view(1, B, L)
        indices, offsets = get_table_batched_offsets_from_dense(x)
        indices, offsets = indices.cuda(), offsets.cuda()
        emb.prefetch(indices, offsets)
        result = emb(indices.int(), offsets.int())

        expected = b_indices(ref2, xs)
        torch.testing.assert_close(
            result.float(),
            expected.float(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_load_snapshot_multiple_swaps(self) -> None:
        """Multiple sequential load_snapshot calls work correctly."""
        import tempfile

        E = 200
        D = 64
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        emb = self._create_emb(E=E, D=D, cache_sets=32, weights_ty=weights_ty)

        for i in range(3):
            snap_dir = tempfile.mkdtemp()
            emb.load_snapshot(snap_dir)

            # Write data specific to this snapshot.
            marker = torch.full((1, D_bytes), fill_value=i + 1, dtype=torch.uint8)
            emb.streaming_update(torch.tensor([0], dtype=torch.int64), marker)

            out = torch.empty(1, D_bytes, dtype=torch.uint8)
            emb.ssd_db.get_cuda(
                torch.tensor([0], dtype=torch.int64), out, torch.tensor([1])
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(out, marker)

    # ─── Assertion / validation tests ───────────────────────────────────

    def test_streaming_update_asserts_1d_indices(self) -> None:
        """streaming_update rejects 2D indices."""
        emb = self._create_emb(E=100, D=64)
        with self.assertRaises(AssertionError):
            emb.streaming_update(
                torch.zeros(2, 2, dtype=torch.int64),
                torch.empty(4, emb.max_D_cache, dtype=torch.uint8),
            )

    def test_streaming_update_asserts_2d_weights(self) -> None:
        """streaming_update rejects 1D weights."""
        emb = self._create_emb(E=100, D=64)
        with self.assertRaises(AssertionError):
            emb.streaming_update(
                torch.tensor([0], dtype=torch.int64),
                torch.empty(emb.max_D_cache, dtype=torch.uint8),
            )

    def test_streaming_update_asserts_shape_mismatch(self) -> None:
        """streaming_update rejects mismatched indices/weights counts."""
        emb = self._create_emb(E=100, D=64)
        with self.assertRaises(AssertionError):
            emb.streaming_update(
                torch.tensor([0, 1], dtype=torch.int64),
                torch.empty(3, emb.max_D_cache, dtype=torch.uint8),
            )


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class TurboSSDInferenceModuleTest(unittest.TestCase):
    """
    Tests for the TurboSSDInferenceModule wrapper (HSTU integration).
    """

    def test_from_embedding_specs_creates_module(self) -> None:
        """Factory method creates a valid module."""
        import tempfile

        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        module = TurboSSDInferenceModule.from_embedding_specs(
            specs=[("table_0", 1000, 128, SparseType.FP32)],
            ssd_directory=tempfile.mkdtemp(),
            cache_hit_rate=0.50,
        )
        self.assertIsNotNone(module.tbe)
        self.assertEqual(len(module.embedding_specs), 1)
        self.assertEqual(module.embedding_specs[0][1], 1000)

    def test_from_embedding_specs_multi_table(self) -> None:
        """Factory with multiple tables (HSTU-style: post_id + user_id)."""
        import tempfile

        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        specs = [
            ("post_id", 5000, 128, SparseType.INT8),
            ("user_id", 2000, 64, SparseType.FP16),
        ]
        module = TurboSSDInferenceModule.from_embedding_specs(
            specs=specs,
            ssd_directory=tempfile.mkdtemp(),
            cache_hit_rate=0.80,
        )
        self.assertEqual(len(module.embedding_specs), 2)

    def test_forward_correctness(self) -> None:
        """Wrapper forward matches raw TBE output."""
        import tempfile

        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        E = 500
        D = 64
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        module = TurboSSDInferenceModule.from_embedding_specs(
            specs=[("t0", E, D, weights_ty)],
            ssd_directory=tempfile.mkdtemp(),
            cache_hit_rate=0.50,
        )

        # Initialize with known data.
        ref = torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
        torch.manual_seed(42)

        weights, scale_shift = module.tbe.split_embedding_weights()[0]
        fake_quantize_embs(
            weights,
            scale_shift,
            ref.weight.detach(),
            weights_ty,
            use_cpu=False,
        )
        copy_bytes = torch.empty([E, D_bytes], dtype=torch.uint8)
        copy_bytes[:, : unpadded_row_size_in_bytes(D, weights_ty)] = weights
        module.tbe.ssd_db.set_cuda(
            torch.arange(E, dtype=torch.int64),
            copy_bytes,
            torch.tensor([E]),
            0,
        )
        torch.cuda.synchronize()

        B, L = 4, 3
        xs = torch.randint(0, E, (B, L)).cuda()
        x = xs.view(1, B, L)
        indices, offsets = get_table_batched_offsets_from_dense(x)
        indices, offsets = indices.cuda(), offsets.cuda()

        result = module(indices, offsets)
        expected = b_indices(ref, xs)
        torch.testing.assert_close(
            result.float(),
            expected.float(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_streaming_update_through_wrapper(self) -> None:
        """streaming_update via wrapper writes to SSD."""
        import tempfile

        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        E = 200
        D = 64
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        module = TurboSSDInferenceModule.from_embedding_specs(
            specs=[("t0", E, D, weights_ty)],
            ssd_directory=tempfile.mkdtemp(),
            cache_hit_rate=0.50,
        )

        # Initialize.
        init_w = torch.randint(0, 255, (E, D_bytes), dtype=torch.uint8)
        module.tbe.ssd_db.set_cuda(
            torch.arange(E, dtype=torch.int64),
            init_w,
            torch.tensor([E]),
            0,
        )
        torch.cuda.synchronize()

        # Update via wrapper.
        idx = torch.tensor([0, 10, 199], dtype=torch.int64)
        new_w = torch.randint(0, 255, (3, D_bytes), dtype=torch.uint8)
        module.streaming_update(idx, new_w)

        out = torch.empty(3, D_bytes, dtype=torch.uint8)
        module.tbe.ssd_db.get_cuda(idx, out, torch.tensor([3]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, new_w)

    def test_load_snapshot_through_wrapper(self) -> None:
        """load_snapshot via wrapper clears cache and swaps DB."""
        import tempfile

        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        module = TurboSSDInferenceModule.from_embedding_specs(
            specs=[("t0", 200, 64, SparseType.FP32)],
            ssd_directory=tempfile.mkdtemp(),
            cache_hit_rate=0.50,
        )

        new_dir = tempfile.mkdtemp()
        module.load_snapshot(new_dir)

        cache_state = module.tbe.lxu_cache_state.cpu()
        self.assertTrue((cache_state == -1).all().item())

    def test_estimate_hbm_gb(self) -> None:
        """HBM estimation returns reasonable values."""
        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        # HSTU-style: 1.6B rows, INT8, 128-dim
        specs = [("post_id", 1_600_000_000, 128, SparseType.INT8)]

        hbm_90 = TurboSSDInferenceModule.estimate_hbm_gb(specs, 0.90)
        hbm_50 = TurboSSDInferenceModule.estimate_hbm_gb(specs, 0.50)

        # 90% hit should need more HBM than 50%.
        self.assertGreater(hbm_90, hbm_50)
        # Sanity: 90% of 1.6B rows at ~132 bytes each = ~190 GB.
        self.assertGreater(hbm_90, 100)  # should be > 100 GB
        self.assertLess(hbm_90, 300)  # should be < 300 GB

    def test_estimate_hbm_gb_multi_table(self) -> None:
        """HBM estimation with multiple tables."""
        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        specs = [
            ("post_id", 1_000_000, 128, SparseType.INT8),
            ("user_id", 500_000, 64, SparseType.FP16),
        ]
        hbm = TurboSSDInferenceModule.estimate_hbm_gb(specs, 0.90)
        self.assertGreater(hbm, 0)

    def test_hbm_budget_cap(self) -> None:
        """Cache sets are capped by HBM budget."""
        import tempfile

        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        specs = [("t0", 1_000_000, 128, SparseType.FP32)]

        # With 0.1 GB budget, cache should be much smaller than 90% hit rate.
        module = TurboSSDInferenceModule.from_embedding_specs(
            specs=specs,
            ssd_directory=tempfile.mkdtemp(),
            hbm_budget_gb=0.1,
            cache_hit_rate=0.90,
        )
        cache_slots = module.tbe.lxu_cache_state.shape[0] * ASSOC
        # 0.1 GB = ~107 million bytes. With ~512 bytes/row, ~209K rows max.
        self.assertLess(cache_slots, 1_000_000)

    def test_full_hstu_flow(self) -> None:
        """
        End-to-end HSTU-style flow: create module, load snapshot,
        populate via streaming_update, run forward, apply delta update,
        verify forward reflects the update.
        """
        import tempfile

        from fbgemm_gpu.tbe.ssd import TurboSSDInferenceModule

        E = 500
        D = 64
        B = 4
        L = 5
        weights_ty = SparseType.FP32
        D_bytes = rounded_row_size_in_bytes(D, weights_ty, 16)

        module = TurboSSDInferenceModule.from_embedding_specs(
            specs=[("post_id", E, D, weights_ty)],
            ssd_directory=tempfile.mkdtemp(),
            cache_hit_rate=0.50,
        )

        # Phase 1: Initial snapshot load.
        ref = torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
        torch.manual_seed(42)

        weights, ss = module.tbe.split_embedding_weights()[0]
        fake_quantize_embs(
            weights,
            ss,
            ref.weight.detach(),
            weights_ty,
            use_cpu=False,
        )
        copy_bytes = torch.empty([E, D_bytes], dtype=torch.uint8)
        copy_bytes[:, : unpadded_row_size_in_bytes(D, weights_ty)] = weights
        for start in range(0, E, 100):
            end = min(start + 100, E)
            module.streaming_update(
                torch.arange(start, end, dtype=torch.int64),
                copy_bytes[start:end],
            )

        # Phase 2: Forward with initial data.
        xs = torch.randint(0, E, (B, L)).cuda()
        x = xs.view(1, B, L)
        indices, offsets = get_table_batched_offsets_from_dense(x)
        indices, offsets = indices.cuda(), offsets.cuda()
        result1 = module(indices, offsets)
        expected1 = b_indices(ref, xs)
        torch.testing.assert_close(
            result1.float(),
            expected1.float(),
            atol=1e-2,
            rtol=1e-2,
        )

        # Phase 3: Delta update (simulate 45-min publish cycle).
        delta_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        ref.weight.data[delta_indices] = torch.randn(3, D).cuda()

        weights_after, ss_after = module.tbe.split_embedding_weights()[0]
        fake_quantize_embs(
            weights_after,
            ss_after,
            ref.weight.detach(),
            weights_ty,
            use_cpu=False,
        )
        delta_copy = torch.empty([E, D_bytes], dtype=torch.uint8)
        delta_copy[:, : unpadded_row_size_in_bytes(D, weights_ty)] = weights_after
        module.streaming_update(delta_indices, delta_copy[delta_indices])

        # Phase 4: Forward reflects delta update.
        xs2 = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long).cuda()
        x2 = xs2.view(1, 1, 5)
        indices2, offsets2 = get_table_batched_offsets_from_dense(x2)
        indices2, offsets2 = indices2.cuda(), offsets2.cuda()
        result2 = module(indices2, offsets2)
        expected2 = b_indices(ref, xs2.view(1, 5))
        torch.testing.assert_close(
            result2.float(),
            expected2.float(),
            atol=1e-2,
            rtol=1e-2,
        )
