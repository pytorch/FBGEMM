# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import copy
import random
import unittest

import fbgemm_gpu.ssd_split_table_batched_embeddings_ops as ssd_split_table_batched_embeddings_ops
import hypothesis.strategies as st
import numpy as np
import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_embedding_utils import (
    b_indices,
    fake_quantize_embs,
    get_table_batched_offsets_from_dense,
    round_up,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    rounded_row_size_in_bytes,
    unpadded_row_size_in_bytes,
)

from hypothesis import given, settings, Verbosity


MAX_EXAMPLES = 40


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class SSDSplitTableBatchedEmbeddingsTest(unittest.TestCase):
    def test_ssd(self) -> None:
        import tempfile

        E = int(1e4)
        D = 128
        N = 100
        indices = torch.as_tensor(np.random.choice(E, replace=False, size=(N,)))
        weights = torch.randn(N, D)
        output_weights = torch.empty_like(weights)
        count = torch.tensor([N])

        feature_table_map = list(range(1))
        emb = ssd_split_table_batched_embeddings_ops.SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=1,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
        )
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        assert (output_weights <= 0.1).all().item()
        assert (output_weights >= -0.1).all().item()

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
        weighted=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_forward(
        self, T: int, D: int, B: int, log_E: int, L: int, weighted: bool
    ) -> None:
        import tempfile

        E = int(10**log_E)
        D = D * 4
        Ds = [D] * T
        Es = [E] * T
        feature_table_map = list(range(T))
        emb = ssd_split_table_batched_embeddings_ops.SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=max(T * B * L, 1),
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
        ).cuda()

        bs = [
            torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
            for (E, D) in zip(Es, Ds)
        ]
        torch.manual_seed(42)
        xs = [torch.randint(low=0, high=e, size=(B, L)).cuda() for e in Es]
        xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        fs = (
            [b_indices(b, x) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1))
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        f = torch.cat([f.view(B, -1) for f in fs], dim=1)

        for t in range(T):
            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                bs[t].weight.cpu(),
                torch.as_tensor([E]),
                t,
            )

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)
        (indices, offsets) = get_table_batched_offsets_from_dense(x)
        fc2 = (
            emb(indices.cuda(), offsets.cuda())
            if not weighted
            else emb(indices.cuda(), offsets.cuda(), xw.contiguous().view(-1).cuda())
        )
        torch.testing.assert_close(
            fc2.float(),
            f.float(),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_backward_adagrad(
        self, T: int, D: int, B: int, log_E: int, L: int, weighted: bool
    ) -> None:
        import tempfile

        E = int(10**log_E)
        D = D * 4
        Ds = [D] * T
        Es = [E] * T
        lr = 0.5
        eps = 0.2

        feature_table_map = list(range(T))
        emb = ssd_split_table_batched_embeddings_ops.SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=max(T * B * L, 1),
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            learning_rate=lr,
            eps=eps,
            ssd_shards=2,
        ).cuda()

        bs = [
            torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
            for (E, D) in zip(Es, Ds)
        ]
        torch.manual_seed(42)
        xs = [torch.randint(low=0, high=e, size=(B, L)).cuda() for e in Es]
        xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        fs = (
            [b_indices(b, x) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1))
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]

        for t in range(T):
            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                bs[t].weight.cpu(),
                torch.as_tensor([E]),
                t,
            )

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)
        (indices, offsets) = get_table_batched_offsets_from_dense(x)
        fc2 = (
            emb(indices.cuda(), offsets.cuda())
            if not weighted
            else emb(indices.cuda(), offsets.cuda(), xw.contiguous().view(-1).cuda())
        )
        torch.testing.assert_close(
            fc2.float(),
            f.float(),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

        fc2.backward(torch.cat([go.view(B, -1) for go in gos], dim=1))

        split_optimizer_states = [s for (s,) in emb.debug_split_optimizer_states()]
        for t in range(T):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_optimizer_state = bs[t].weight.grad.float().to_dense().pow(2)
            torch.testing.assert_close(
                split_optimizer_states[t].float(),
                ref_optimizer_state.mean(dim=1),
                atol=1.0e-4,
                rtol=1.0e-4,
            )
        emb.flush()
        for t in range(T):
            torch.testing.assert_close(
                emb.debug_split_embedding_weights()[t].float().cuda(),
                torch.addcdiv(
                    bs[t].weight.float(),
                    value=-lr,
                    tensor1=bs[t].weight.grad.float().to_dense(),
                    tensor2=split_optimizer_states[t]
                    .float()
                    .sqrt_()
                    .add_(eps)
                    .view(Es[t], 1),
                ),
                atol=1.0e-4,
                rtol=1.0e-4,
            )

    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_cache(
        self, T: int, D: int, B: int, log_E: int, L: int, weighted: bool
    ) -> None:
        # T=2
        # D=2
        # B=9
        # log_E=3
        # L=14
        # weighted=False
        import tempfile

        E = int(10**log_E)
        D = D * 4
        Ds = [D] * T
        Es = [E] * T
        lr = 0.5
        eps = 0.2
        C = max(T * B * L, 1)

        feature_table_map = list(range(T))
        emb = ssd_split_table_batched_embeddings_ops.SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=C,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            learning_rate=lr,
            eps=eps,
            ssd_shards=2,
        ).cuda()

        bs = [
            torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
            for (E, D) in zip(Es, Ds)
        ]
        torch.manual_seed(42)

        for t in range(T):
            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                bs[t].weight.cpu(),
                torch.as_tensor([E]),
                t,
            )

        for i in range(10):
            xs = [torch.randint(low=0, high=e, size=(B, L)).cuda() for e in Es]
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]
            xws_acc_type = copy.deepcopy(xws)
            xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

            (indices, offsets) = get_table_batched_offsets_from_dense(x)
            (indices, offsets) = indices.cuda(), offsets.cuda()
            assert emb.timestep == i

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
            ) = torch.ops.fbgemm.ssd_cache_populate_actions(  # noqa
                linear_cache_indices,
                emb.total_hash_size,
                emb.lxu_cache_state,
                emb.timestep,
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
                assert lru_state_cpu[loc_set, loc_slot] == emb.timestep
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
                emb(indices, offsets)
                if not weighted
                else emb(indices, offsets, xw.contiguous().view(-1).cuda())
            )
            torch.testing.assert_close(
                fc2.float(),
                f.float(),
                atol=1.0e-5,
                rtol=1.0e-5,
            )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class SSDIntNBitTableBatchedEmbeddingsTest(unittest.TestCase):
    def test_nbit_ssd(self) -> None:
        import tempfile

        E = int(1e4)
        D = 128
        N = 100
        indices = torch.as_tensor(np.random.choice(E, replace=False, size=(N,)))
        weights = torch.empty(N, D, dtype=torch.uint8)
        output_weights = torch.empty_like(weights)
        count = torch.tensor([N])

        feature_table_map = list(range(1))
        emb = (
            ssd_split_table_batched_embeddings_ops.SSDIntNBitTableBatchedEmbeddingBags(
                embedding_specs=[("", E, D, SparseType.FP32)],
                feature_table_map=feature_table_map,
                ssd_storage_directory=tempfile.mkdtemp(),
                cache_sets=1,
            )
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
        emb = (
            ssd_split_table_batched_embeddings_ops.SSDIntNBitTableBatchedEmbeddingBags(
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
        )
        # # NOTE: test TorchScript-compatible!
        # emb = torch.jit.script(emb)

        bs = [
            torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
            for (E, D) in zip(Es, Ds)
        ]
        torch.manual_seed(42)
        xs = [torch.randint(low=0, high=e, size=(B, L)).cuda() for e in Es]
        xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        for t in range(T):
            (weights, scale_shift) = emb.split_embedding_weights()[t]

            if scale_shift is not None:
                (E, R) = scale_shift.shape
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
                copy_byte_tensor[
                    :, : emb.scale_bias_size_in_bytes
                ] = scale_shift  # q_scale_shift

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
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)
        (indices, offsets) = get_table_batched_offsets_from_dense(x)
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
        emb = (
            ssd_split_table_batched_embeddings_ops.SSDIntNBitTableBatchedEmbeddingBags(
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
        )
        # # NOTE: test TorchScript-compatible!
        # emb = torch.jit.script(emb)

        bs = [
            torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
            for (E, D) in zip(Es, Ds)
        ]
        torch.manual_seed(42)

        for t in range(T):
            (weights, scale_shift) = emb.split_embedding_weights()[t]

            if scale_shift is not None:
                (E, R) = scale_shift.shape
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
                copy_byte_tensor[
                    :, : emb.scale_bias_size_in_bytes
                ] = scale_shift  # q_scale_shift

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
            xws_acc_type = copy.deepcopy(xws)
            xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

            (indices, offsets) = get_table_batched_offsets_from_dense(x)
            (indices, offsets) = indices.cuda(), offsets.cuda()
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
