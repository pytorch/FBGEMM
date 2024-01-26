#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import random
import unittest

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class ReorderBatchedTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
        Dtype=st.sampled_from([torch.int32, torch.float, torch.int64]),
        broadcast_lengths=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_reorder_batched_ad_lengths(
        self,
        B: int,
        T: int,
        L: int,
        A: int,
        Dtype: torch.dtype,
        broadcast_lengths: bool,
    ) -> None:
        if broadcast_lengths:
            cat_ad_lengths = (
                torch.cat([torch.tensor([L for _ in range(T)]) for _ in range(B)], 0)
                .cuda()
                .to(Dtype)
            )
            cat_ad_lengths_broadcasted = cat_ad_lengths.tile([A])
        else:
            cat_ad_lengths = (
                torch.cat(
                    [torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0
                )
                .cuda()
                .to(Dtype)
            )
            cat_ad_lengths_broadcasted = cat_ad_lengths
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().cuda()
        num_ads_in_batch = B * A
        reordered_batched_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_lengths
        )
        torch.testing.assert_close(
            cat_ad_lengths_broadcasted, reordered_batched_ad_lengths
        )

        cat_ad_lengths_cpu = cat_ad_lengths.cpu()
        batch_offsets_cpu = batch_offsets.cpu()
        reordered_batched_ad_lengths_cpu = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths_cpu, batch_offsets_cpu, num_ads_in_batch, broadcast_lengths
        )
        torch.testing.assert_close(
            reordered_batched_ad_lengths_cpu, reordered_batched_ad_lengths.cpu()
        )

    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
        Dtype=st.sampled_from([torch.int32, torch.float, torch.int64]),
        broadcast_lengths=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_reorder_batched_ad_lengths_cpu(
        self,
        B: int,
        T: int,
        L: int,
        A: int,
        Dtype: torch.dtype,
        broadcast_lengths: bool,
    ) -> None:
        if broadcast_lengths:
            cat_ad_lengths = (
                torch.cat([torch.tensor([L for _ in range(T)]) for _ in range(B)], 0)
                .int()
                .to(Dtype)
            )
            cat_ad_lengths_broadcasted = cat_ad_lengths.tile([A])
        else:
            cat_ad_lengths = (
                torch.cat(
                    [torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0
                )
                .int()
                .to(Dtype)
            )
            cat_ad_lengths_broadcasted = cat_ad_lengths
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int()
        num_ads_in_batch = B * A
        reordered_batched_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_lengths
        )
        torch.testing.assert_close(
            cat_ad_lengths_broadcasted, reordered_batched_ad_lengths
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
        Dtype=st.sampled_from([torch.int32, torch.float, torch.int64, torch.bfloat16]),
        Itype=st.sampled_from([torch.int32, torch.int64]),
        broadcast_indices=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_reorder_batched_ad_indices(
        self,
        B: int,
        T: int,
        L: int,
        A: int,
        Dtype: torch.dtype,
        Itype: torch.dtype,
        broadcast_indices: bool,
    ) -> None:
        if broadcast_indices:
            cat_ad_indices = (
                torch.randint(
                    low=0,
                    high=100,
                    size=(B * T * L,),
                )
                .int()
                .cuda()
                .to(Dtype)
            )
            cat_ad_lengths = (
                torch.cat(
                    [torch.tensor([L for _ in range(T)]) for _ in range(B)],
                    0,
                )
                .int()
                .cuda()
            )
            cat_ad_lengths_broadcasted = cat_ad_lengths.tile([A])
        else:
            cat_ad_indices = (
                torch.randint(
                    low=0,
                    high=100,
                    size=(B * T * A * L,),
                )
                .int()
                .cuda()
                .to(Dtype)
            )
            cat_ad_lengths = (
                torch.cat(
                    [torch.tensor([L for _ in range(T * A)]) for _ in range(B)],
                    0,
                )
                .int()
                .cuda()
            )
            cat_ad_lengths_broadcasted = cat_ad_lengths
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().cuda()
        num_ads_in_batch = B * A
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
        )
        torch.testing.assert_close(cat_ad_lengths_broadcasted, reordered_cat_ad_lengths)

        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            cat_ad_lengths
        ).to(Itype)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        ).to(Itype)
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
            B * T * A * L,
        )
        torch.testing.assert_close(
            reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
            cat_ad_indices.view(B, T, 1, L).tile([1, 1, A, 1])
            if broadcast_indices
            else cat_ad_indices.view(B, T, A, L),
        )

    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
        Dtype=st.sampled_from([torch.int32, torch.float, torch.int64]),
        Itype=st.sampled_from([torch.int32, torch.int64]),
        broadcast_indices=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_cat_reorder_batched_ad_indices_cpu(
        self,
        B: int,
        T: int,
        L: int,
        A: int,
        Dtype: torch.dtype,
        Itype: torch.dtype,
        broadcast_indices: bool,
    ) -> None:
        if broadcast_indices:
            ad_indices = [
                (
                    torch.randint(
                        low=0,
                        high=100,
                        size=(T * L,),
                    )
                    .int()
                    .to(Dtype)
                )
                for _ in range(B)
            ]
            cat_ad_lengths = torch.cat(
                [torch.tensor([L for _ in range(T)]) for _ in range(B)],
                0,
            ).int()
            cat_ad_lengths_broadcasted = cat_ad_lengths.tile([A])
            cat_ad_indices = torch.cat(ad_indices, 0)
        else:
            ad_indices = [
                (
                    torch.randint(
                        low=0,
                        high=100,
                        size=(T * A * L,),
                    )
                    .int()
                    .to(Dtype)
                )
                for _ in range(B)
            ]
            cat_ad_lengths = torch.cat(
                [torch.tensor([L for _ in range(T * A)]) for _ in range(B)],
                0,
            ).int()
            cat_ad_lengths_broadcasted = cat_ad_lengths
            cat_ad_indices = torch.cat(ad_indices, 0)
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int()
        num_ads_in_batch = B * A
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
        )
        torch.testing.assert_close(cat_ad_lengths_broadcasted, reordered_cat_ad_lengths)

        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            cat_ad_lengths
        ).to(Itype)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        ).to(Itype)
        reordered_cat_ad_indices = torch.ops.fbgemm.cat_reorder_batched_ad_indices(
            cat_ad_offsets,
            ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
            B * T * A * L,
        )
        torch.testing.assert_close(
            reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
            cat_ad_indices.view(B, T, 1, L).tile([1, 1, A, 1])
            if broadcast_indices
            else cat_ad_indices.view(B, T, A, L),
        )

    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
        Dtype=st.sampled_from([torch.int32, torch.float, torch.int64]),
        Itype=st.sampled_from([torch.int32, torch.int64]),
        broadcast_indices=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_reorder_batched_ad_indices_cpu(
        self,
        B: int,
        T: int,
        L: int,
        A: int,
        Dtype: torch.dtype,
        Itype: torch.dtype,
        broadcast_indices: bool,
    ) -> None:
        if broadcast_indices:
            cat_ad_indices = (
                torch.randint(
                    low=0,
                    high=100,
                    size=(B * T * L,),
                )
                .int()
                .to(Dtype)
            )
            cat_ad_lengths = torch.cat(
                [torch.tensor([L for _ in range(T)]) for _ in range(B)],
                0,
            ).int()
            cat_ad_lengths_broadcasted = cat_ad_lengths.tile([A])
        else:
            cat_ad_indices = (
                torch.randint(
                    low=0,
                    high=100,
                    size=(B * T * A * L,),
                )
                .int()
                .to(Dtype)
            )
            cat_ad_lengths = torch.cat(
                [torch.tensor([L for _ in range(T * A)]) for _ in range(B)],
                0,
            ).int()
            cat_ad_lengths_broadcasted = cat_ad_lengths
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int()
        num_ads_in_batch = B * A
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
        )
        torch.testing.assert_close(cat_ad_lengths_broadcasted, reordered_cat_ad_lengths)
        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            cat_ad_lengths
        ).to(Itype)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        ).to(Itype)
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
            B * T * A * L,
        )
        torch.testing.assert_close(
            reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
            cat_ad_indices.view(B, T, 1, L).tile([1, 1, A, 1])
            if broadcast_indices
            else cat_ad_indices.view(B, T, A, L),
        )

    @given(
        B=st.integers(min_value=1, max_value=20),
        R=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        index_dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_reorder_batched_sequence_embeddings_cpu(
        self,
        B: int,
        R: int,
        T: int,
        L: int,
        index_dtype: torch.dtype,
    ) -> None:
        MAX_H = 1000
        DIM = 32
        ref_embeddings = torch.rand(MAX_H, DIM, dtype=torch.float, device="cpu")
        feature_lengths = [
            torch.randint(1, L, (T, random.randint(1, B + 1)), dtype=index_dtype)
            for _ in range(R)
        ]
        feature_indices = [
            torch.randint(
                0, MAX_H, (int(feature_length.sum().item()),), dtype=index_dtype
            )
            for feature_length in feature_lengths
        ]
        cat_feature_indices = torch.cat(feature_indices, 0)
        num_items_in_batch = sum(
            feature_length.size(1) for feature_length in feature_lengths
        )
        num_items_in_batch_list = torch.tensor(
            [feature_length.size(1) for feature_length in feature_lengths],
            dtype=index_dtype,
        )
        embeddings = [
            ref_embeddings[feature_indice] for feature_indice in feature_indices
        ]
        cat_sequence_embeddings = torch.cat(embeddings, 0)
        cat_sequence_embeddings_lengths = torch.cat(
            [feature_length.view(-1) for feature_length in feature_lengths], 0
        ).to(index_dtype)
        cat_sequence_embeddings_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            cat_sequence_embeddings_lengths
        )
        batch_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            num_items_in_batch_list
        )

        reordered_cat_sequence_embeddings_lengths = (
            torch.ops.fbgemm.reorder_batched_ad_lengths(
                cat_sequence_embeddings_lengths, batch_offsets, num_items_in_batch
            )
        )
        reordered_cat_sequence_embeddings_offsets = (
            torch.ops.fbgemm.asynchronous_complete_cumsum(
                reordered_cat_sequence_embeddings_lengths
            )
        )
        reordered_cat_sequence_embeddings = (
            torch.ops.fbgemm.reorder_batched_sequence_embeddings(
                cat_sequence_embeddings_offsets,
                cat_sequence_embeddings,
                reordered_cat_sequence_embeddings_offsets,
                batch_offsets,
                num_items_in_batch,
            )
        )
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_sequence_embeddings_offsets,
            cat_feature_indices,
            reordered_cat_sequence_embeddings_offsets,
            batch_offsets.int(),
            num_items_in_batch,
        )
        reordered_sequence_embedding_from_indices = ref_embeddings[
            reordered_cat_ad_indices
        ]
        torch.testing.assert_close(
            reordered_sequence_embedding_from_indices, reordered_cat_sequence_embeddings
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=1, max_value=20),
        R=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        index_dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_reorder_batched_sequence_embeddings(
        self,
        B: int,
        R: int,
        T: int,
        L: int,
        index_dtype: torch.dtype,
    ) -> None:
        MAX_H = 1000
        DIM = 32
        device = torch.device("cuda")
        ref_embeddings = torch.rand(MAX_H, DIM, dtype=torch.float, device=device)
        feature_lengths = [
            torch.randint(
                1, L, (T, random.randint(1, B + 1)), dtype=index_dtype, device=device
            )
            for _ in range(R)
        ]
        feature_indices = [
            torch.randint(
                0,
                MAX_H,
                (int(feature_length.sum().item()),),
                dtype=index_dtype,
                device=device,
            )
            for feature_length in feature_lengths
        ]
        cat_feature_indices = torch.cat(feature_indices, 0)
        num_items_in_batch = sum(
            feature_length.size(1) for feature_length in feature_lengths
        )
        num_items_in_batch_list = torch.tensor(
            [feature_length.size(1) for feature_length in feature_lengths],
            dtype=index_dtype,
            device=device,
        )
        embeddings = [
            ref_embeddings[feature_indice] for feature_indice in feature_indices
        ]
        cat_sequence_embeddings = torch.cat(embeddings, 0)
        cat_sequence_embeddings_lengths = torch.cat(
            [feature_length.view(-1) for feature_length in feature_lengths], 0
        ).to(index_dtype)
        cat_sequence_embeddings_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            cat_sequence_embeddings_lengths
        )
        batch_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            num_items_in_batch_list
        )

        reordered_cat_sequence_embeddings_lengths = (
            torch.ops.fbgemm.reorder_batched_ad_lengths(
                cat_sequence_embeddings_lengths, batch_offsets.int(), num_items_in_batch
            )
        )
        reordered_cat_sequence_embeddings_offsets = (
            torch.ops.fbgemm.asynchronous_complete_cumsum(
                reordered_cat_sequence_embeddings_lengths
            )
        )
        reordered_cat_sequence_embeddings = (
            torch.ops.fbgemm.reorder_batched_sequence_embeddings(
                cat_sequence_embeddings_offsets,
                cat_sequence_embeddings,
                reordered_cat_sequence_embeddings_offsets,
                batch_offsets,
                num_items_in_batch,
            )
        )
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_sequence_embeddings_offsets,
            cat_feature_indices,
            reordered_cat_sequence_embeddings_offsets,
            batch_offsets.int(),
            num_items_in_batch,
        )
        reordered_sequence_embedding_from_indices = ref_embeddings[
            reordered_cat_ad_indices
        ]
        torch.testing.assert_close(
            reordered_sequence_embedding_from_indices, reordered_cat_sequence_embeddings
        )


extend_test_class(ReorderBatchedTest)

if __name__ == "__main__":
    unittest.main()
