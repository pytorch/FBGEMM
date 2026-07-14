#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import random
import unittest
from collections.abc import Callable
from typing import Any

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_memory_lt_gb, gpu_unavailable, optests, skipIfRocm
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import (
        gpu_memory_lt_gb,
        gpu_unavailable,
        optests,
        skipIfRocm,
    )


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
    @settings(verbosity=Verbosity.normal, max_examples=20, deadline=None)
    @optests.dontGenerateOpCheckTests(
        "GPU-only test; opcheck variants only skip on CPU samples; op covered by *_cpu twin (T191384137)"
    )
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
    @settings(verbosity=Verbosity.normal, max_examples=40, deadline=None)
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

    @skipIfRocm
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
    @settings(verbosity=Verbosity.normal, max_examples=20, deadline=None)
    @optests.dontGenerateOpCheckTests(
        "GPU-only test; opcheck variants only skip on CPU samples; op covered by *_cpu twin (T191384137)"
    )
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
            (
                cat_ad_indices.view(B, T, 1, L).tile([1, 1, A, 1])
                if broadcast_indices
                else cat_ad_indices.view(B, T, A, L)
            ),
        )

    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
        Dtype=st.sampled_from([torch.int32, torch.float, torch.int64]),
        Itype=st.sampled_from([torch.int32, torch.int64]),
        broadcast_indices=st.booleans(),
        max_batch_size=st.integers(min_value=-2, max_value=5),
    )
    @settings(verbosity=Verbosity.normal, max_examples=20, deadline=None)
    def test_cat_reorder_batched_ad_indices_cpu(
        self,
        B: int,
        T: int,
        L: int,
        A: int,
        Dtype: torch.dtype,
        Itype: torch.dtype,
        broadcast_indices: bool,
        max_batch_size: int,
    ) -> None:
        num_ads_in_batch = B * A
        max_batch_size = (
            (max_batch_size + num_ads_in_batch)
            if max_batch_size > 0 and not broadcast_indices
            else 0
        )
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
            expected_reordered_ad_lengths = cat_ad_lengths_broadcasted
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
            expected_reordered_ad_lengths = torch.cat(
                [
                    torch.tensor(
                        [L for _ in range(num_ads_in_batch)]
                        + (
                            [0 for _ in range(max_batch_size - num_ads_in_batch)]
                            if max_batch_size > 0
                            else []
                        )
                    )
                    for _ in range(T)
                ],
                0,
            ).int()
            cat_ad_indices = torch.cat(ad_indices, 0)
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int()
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
            max_batch_size,
        )
        torch.testing.assert_close(
            expected_reordered_ad_lengths, reordered_cat_ad_lengths
        )

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
            max_batch_size=max_batch_size,
        )
        torch.testing.assert_close(
            reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
            (
                cat_ad_indices.view(B, T, 1, L).tile([1, 1, A, 1])
                if broadcast_indices
                else cat_ad_indices.view(B, T, A, L)
            ),
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
    @settings(verbosity=Verbosity.normal, max_examples=40, deadline=None)
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
            (
                cat_ad_indices.view(B, T, 1, L).tile([1, 1, A, 1])
                if broadcast_indices
                else cat_ad_indices.view(B, T, A, L)
            ),
        )

    @given(
        B=st.integers(min_value=1, max_value=20),
        R=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        index_dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.normal, max_examples=40, deadline=None)
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

    @skipIfRocm
    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=1, max_value=20),
        R=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        index_dtype=st.sampled_from([torch.int32, torch.int64]),
        emb_dtype=st.sampled_from(
            [torch.float32, torch.uint8, torch.bfloat16, torch.float16]
        ),
    )
    @settings(verbosity=Verbosity.normal, max_examples=40, deadline=None)
    @optests.dontGenerateOpCheckTests(
        "GPU-only test; opcheck variants only skip on CPU samples; op covered by *_cpu twin (T191384137)"
    )
    def test_reorder_batched_sequence_embeddings(
        self,
        B: int,
        R: int,
        T: int,
        L: int,
        index_dtype: torch.dtype,
        emb_dtype: torch.dtype,
    ) -> None:
        MAX_H = 1000
        DIM = 32
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        ref_embeddings = torch.rand(MAX_H, DIM, dtype=emb_dtype, device=device)
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

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(4))
    def test_reorder_batched_ad_lengths_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in reorder_batched_ad_lengths_kernel
        and verifies output correctness against the CPU dispatch.

        With block dim3(32, 32) (1024 threads per block), grid = (B*T+31)/32.
        Total threads ~= B*T * 32. For B*T > 2**27, total threads exceed the
        HIP 2**32 limit, causing FBGEMM_LAUNCH_KERNEL ->
        KernelLauncher::checkThreadCountNotExceeded to TORCH_CHECK-fail on
        ROCm pre-fix.

        ``cat_ad_lengths`` is sparse: zero everywhere except sentinel
        non-zero entries at start / middle / end so any "kernel
        addressed wrong row" bug surfaces in the assertion below.
        """
        T = 1
        B = (1 << 27) + 1  # B*T > 2**27
        num_ads_in_batch = B

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # Sparse non-zero lengths at sentinel positions.
        cat_ad_lengths_cpu = torch.zeros(T * B, dtype=torch.int32)
        cat_ad_lengths_cpu[0] = 1
        cat_ad_lengths_cpu[(T * B) // 2] = 2
        cat_ad_lengths_cpu[T * B - 1] = 3
        batch_offsets_cpu = torch.arange(B + 1, dtype=torch.int32)

        # CPU reference oracle.
        reordered_cpu = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths_cpu, batch_offsets_cpu, num_ads_in_batch, False, -1
        )

        # GPU op under test. Pre-fix, this launch trips
        # KernelLauncher::checkThreadCountNotExceeded on ROCm.
        reordered_gpu = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths_cpu.to(device),
            batch_offsets_cpu.to(device),
            num_ads_in_batch,
            False,
            -1,
        )

        torch.testing.assert_close(reordered_gpu.cpu(), reordered_cpu)

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(4))
    def test_reorder_batched_ad_indices_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in
        reorder_batched_ad_indices_kernel_vec and verifies output
        correctness against the CPU dispatch. AMD branch uses
        dim3(32, NUM_WARPS=4) blocks; total threads ~= B*T * 32.

        ``cat_ad_offsets`` is sparse: most segments have length zero with
        sentinel non-zero entries at start / middle / end, so the kernel
        actually reorders a small number of indices while the launch
        grid still trips the cap pre-fix.
        """
        T = 1
        B = (1 << 27) + 1
        num_ads_in_batch = B

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # Sparse non-zero lengths at sentinel positions, total = 6.
        lengths_cpu = torch.zeros(T * B, dtype=torch.int32)
        lengths_cpu[0] = 1
        lengths_cpu[(T * B) // 2] = 2
        lengths_cpu[T * B - 1] = 3
        # Cumulative offsets.
        cat_ad_offsets_cpu = torch.zeros(T * B + 1, dtype=torch.int32)
        cat_ad_offsets_cpu[1:] = torch.cumsum(lengths_cpu, dim=0).to(torch.int32)
        # Distinct indices so any out-of-order copy bug surfaces.
        cat_ad_indices_cpu = torch.tensor([10, 20, 21, 30, 31, 32], dtype=torch.int32)
        batch_offsets_cpu = torch.arange(B + 1, dtype=torch.int32)
        # reordered_cat_ad_offsets matches cat_ad_offsets when num_ads_in_batch == B.
        reordered_cat_ad_offsets_cpu = cat_ad_offsets_cpu.clone()

        # CPU reference oracle.
        reordered_cpu = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets_cpu,
            cat_ad_indices_cpu,
            reordered_cat_ad_offsets_cpu,
            batch_offsets_cpu,
            num_ads_in_batch,
            False,  # broadcast_indices
            int(cat_ad_indices_cpu.numel()),
        )

        # GPU op under test.
        reordered_gpu = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets_cpu.to(device),
            cat_ad_indices_cpu.to(device),
            reordered_cat_ad_offsets_cpu.to(device),
            batch_offsets_cpu.to(device),
            num_ads_in_batch,
            False,
            int(cat_ad_indices_cpu.numel()),
        )

        torch.testing.assert_close(reordered_gpu.cpu(), reordered_cpu)

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(4))
    def test_reorder_batched_sequence_embeddings_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in
        reorder_batched_sequence_embeddings_kernel and verifies output
        correctness against the CPU dispatch.

        ``cat_sequence_embeddings_offsets`` is sparse with sentinel
        non-zero segments at start / middle / end; the embedding values
        are distinct per row so any "kernel addressed wrong row" bug
        surfaces.
        """
        T = 1
        B = (1 << 27) + 1
        num_items_in_batch = B

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # Sparse non-zero lengths, total = 6.
        lengths_cpu = torch.zeros(T * B, dtype=torch.int32)
        lengths_cpu[0] = 1
        lengths_cpu[(T * B) // 2] = 2
        lengths_cpu[T * B - 1] = 3
        cat_seq_offsets_cpu = torch.zeros(T * B + 1, dtype=torch.int32)
        cat_seq_offsets_cpu[1:] = torch.cumsum(lengths_cpu, dim=0).to(torch.int32)
        # Distinct embedding values, shape (6, D).
        cat_seq_emb_cpu = torch.tensor(
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=torch.float32
        )
        reordered_cat_seq_offsets_cpu = cat_seq_offsets_cpu.clone()
        batch_offsets_cpu = torch.arange(B + 1, dtype=torch.int32)

        # CPU reference oracle.
        reordered_cpu = torch.ops.fbgemm.reorder_batched_sequence_embeddings(
            cat_seq_offsets_cpu,
            cat_seq_emb_cpu,
            reordered_cat_seq_offsets_cpu,
            batch_offsets_cpu,
            num_items_in_batch,
        )

        # GPU op under test.
        reordered_gpu = torch.ops.fbgemm.reorder_batched_sequence_embeddings(
            cat_seq_offsets_cpu.to(device),
            cat_seq_emb_cpu.to(device),
            reordered_cat_seq_offsets_cpu.to(device),
            batch_offsets_cpu.to(device),
            num_items_in_batch,
        )

        torch.testing.assert_close(reordered_gpu.cpu(), reordered_cpu)


# Large-grid tests allocate B = 2^27 elements, which causes OOM/timeout
# during opcheck tracing (faketensor, aot_dispatch_dynamic).  Skip them.
additional_decorators: dict[str, list[Callable[..., Any]]] = {
    "test_faketensor__test_reorder_batched_ad_lengths_large_grid": [
        unittest.skip("large-grid test OOMs under faketensor tracing")
    ],
    "test_aot_dispatch_dynamic__test_reorder_batched_ad_lengths_large_grid": [
        unittest.skip("large-grid test OOMs under aot_dispatch tracing")
    ],
    "test_faketensor__test_reorder_batched_ad_indices_large_grid": [
        unittest.skip("large-grid test OOMs under faketensor tracing")
    ],
    "test_aot_dispatch_dynamic__test_reorder_batched_ad_indices_large_grid": [
        unittest.skip("large-grid test OOMs under aot_dispatch tracing")
    ],
    "test_faketensor__test_reorder_batched_sequence_embeddings_large_grid": [
        unittest.skip("large-grid test OOMs under faketensor tracing")
    ],
    "test_aot_dispatch_dynamic__test_reorder_batched_sequence_embeddings_large_grid": [
        unittest.skip("large-grid test OOMs under aot_dispatch tracing")
    ],
}

extend_test_class(ReorderBatchedTest, additional_decorators)

if __name__ == "__main__":
    unittest.main()
