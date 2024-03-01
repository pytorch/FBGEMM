#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest
from typing import Callable, Dict, List

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_embedding_utils import get_table_batched_offsets_from_dense
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import DEFAULT_ASSOC
from hypothesis import given, settings, Verbosity

from .. import common  # noqa E402
from ..common import MAX_EXAMPLES, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests


VERBOSITY: Verbosity = Verbosity.verbose


# pyre-ignore
additional_decorators: Dict[str, List[Callable]] = {
    "test_faketensor__test_nbit_uvm_cache_stats": [
        unittest.skip("very slow"),
    ],
    "test_faketensor__test_nbit_direct_mapped_uvm_cache_stats": [
        unittest.skip("very slow"),
    ],
}


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class NBitCacheTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        L=st.integers(min_value=0, max_value=16),
        H=st.integers(min_value=512, max_value=1024),
        S=st.integers(min_value=0, max_value=128),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_cache_update_function(self, L: int, H: int, S: int) -> None:
        # Generate synthetic data
        linear_cache_indices_cpu = torch.randint(L, H, (S,))
        lxu_cache_locations_cpu = torch.clone(linear_cache_indices_cpu)

        indices = [True if np.random.rand() < 0.5 else False for _ in range(S)]
        lxu_cache_locations_cpu[indices] = -1

        cache_miss_ids = torch.clone(linear_cache_indices_cpu)
        cache_miss_ids[lxu_cache_locations_cpu != -1] = -2

        # Calculate the correct output
        unique_cache_miss_ids = torch.unique(cache_miss_ids)
        expect_out = sum(unique_cache_miss_ids >= 0)
        linear_cache_indices = linear_cache_indices_cpu.to(torch.int32).cuda()
        lxu_cache_locations = lxu_cache_locations_cpu.to(torch.int32).cuda()
        expected_unique_access = len(torch.unique(linear_cache_indices_cpu))
        expected_total_access = len(linear_cache_indices_cpu)

        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            record_cache_metrics=RecordCacheMetrics(True, False),
        )
        cc.fill_random_weights()

        cc._update_cache_miss_counter(lxu_cache_locations, linear_cache_indices)
        (
            cache_miss_forward_count,
            unique_cache_miss_count,
            unique_access_count,
            total_access_count,
        ) = cc.get_cache_miss_counter().cpu()

        self.assertEqual(unique_cache_miss_count, expect_out)
        self.assertLessEqual(cache_miss_forward_count, unique_cache_miss_count)
        self.assertEqual(unique_access_count, expected_unique_access)
        self.assertEqual(total_access_count, expected_total_access)

    @unittest.skipIf(*gpu_unavailable)
    @given(N=st.integers(min_value=1, max_value=8))
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_cache_miss_counter(self, N: int) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            record_cache_metrics=RecordCacheMetrics(True, True),
        )
        cc.fill_random_weights()

        # Create fake input data and the target output
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]]).cuda()
        x2 = torch.Tensor([[[2], [1]], [[3], [4]]]).cuda()
        x3 = torch.Tensor([[[5], [6]], [[7], [8]]]).cuda()

        xs = [x1, x2, x3]
        target_counter_list = [[1, 3], [2, 4], [3, 8]]
        target_tablewise_cache_miss_list = [[1, 2], [2, 2], [4, 4]]
        for x, t_counter, t_tablewise_cache_miss in zip(
            xs, target_counter_list, target_tablewise_cache_miss_list
        ):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc(indices.int(), offsets.int())
                (
                    cache_miss_forward_count,
                    unique_cache_miss_count,
                    _,
                    _,
                ) = cc.get_cache_miss_counter().cpu()
                tablewise_cache_miss = cc.get_table_wise_cache_miss().cpu()
                self.assertEqual(cache_miss_forward_count, t_counter[0])
                self.assertEqual(unique_cache_miss_count, t_counter[1])
                for i in range(len(tablewise_cache_miss)):
                    self.assertEqual(tablewise_cache_miss[i], t_tablewise_cache_miss[i])

    @unittest.skipIf(*gpu_unavailable)
    @given(
        N=st.integers(min_value=1, max_value=8),
        dtype=st.sampled_from([SparseType.INT8, SparseType.INT4, SparseType.INT2]),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_uvm_cache_stats(self, N: int, dtype: SparseType) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    dtype,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
        )
        cc.fill_random_weights()

        # Create fake input data and the target output
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]]).cuda()
        x2 = torch.Tensor([[[2], [1]], [[3], [4]]]).cuda()
        x3 = torch.Tensor([[[5], [6]], [[7], [8]]]).cuda()

        xs = [x1, x2, x3]
        # num_unique_indices, num_unique_misses
        # note that these are cumulative over calls; and also "unique" is per batch.
        target_counter_list = [[3, 3], [4, 4], [4, 8]]
        num_calls_expected = 0
        num_indices_expcted = 0
        num_unique_indices_expected = 0
        for x, t_counter in zip(xs, target_counter_list):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                num_calls_expected = num_calls_expected + 1
                num_indices_expcted = num_indices_expcted + len(indices)
                cc(indices.int(), offsets.int())
                (
                    num_calls,
                    num_indices,
                    num_unique_indices,
                    num_unique_misses,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc.get_uvm_cache_stats().cpu()
                # Note num_unique_indices is cumulative stats.
                num_unique_indices_expected = num_unique_indices_expected + t_counter[0]
                self.assertEqual(num_calls, num_calls_expected)
                self.assertEqual(num_indices, num_indices_expcted)
                self.assertEqual(num_unique_indices, num_unique_indices_expected)
                self.assertEqual(num_unique_misses, t_counter[1])
                self.assertEqual(num_conflict_unique_miss, 0)
                self.assertEqual(num_conflict_miss, 0)

        T = 1  # for simplicity
        Ds = [D] * T
        Es = [E] * T
        cc1 = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
            cache_sets=1,  # Only one set.
        )
        cc1.fill_random_weights()

        associativty = DEFAULT_ASSOC  # 32 for NVidia / 64 for AMD.
        repetition = 17
        indices1 = torch.Tensor(
            [[list(range(0, associativty))] * repetition]
        ).cuda()  # 0, 1, ..., 31.
        indices2 = torch.Tensor(
            [[list(range(0, associativty + 1))] * repetition]
        ).cuda()  # 0, 1, ..., 31, 32.
        indices3 = torch.Tensor(
            [[list(range(0, associativty + 10))] * repetition]
        ).cuda()  # 0, 1, ..., 31, 32, ..., 41.

        # num_conflict_unique_miss, num_conflict_miss
        expected = [[0, 0], [1, 17], [10, 170]]

        for x, e in zip((indices1, indices2, indices3), expected):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc1(indices.int(), offsets.int())
                (
                    _,
                    _,
                    _,
                    _,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc1.get_uvm_cache_stats().cpu()
                self.assertEqual(num_conflict_unique_miss, e[0])
                self.assertEqual(num_conflict_miss, e[1])
                cc1.reset_uvm_cache_stats()

    @unittest.skipIf(*gpu_unavailable)
    @given(
        N=st.integers(min_value=1, max_value=8),
        dtype=st.sampled_from([SparseType.INT8, SparseType.INT4, SparseType.INT2]),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_direct_mapped_uvm_cache_stats(
        self, N: int, dtype: SparseType
    ) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    dtype,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
            cache_assoc=1,  # Direct Mapped
        )
        cc.fill_random_weights()

        # Create fake input data and the target output
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]]).cuda()
        x2 = torch.Tensor([[[2], [1]], [[3], [4]]]).cuda()
        x3 = torch.Tensor([[[5], [6]], [[7], [8]]]).cuda()

        xs = [x1, x2, x3]
        # num_unique_indices, num_unique_misses
        # note that these are cumulative over calls; and also "unique" is per batch.
        target_counter_list = [[3, 3], [4, 4], [4, 8]]
        num_calls_expected = 0
        num_indices_expcted = 0
        num_unique_indices_expected = 0
        for x, t_counter in zip(xs, target_counter_list):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                num_calls_expected = num_calls_expected + 1
                num_indices_expcted = num_indices_expcted + len(indices)
                cc(indices.int(), offsets.int())
                (
                    num_calls,
                    num_indices,
                    num_unique_indices,
                    num_unique_misses,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc.get_uvm_cache_stats().cpu()
                # Note num_unique_indices is cumulative stats.
                num_unique_indices_expected = num_unique_indices_expected + t_counter[0]
                self.assertEqual(num_calls, num_calls_expected)
                self.assertEqual(num_indices, num_indices_expcted)
                self.assertEqual(num_unique_indices, 0)  # N/A for Direct Mapped
                self.assertEqual(num_unique_misses, 0)  # N/A for Direct Mapped
                self.assertEqual(
                    num_conflict_unique_miss, t_counter[1]
                )  # number of actually inserted rows for Direct Mapped
                self.assertEqual(num_conflict_miss, 0)

        T = 1  # for simplicity
        Ds = [D] * T
        Es = [E] * T
        cc1 = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
            cache_sets=1,  # Only one set.
            cache_assoc=1,  # Direct Mapped
        )
        cc1.fill_random_weights()

        associativty = 1  # Direct-Mapped
        repetition = 17
        indices1 = torch.Tensor(
            [[list(range(0, associativty))] * repetition]
        ).cuda()  # no conflict miss
        indices2 = torch.Tensor(
            [[list(range(0, associativty + 1))] * repetition]
        ).cuda()  # 1 * 17 conflict miss per request
        indices3 = torch.Tensor(
            [[list(range(0, associativty + 10))] * repetition]
        ).cuda()  # 10 * 17 conflict misses per request

        # num_conflict_unique_miss, num_conflict_miss
        expected = [[1, 0], [1, 17], [1, 170]]

        accum_num_conflict_miss = 0
        for x, e in zip((indices1, indices2, indices3), expected):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc1(indices.int(), offsets.int())
                (
                    _,
                    _,
                    _,
                    _,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc1.get_uvm_cache_stats().cpu()
                # for DM this represents number of actually inserted rows
                self.assertEqual(num_conflict_unique_miss, e[0])
                accum_num_conflict_miss += e[1]
                self.assertEqual(num_conflict_miss, accum_num_conflict_miss)


if __name__ == "__main__":
    unittest.main()
