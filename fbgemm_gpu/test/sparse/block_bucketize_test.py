#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import unittest
from typing import Optional, Type

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class


# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available
else:
    from fbgemm_gpu.test.test_utils import gpu_available


def unbucketize_indices_value(
    bucketized_indices: torch.Tensor,
    bucketized_lengths: torch.Tensor,
    block_sizes: torch.Tensor,
    W: int,
    B: int,
) -> torch.Tensor:
    block_size_expand = torch.empty_like(bucketized_indices)
    bucket_expand = torch.empty_like(bucketized_indices)
    T = block_sizes.size()[0]
    offset = 0
    for w in range(W):
        for t in range(T):
            for b in range(B):
                seg_length = bucketized_lengths[w * T * B + t * B + b]
                for i in range(offset, offset + seg_length):
                    block_size_expand[i] = block_sizes[t]
                    bucket_expand[i] = w
                offset += seg_length
    return bucket_expand * block_size_expand + bucketized_indices


class BlockBucketizeTest(unittest.TestCase):
    @given(
        long_indices=st.booleans(),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_long_indices(
        self, long_indices: bool, use_cpu: bool
    ) -> None:
        bucketize_pos = False
        sequence = False
        index_type = torch.long if long_indices else torch.int

        # 3 GPUs
        my_size = 3
        block_sizes = torch.tensor([3, 4, 5], dtype=index_type)

        if not long_indices:
            lengths = torch.tensor([0, 3, 2, 0, 1, 4], dtype=index_type)
            indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=index_type)
            new_lengths_ref = torch.tensor(
                [0, 2, 0, 0, 0, 0, 0, 1, 2, 0, 1, 3, 0, 0, 0, 0, 0, 1], dtype=index_type
            )
            new_indices_ref = torch.tensor(
                [1, 2, 0, 0, 1, 1, 2, 3, 4, 0], dtype=index_type
            )
        else:
            lengths = torch.tensor([0, 3, 2, 0, 1, 4], dtype=index_type)
            # Test long and negative indices: -8 will be casted to 18446644015555759292
            indices = torch.tensor(
                [1, 2, 3, 100061827127359, 5, 6, 7, -8, 100058153792324, 10],
                dtype=index_type,
            )
            new_lengths_ref = torch.tensor(
                [0, 2, 0, 0, 0, 0, 0, 1, 2, 0, 1, 1, 0, 0, 0, 0, 0, 3], dtype=index_type
            )
            new_indices_ref = torch.tensor(
                [
                    1,
                    2,
                    0,
                    33353942375786,  # 100061827127359/3 = 33353942375786
                    1,
                    1,
                    2,
                    6148914691236517202,  # -8 cast to 18446644015555759292, 18446644015555759292 /3 = 6148914691236517202
                    33352717930774,  # 100058153792324/3 = 33352717930774
                    0,
                ],
                dtype=index_type,
            )

        (
            new_lengths_cpu,
            new_indices_cpu,
            new_weights_cpu,
            new_pos_cpu,
            unbucketize_permute_cpu,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            None,
        )
        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref)

        if not use_cpu:
            (
                new_lengths_gpu,
                new_indices_gpu,
                new_weights_gpu,
                new_pos_gpu,
                unbucketize_permute_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                None,
            )

            torch.testing.assert_close(new_lengths_gpu.cpu(), new_lengths_ref)
            torch.testing.assert_close(new_indices_gpu.cpu(), new_indices_ref)
            torch.testing.assert_close(new_lengths_gpu.cpu(), new_lengths_cpu)
            torch.testing.assert_close(new_indices_gpu.cpu(), new_indices_cpu)

    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        has_weight=st.booleans(),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features(
        self,
        index_type: Type[torch.dtype],
        has_weight: bool,
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        B = 2
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
        )
        weights = (
            torch.tensor(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ],
                dtype=torch.float,
            )
            if has_weight
            else None
        )
        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        new_lengths_ref = torch.tensor(
            [0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 1, 2, 1, 3, 2, 1],
            # pyre-ignore [6]
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [3, 4, 11, 1, 11, 0, 13, 14, 0, 1, 2, 3, 2, 0, 0],
            # pyre-ignore [6]
            dtype=index_type,
        )
        new_weights_ref = torch.tensor(
            [
                1.0,
                2.0,
                4.0,
                7.0,
                12.0,
                3.0,
                5.0,
                6.0,
                8.0,
                9.0,
                10.0,
                11.0,
                13.0,
                14.0,
                15.0,
            ],
            dtype=torch.float,
        )
        new_pos_ref = torch.tensor(
            [0, 1, 0, 0, 0, 0, 1, 2, 1, 0, 1, 2, 1, 2, 0],
            # pyre-ignore [6]
            dtype=index_type,
        )
        (
            new_lengths_cpu,
            new_indices_cpu,
            new_weights_cpu,
            new_pos_cpu,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths, indices, bucketize_pos, sequence, block_sizes, my_size, weights
        )
        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref, rtol=0, atol=0)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref, rtol=0, atol=0)
        if has_weight:
            torch.testing.assert_close(new_weights_cpu, new_weights_ref)
        if bucketize_pos:
            torch.testing.assert_close(new_pos_cpu, new_pos_ref)
        if sequence:
            value_unbucketized_indices = unbucketize_indices_value(
                new_indices_cpu, new_lengths_cpu, block_sizes, my_size, B
            )
            unbucketized_indices = torch.index_select(
                value_unbucketized_indices, 0, unbucketize_permute
            )
            torch.testing.assert_close(unbucketized_indices, indices, rtol=0, atol=0)

        if gpu_available:
            (
                new_lengths_gpu,
                new_indices_gpu,
                new_weights_gpu,
                new_pos_gpu,
                unbucketize_permute_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                # pyre-fixme[16]: `Optional` has no attribute `cuda`.
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )
            torch.testing.assert_close(
                new_indices_gpu.cpu(), new_indices_ref, rtol=0, atol=0
            )
            if has_weight:
                torch.testing.assert_close(new_weights_gpu.cpu(), new_weights_cpu)
            if bucketize_pos:
                torch.testing.assert_close(new_pos_gpu.cpu(), new_pos_cpu)
            if sequence:
                value_unbucketized_indices = unbucketize_indices_value(
                    new_indices_gpu.cpu(),
                    new_lengths_gpu.cpu(),
                    block_sizes,
                    my_size,
                    B,
                )
                unbucketized_indices = torch.index_select(
                    value_unbucketized_indices, 0, unbucketize_permute_gpu.cpu()
                )
                torch.testing.assert_close(
                    unbucketized_indices, indices, rtol=0, atol=0
                )

    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        has_weight=st.booleans(),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_with_variable_batch_sizes(
        self,
        index_type: Optional[torch.dtype],
        has_weight: bool,
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        lengths = torch.tensor([2, 1, 1, 2, 0, 2], dtype=index_type)
        indices = torch.tensor(
            [1, 8, 5, 6, 7, 8, 8, 4],
            dtype=index_type,
        )
        batch_sizes = torch.tensor([3, 1, 2], dtype=index_type)
        weights = (
            torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                dtype=torch.float,
            )
            if has_weight
            else None
        )

        block_sizes = torch.tensor([5, 10, 8], dtype=index_type)
        my_size = 2
        max_B = batch_sizes.max().item()

        new_lengths_ref = torch.tensor(
            [1, 0, 0, 2, 0, 1, 1, 1, 1, 0, 0, 1],
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [1, 7, 8, 4, 3, 0, 1, 0],
            dtype=index_type,
        )

        (
            new_lengths_cpu,
            new_indices_cpu,
            new_weights_cpu,
            new_pos_cpu,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights,
            batch_sizes,
        )
        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref, rtol=0, atol=0)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref, rtol=0, atol=0)

        if gpu_available:
            (
                new_lengths_gpu,
                new_indices_gpu,
                new_weights_gpu,
                new_pos_gpu,
                unbucketize_permute_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda() if weights is not None else None,
                batch_sizes.cuda(),
                max_B,
            )

            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )
            torch.testing.assert_close(
                new_indices_gpu.cpu(), new_indices_ref, rtol=0, atol=0
            )

    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        has_weight=st.booleans(),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_with_block_bucketize_pos(
        self,
        index_type: Optional[torch.dtype],
        has_weight: bool,
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        """
        Test variable bucket size for block bucketize_sparse features for RW sharding.
        E.g. Given bucket_sizes_pos as [[0,5,15], [0,10,13]]
        For batch 0, indices in [0,5) will be assigned to bucket 0, indices in [5,15) will be assigned to bucket 1.
        For batch 1, indices in [0,10) will be assigned to bucket 0, indices in [10,13) will be assigned to bucket 1.
        The new index will be original index - bucket_sizes_pos[new_bucket_id-1]
        i.e. for batch = 0, index = 12, it will be assigned to bucket 1 and the new index is 12 - 5 = 7.
        """
        # For the following test case, we have
        # batch 0: 2 (1,7), 1 (2), 1 (6)
        # 1: bucket 0, new_idx 1
        # 7: bucket 1, new_idx 5
        # 2: bucket 1, new_idx 0
        # 6: bucket 1, new_idx 4

        # batch 1: 2 (7,8)
        # 7: bucket 1, new_idx 2
        # 8: bucket 1, new_idx 3

        # batch 2: 0, 2 (8,4)
        # 8: bucket 1, new_idx 1
        # 4: bucket 0, new_idx 4

        # new_lengths for 0: 1, 0, 0, 0, 0, 1
        # new_indices for 0: 1|  |  |  |  | 4
        # new_lengths for 1: 1, 1, 1, 2,   0, 1
        # new_indices for 1: 5| 0| 4| 2,3|   |1
        lengths = torch.tensor([2, 1, 1, 2, 0, 2], dtype=index_type)
        indices = torch.tensor(
            [1, 7, 2, 6, 7, 8, 8, 4],
            dtype=index_type,
        )
        batch_sizes = torch.tensor([3, 1, 2], dtype=index_type)
        weights = (
            torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                dtype=torch.float,
            )
            if has_weight
            else None
        )

        block_sizes = torch.tensor([5, 10, 8], dtype=index_type)
        my_size = 2
        max_B = batch_sizes.max().item()  # unused

        block_bucketize_pos = [
            torch.tensor([0, 2, 8], dtype=index_type),
            torch.tensor([0, 5, 10], dtype=index_type),
            torch.tensor([0, 7, 12], dtype=index_type),
        ]

        new_lengths_ref = torch.tensor(
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1],
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [1, 4, 5, 0, 4, 2, 3, 1],
            dtype=index_type,
        )
        new_weights_ref = torch.tensor(
            [
                1.0,
                8.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
            ],
            dtype=torch.float,
        )
        (
            new_lengths_cpu,
            new_indices_cpu,
            new_weights_cpu,
            new_pos_cpu,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights,
            batch_sizes,
            max_B,
            block_bucketize_pos,
        )
        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref, rtol=0, atol=0)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref, rtol=0, atol=0)
        if has_weight:
            torch.testing.assert_close(new_weights_cpu, new_weights_ref)

        if gpu_available:
            block_bucketize_pos = [
                torch.tensor([0, 2, 8], dtype=index_type, device="cuda"),
                torch.tensor([0, 5, 10], dtype=index_type, device="cuda"),
                torch.tensor([0, 7, 12], dtype=index_type, device="cuda"),
            ]
            (
                new_lengths_gpu,
                new_indices_gpu,
                new_weights_gpu,
                new_pos_gpu,
                unbucketize_permute,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda() if weights is not None else None,
                batch_sizes.cuda(),
                max_B,
                block_bucketize_pos,
            )
            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )
            torch.testing.assert_close(
                new_indices_gpu.cpu(), new_indices_ref, rtol=0, atol=0
            )
            if has_weight:
                torch.testing.assert_close(new_weights_gpu.cpu(), new_weights_ref)


extend_test_class(BlockBucketizeTest)

if __name__ == "__main__":
    unittest.main()
