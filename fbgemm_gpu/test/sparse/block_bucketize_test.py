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
from typing import Optional, Type

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, skipIfRocm
else:
    from fbgemm_gpu.test.test_utils import gpu_available, skipIfRocm

ROCM_FAILURE_MESSAGE = "Test is causing HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION"


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
    def validate_out_of_order_output(
        self,
        expected: torch.Tensor,
        actual: torch.Tensor,
        lengths: torch.Tensor,
        is_int: bool = True,
    ) -> None:
        self.assertEqual(actual.numel(), expected.numel())
        self.assertEqual(torch.sum(lengths).item(), actual.numel())
        expected_list = expected.tolist()
        actual_list = actual.tolist()
        offset_list = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths).tolist()

        for i in range(len(offset_list) - 1):
            expected_sample = sorted(expected_list[offset_list[i] : offset_list[i + 1]])
            actual_sample = sorted(actual_list[offset_list[i] : offset_list[i + 1]])
            if is_int:
                self.assertEqual(expected_sample, actual_sample)
            else:
                for left, right in zip(expected_sample, actual_sample):
                    self.assertAlmostEqual(left, right)
        return

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        long_indices=st.booleans(),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        keep_orig_idx=st.booleans(),
        sequence=st.booleans(),
        bucketize_pos=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_long_indices(
        self,
        long_indices: bool,
        use_cpu: bool,
        keep_orig_idx: bool,
        sequence: bool,
        bucketize_pos: bool,
    ) -> None:
        index_type = torch.long if long_indices else torch.int
        # 3 GPUs
        my_size = 3
        block_sizes = torch.tensor([3, 4, 5], dtype=index_type)

        if not long_indices:
            # batch size 2, 3 features to 3 gpus
            lengths = torch.tensor([0, 3, 2, 0, 1, 5], dtype=index_type)
            indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0], dtype=index_type)

            new_lengths_ref = torch.tensor(
                [
                    0,
                    2,
                    0,
                    0,
                    0,
                    1,  # GPU 0, F0 = [0-3), F1 = [0-4), F2 = [0-5)
                    0,
                    1,
                    2,
                    0,
                    1,
                    3,  # GPU 1, F0 = [3-6), F1 = [4-8), F2 = [5-10)
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,  # GPU 2, F0 = [6-9), F1 = [8-12), F2 = [10-15)
                ],
                dtype=index_type,
            )
            if keep_orig_idx:
                new_indices_ref = torch.tensor(
                    [
                        1,
                        2,
                        0,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                    ],
                    dtype=index_type,
                )
            else:
                new_indices_ref = torch.tensor(
                    [
                        1,
                        2,
                        0,
                        0,
                        0,
                        1,
                        1,
                        2,
                        3,
                        4,
                        0,
                    ],
                    dtype=index_type,
                )

        else:
            lengths = torch.tensor([0, 3, 2, 0, 1, 5], dtype=index_type)
            # Test long and negative indices: -8 will be casted to 18446644015555759292
            indices = torch.tensor(
                [1, 2, 3, 100061827127359, 5, 6, 7, -8, 100058153792324, 10, 0],
                dtype=index_type,
            )
            new_lengths_ref = torch.tensor(
                [
                    0,
                    2,
                    0,
                    0,
                    0,
                    1,  # GPU 0, F0 = [0-3), F1 = [0-4), F2 = [0-5) + relevant outliers
                    0,
                    1,
                    2,
                    0,
                    1,
                    1,  # GPU 1, F0 = [3-6), F1 = [4-8), F2 = [5-10) + relevant outliers
                    0,
                    0,
                    0,
                    0,
                    0,
                    3,  # GPU 2, F0 = [6-9), F1 = [8-12), F2 = [10-15) + relevant outliers
                ],
                dtype=index_type,
            )

            if keep_orig_idx:
                new_indices_ref = torch.tensor(
                    [1, 2, 0, 3, 100061827127359, 5, 6, 7, -8, 100058153792324, 10],
                    dtype=index_type,
                )

            else:
                new_indices_ref = torch.tensor(
                    [
                        1,
                        2,
                        0,
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
            keep_orig_idx=keep_orig_idx,
        )
        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref)
        torch.testing.assert_close(
            new_indices_cpu,
            new_indices_ref,
            msg=f"{new_indices_cpu=} != {new_indices_ref=}",
        )

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
                keep_orig_idx=keep_orig_idx,
            )

            torch.testing.assert_close(new_lengths_gpu.cpu(), new_lengths_ref)
            torch.testing.assert_close(new_lengths_gpu.cpu(), new_lengths_cpu)

            if not sequence:
                self.validate_out_of_order_output(
                    new_indices_ref,
                    new_indices_gpu.cpu(),
                    new_lengths_gpu.cpu(),
                )
                self.validate_out_of_order_output(
                    new_indices_cpu,
                    new_indices_gpu.cpu(),
                    new_lengths_gpu.cpu(),
                )
            else:
                torch.testing.assert_close(new_indices_gpu.cpu(), new_indices_ref)
                torch.testing.assert_close(new_indices_gpu.cpu(), new_indices_cpu)

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        long_indices=st.booleans(),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        keep_orig_idx=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_total_num_blocks_uneven_raw_ids(
        self,
        long_indices: bool,
        use_cpu: bool,
        keep_orig_idx: bool,
        sequence: bool,
    ) -> None:
        index_type = torch.long if long_indices else torch.int
        # 3 GPUs
        my_size = 3
        block_sizes = torch.tensor([0, 0, 0], dtype=index_type)
        total_num_blocks = torch.tensor([6, 6, 12], dtype=index_type)
        lengths = torch.tensor([0, 3, 2, 0, 1, 5], dtype=index_type)
        indices = torch.tensor(
            [
                1,
                2,
                10,
                4,
                16,
                6,
                7,
                18,
                19,
                10,
                0,
            ],
            dtype=index_type,
        )
        block_bucketize_pos = [
            torch.tensor([0, 2, 8, 12], dtype=index_type),
            torch.tensor([0, 3, 12, 18], dtype=index_type),
            torch.tensor([0, 4, 18, 24], dtype=index_type),
        ]

        new_lengths_ref = torch.tensor(
            [
                0,
                0,
                0,
                0,
                0,
                1,  # GPU 0, 0's, F2=[0,1]
                0,
                2,
                0,
                0,
                1,
                3,  # GPU 1, [1,2,3], F2=[2:8]
                0,
                1,
                2,
                0,
                0,
                1,  # GPU 2, [4, 5], F2=[9:11]
            ],
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [
                0 if keep_orig_idx else 0 // 12,  # F2 / GPU0
                1 if keep_orig_idx else 1 // 6,  # F0 / GPU0
                2 if keep_orig_idx else 2 // 6,  # F0 / GPU1
                6 if keep_orig_idx else 6 // 12,  # F2 / GPU1
                7 if keep_orig_idx else 7 // 12,  # F2 / GPU1
                18 if keep_orig_idx else 18 // 12,  # F2 / GPU2
                19 if keep_orig_idx else 19 // 12,  # F2 / GPU2
                10 if keep_orig_idx else 10 // 6,  # F1 / GPU2
                4 if keep_orig_idx else 4 // 6,  # F1 / GPU2
                16 if keep_orig_idx else 16 // 6,  # F1 / GPU2
                10 if keep_orig_idx else 10 // 12,  # F0 / GPU2
            ],
            dtype=index_type,
        )
        unbucketize_permute_ref = torch.tensor(
            [
                1,  # F0
                2,  # F0
                7,  # F0
                8,  # F1
                9,  # F1
                3,  # F2
                4,  # F2
                5,  # F2
                6,  # F2
                10,  # F2
                0,  # F2
            ],
            dtype=index_type,
        )

        (
            new_lengths,
            new_indices,
            new_weights,
            new_pos,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths.cuda() if not use_cpu else lengths,
            indices.cuda() if not use_cpu else indices,
            None,
            sequence,
            block_sizes.cuda() if not use_cpu else block_sizes,
            my_size,
            block_bucketize_pos=(
                ([t.cuda() for t in block_bucketize_pos])
                if not use_cpu
                else block_bucketize_pos
            ),
            keep_orig_idx=keep_orig_idx,
            total_num_blocks=(
                total_num_blocks.cuda() if not use_cpu else total_num_blocks
            ),
        )

        torch.testing.assert_close(
            new_lengths.cpu(), new_lengths_ref, msg=f"{new_lengths=}"
        )
        torch.testing.assert_close(
            new_indices.cpu(), new_indices_ref, msg=f"{new_indices=}"
        )
        if unbucketize_permute is not None:
            torch.testing.assert_close(
                unbucketize_permute.cpu(),
                unbucketize_permute_ref,
                msg=f"{unbucketize_permute=}",
            )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        long_indices=st.booleans(),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        keep_orig_idx=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_total_num_blocks_uneven(
        self,
        long_indices: bool,
        use_cpu: bool,
        keep_orig_idx: bool,
        sequence: bool,
    ) -> None:

        index_type = torch.long if long_indices else torch.int
        # 3 GPUs
        my_size = 3
        block_sizes = torch.tensor([2, 3, 4], dtype=index_type)
        total_num_blocks = torch.tensor([6, 6, 6], dtype=index_type)
        lengths = torch.tensor([0, 3, 2, 0, 1, 5], dtype=index_type)
        indices = torch.tensor([1, 2, 10, 4, 16, 6, 7, 18, 19, 10, 0], dtype=index_type)

        block_bucketize_pos = [
            torch.tensor([0, 2, 8, 12], dtype=index_type),
            torch.tensor([0, 3, 12, 18], dtype=index_type),
            torch.tensor([0, 4, 16, 24], dtype=index_type),
        ]

        new_lengths_ref = torch.tensor(
            [
                0,
                1,
                0,
                0,
                0,
                1,  # GPU 0, F0 = [0-2), F1 = [0-3), F2 = [0-4)
                0,
                1,
                1,
                0,
                1,
                2,  # GPU 1, F0 = [2-8), F1 = [3-12), F2 = [4-16)
                0,
                1,
                1,
                0,
                0,
                2,  # GPU 2, F0 = [8-12), F1 = [12-18), F2 = [16-24)
            ],
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [
                1,  # F0 / GPU0
                0,  # F2 / GPU0
                2 if keep_orig_idx else 2 - 2,  # F0 / GPU1
                4 if keep_orig_idx else 4 - 3,  # F1 / GPU1
                6 if keep_orig_idx else 6 - 4,  # F2 / GPU1
                7 if keep_orig_idx else 7 - 4,  # F2 / GPU1
                10 if keep_orig_idx else 10 - 4,  # F2 / GPU1
                10 if keep_orig_idx else 10 - 8,  # F0 / GPU2
                16 if keep_orig_idx else 16 - 12,  # F1 / GPU2
                18 if keep_orig_idx else 18 - 16,  # F2 / GPU2
                19 if keep_orig_idx else 19 - 16,  # F2 / GPU2
            ],
            dtype=index_type,
        )
        unbucketize_permute_ref = torch.tensor(
            [
                0,  # F0
                2,  # F0
                7,  # F0
                3,  # F1
                8,  # F1
                4,  # F2
                5,  # F2
                9,  # F2
                10,  # F2
                6,  # F2
                1,  # F2
            ],
            dtype=index_type,
        )

        (
            new_lengths,
            new_indices,
            new_weights,
            new_pos,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths.cuda() if not use_cpu else lengths,
            indices.cuda() if not use_cpu else indices,
            None,
            sequence,
            block_sizes.cuda() if not use_cpu else block_sizes,
            my_size,
            block_bucketize_pos=(
                ([t.cuda() for t in block_bucketize_pos])
                if not use_cpu
                else block_bucketize_pos
            ),
            keep_orig_idx=keep_orig_idx,
            total_num_blocks=(
                total_num_blocks.cuda() if not use_cpu else total_num_blocks
            ),
        )

        torch.testing.assert_close(
            new_lengths.cpu(), new_lengths_ref, msg=f"{new_lengths=}"
        )
        torch.testing.assert_close(
            new_indices.cpu(), new_indices_ref, msg=f"{new_indices=}"
        )
        assert new_weights is None and new_pos is None
        if unbucketize_permute is not None:
            torch.testing.assert_close(
                unbucketize_permute.cpu(),
                unbucketize_permute_ref,
                msg=f"{unbucketize_permute=}",
            )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        long_indices=st.booleans(),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        keep_orig_idx=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_total_num_blocks(
        self,
        long_indices: bool,
        use_cpu: bool,
        keep_orig_idx: bool,
        sequence: bool,
    ) -> None:
        index_type = torch.long if long_indices else torch.int
        # 3 GPUs
        my_size = 3
        block_sizes = torch.tensor([2, 3, 4], dtype=index_type)
        total_num_blocks = torch.tensor([6, 6, 6], dtype=index_type)

        lengths = torch.tensor([0, 3, 2, 0, 1, 5], dtype=index_type)
        indices = torch.tensor([1, 2, 10, 4, 16, 6, 7, 18, 19, 10, 0], dtype=index_type)

        new_lengths_ref = torch.tensor(
            [
                0,
                2,
                1,
                0,
                1,
                2,  # GPU 0, F0 = [0-4), F1 = [0-6), F2 = [0-8)
                0,
                0,
                0,
                0,
                0,
                1,  # GPU 1, F0 = [4-8), F1 = [6-12), F2 = [8-16)
                0,
                1,
                1,
                0,
                0,
                2,  # GPU 2, F0 = [8-12), F1 = [12-18), F2 = [16-24)
            ],
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [
                1,  # F0
                2,  # F0
                4,  # F1
                6,  # F2
                7,  # F2
                0,  # F2
                10 if keep_orig_idx else 10 - 1 * 8,  # F2
                10 if keep_orig_idx else 10 - 2 * 4,  # F0
                16 if keep_orig_idx else 16 - 2 * 6,  # F1
                18 if keep_orig_idx else 18 - 2 * 8,  # F2
                19 if keep_orig_idx else 19 - 2 * 8,  # F2
            ],
            dtype=index_type,
        )
        unbucketize_permute_ref = torch.tensor(
            [
                0,  # F0
                1,  # F0
                7,  # F0
                2,  # F1
                8,  # F1
                3,  # F2
                4,  # F2
                9,  # F2
                10,  # F2
                6,  # F2
                5,  # F2
            ],
            dtype=index_type,
        )

        (
            new_lengths,
            new_indices,
            new_weights,
            new_pos,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths.cuda() if not use_cpu else lengths,
            indices.cuda() if not use_cpu else indices,
            None,
            sequence,
            block_sizes.cuda() if not use_cpu else block_sizes,
            my_size,
            keep_orig_idx=keep_orig_idx,
            total_num_blocks=(
                total_num_blocks.cuda() if not use_cpu else total_num_blocks
            ),
        )

        torch.testing.assert_close(
            new_lengths.cpu(), new_lengths_ref, msg=f"{new_lengths=}"
        )
        torch.testing.assert_close(
            new_indices.cpu(), new_indices_ref, msg=f"{new_indices=}"
        )
        if unbucketize_permute is not None:
            torch.testing.assert_close(
                unbucketize_permute.cpu(),
                unbucketize_permute_ref,
                msg=f"{unbucketize_permute=}",
            )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        long_indices=st.booleans(),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        keep_orig_idx=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_total_num_blocks_raw_ids(
        self,
        long_indices: bool,
        use_cpu: bool,
        keep_orig_idx: bool,
        sequence: bool,
    ) -> None:
        index_type = torch.long if long_indices else torch.int
        # 3 GPUs
        my_size = 3
        block_sizes = torch.tensor([0, 0, 0], dtype=index_type)
        total_num_blocks = torch.tensor([3, 6, 9], dtype=index_type)

        lengths = torch.tensor([0, 3, 2, 0, 1, 5], dtype=index_type)
        indices = torch.tensor([1, 2, 10, 4, 16, 6, 7, 18, 19, 10, 0], dtype=index_type)
        new_lengths_ref = torch.tensor(
            [
                0,
                0,
                0,
                0,
                0,
                4,  # GPU 0, F0: 0, F1: 0,1, F2: 0,1,2
                0,
                2,
                0,
                0,
                0,
                0,  # GPU 1, F0: 1, F1: 2,3, F2: 3,4,5
                0,
                1,
                2,
                0,
                1,
                1,  # GPU 2, F0: 2, F1: 4,5, F2: 6,7,8
            ],
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [
                18 if keep_orig_idx else 18 // 9,  # F2
                19 if keep_orig_idx else 19 // 9,  # F2
                10 if keep_orig_idx else 10 // 9,  # F2
                0,  # F2
                1 if keep_orig_idx else 1 // 3,  # F0
                10 if keep_orig_idx else 10 // 3,  # F0
                2 if keep_orig_idx else 2 // 3,  # F0
                4 if keep_orig_idx else 4 // 6,  # F1
                16 if keep_orig_idx else 16 // 6,  # F1
                6 if keep_orig_idx else 6 // 9,  # F2
                7 if keep_orig_idx else 7 // 9,  # F2
            ],
            dtype=index_type,
        )
        unbucketize_permute_ref = torch.tensor(
            [
                4,  # F0
                6,  # F0
                5,  # F0
                7,  # F1
                8,  # F1
                9,  # F2
                10,  # F2
                0,  # F2
                1,  # F2
                2,  # F2
                3,  # F2
            ],
            dtype=index_type,
        )

        (
            new_lengths,
            new_indices,
            new_weights,
            new_pos,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths.cuda() if not use_cpu else lengths,
            indices.cuda() if not use_cpu else indices,
            None,
            sequence,
            block_sizes.cuda() if not use_cpu else block_sizes,
            my_size,
            keep_orig_idx=keep_orig_idx,
            total_num_blocks=(
                total_num_blocks.cuda() if not use_cpu else total_num_blocks
            ),
        )

        torch.testing.assert_close(
            new_lengths.cpu(), new_lengths_ref, msg=f"{new_lengths=}"
        )
        torch.testing.assert_close(
            new_indices.cpu(), new_indices_ref, msg=f"{new_indices=}"
        )
        if unbucketize_permute is not None:
            torch.testing.assert_close(
                unbucketize_permute.cpu(),
                unbucketize_permute_ref,
                msg=f"{unbucketize_permute=}",
            )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
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
                torch.testing.assert_close(
                    new_indices_gpu.cpu(), new_indices_ref, rtol=0, atol=0
                )
                if has_weight:
                    torch.testing.assert_close(new_weights_gpu.cpu(), new_weights_cpu)
                if bucketize_pos:
                    torch.testing.assert_close(new_pos_gpu.cpu(), new_pos_cpu)
            else:
                self.validate_out_of_order_output(
                    new_indices_ref, new_indices_gpu.cpu(), new_lengths_ref
                )
                if has_weight:
                    self.validate_out_of_order_output(
                        new_weights_ref,
                        new_weights_gpu.cpu(),
                        new_lengths_ref,
                        is_int=False,
                    )
                if bucketize_pos:
                    self.validate_out_of_order_output(
                        new_pos_ref, new_pos_gpu.cpu(), new_lengths_ref
                    )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_inference(
        self,
        index_type: Type[torch.dtype],
    ) -> None:
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
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
        (
            new_lengths_cpu,
            new_indices_cpu,
            _,
            _,
            _,
            bucket_mapping,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
            lengths,
            indices,
            False,
            True,
            block_sizes,
            my_size,
            None,
            return_bucket_mapping=True,
        )

        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref, rtol=0, atol=0)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref, rtol=0, atol=0)

        if gpu_available:
            (
                new_lengths_gpu,
                new_indices_gpu,
                _,
                _,
                _,
                bucket_mapping_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
                lengths.cuda(),
                indices.cuda(),
                False,
                True,
                block_sizes.cuda(),
                my_size,
                None,
                return_bucket_mapping=True,
            )
            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )
            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )
            torch.testing.assert_allclose(
                bucket_mapping_gpu.cpu(),
                bucket_mapping,
            )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_populate_bucketized_permute(
        self,
        index_type: Type[torch.dtype],
    ) -> None:
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
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
        (
            new_lengths_cpu,
            new_indices_cpu,
            _,
            _,
            unbucketize_permute_cpu,
            bucket_mapping_cpu,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
            lengths,
            indices,
            False,
            True,
            block_sizes,
            my_size,
            None,
            return_bucket_mapping=True,
        )

        unbucketize_permute_populated_cpu = (
            torch.ops.fbgemm.populate_bucketized_permute(
                lengths,
                new_lengths_cpu,
                bucket_mapping_cpu,
            )
        )
        torch.testing.assert_close(
            unbucketize_permute_populated_cpu, unbucketize_permute_cpu, rtol=0, atol=0
        )
        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref, rtol=0, atol=0)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref, rtol=0, atol=0)

        if gpu_available:
            (
                new_lengths_gpu,
                new_indices_gpu,
                _,
                _,
                unbucketize_permute_gpu,
                bucket_mapping_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
                lengths.cuda(),
                indices.cuda(),
                False,
                True,
                block_sizes.cuda(),
                my_size,
                None,
                return_bucket_mapping=True,
            )

            unbucketize_permute_populated_gpu = (
                torch.ops.fbgemm.populate_bucketized_permute(
                    lengths.cuda(),
                    new_lengths_gpu,
                    bucket_mapping_gpu,
                )
            )
            torch.testing.assert_close(
                unbucketize_permute_gpu.cpu(),
                unbucketize_permute_populated_gpu.cpu(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )
            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )
            torch.testing.assert_allclose(
                bucket_mapping_gpu.cpu(),
                bucket_mapping_cpu,
            )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
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
            if sequence:
                torch.testing.assert_close(
                    new_indices_gpu.cpu(), new_indices_ref, rtol=0, atol=0
                )
            else:
                self.validate_out_of_order_output(
                    new_indices_ref, new_indices_gpu.cpu(), new_lengths_ref
                )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
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
            if sequence:
                torch.testing.assert_close(
                    new_indices_gpu.cpu(), new_indices_ref, rtol=0, atol=0
                )
                if has_weight:
                    torch.testing.assert_close(new_weights_gpu.cpu(), new_weights_ref)
            else:
                self.validate_out_of_order_output(
                    new_indices_ref, new_indices_gpu.cpu(), new_lengths_ref
                )
                if has_weight:
                    self.validate_out_of_order_output(
                        new_weights_ref,
                        new_weights_gpu.cpu(),
                        new_lengths_ref,
                        is_int=False,
                    )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @unittest.skipIf(not gpu_available, "Skip is GPU is not available.")
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        has_weight=st.booleans(),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
        my_size=st.sampled_from([3, 194, 256, 1024]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=32, deadline=None)
    def test_block_bucketize_sparse_features_large(
        self,
        index_type: Type[torch.dtype],
        has_weight: bool,
        bucketize_pos: bool,
        sequence: bool,
        my_size: int,
    ) -> None:
        bucket_size = 5
        warp_size = 32
        max_num_thread_in_a_block = 1024
        num_of_items = max_num_thread_in_a_block * 2
        avg_item_len = warp_size + 8
        lengths = [
            int(random.gauss(mu=avg_item_len, sigma=1.0)) for _ in range(num_of_items)
        ]
        total_len = sum(lengths)
        # pyre-ignore [6]
        block_sizes = torch.tensor([bucket_size] * 4, dtype=index_type)
        self.assertTrue(num_of_items % block_sizes.numel() == 0)
        B = num_of_items // block_sizes.numel()
        # pyre-ignore [6]
        lengths = torch.tensor(lengths, dtype=index_type)
        indices = torch.randint(
            0,
            my_size * bucket_size,
            (total_len,),
            # pyre-ignore [6]
            dtype=index_type,
        )
        weights = (
            torch.rand(
                (total_len,),
                dtype=torch.float,
            )
            if has_weight
            else None
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
            torch.testing.assert_close(unbucketized_indices, indices, rtol=0, atol=0)
            torch.testing.assert_close(
                new_indices_gpu.cpu(), new_indices_cpu, rtol=0, atol=0
            )
            if has_weight:
                torch.testing.assert_close(new_weights_gpu.cpu(), new_weights_cpu)
            if bucketize_pos:
                torch.testing.assert_close(new_pos_gpu.cpu(), new_pos_cpu)
        else:
            self.validate_out_of_order_output(
                new_indices_cpu, new_indices_gpu.cpu(), new_lengths_cpu
            )
            if has_weight:
                self.validate_out_of_order_output(
                    new_weights_cpu,
                    new_weights_gpu.cpu(),
                    new_lengths_cpu,
                    is_int=False,
                )
            if bucketize_pos:
                self.validate_out_of_order_output(
                    new_pos_cpu, new_pos_gpu.cpu(), new_lengths_cpu
                )


extend_test_class(BlockBucketizeTest)

if __name__ == "__main__":
    unittest.main()
