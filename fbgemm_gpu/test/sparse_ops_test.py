#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
import random
from typing import Optional, Tuple, Type, Union
from itertools import accumulate

import hypothesis.strategies as st

import torch
from hypothesis import Verbosity, given, settings

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")

np_int_types = Union[Type[np.int32], Type[np.int64]]

def unbucketize_indices_value(
    bucketized_indices: torch.Tensor,
    bucketized_lengths: torch.Tensor,
    block_sizes: torch.Tensor,
    W: int,
    B: int,
) -> torch.Tensor:
    lengths_sum = bucketized_indices.size()[0]
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


class SparseOpsTest(unittest.TestCase):
    @staticmethod
    def permute_indices_ref_(
        lengths: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        permute: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        T = lengths.size(0)
        permuted_lengths = torch.index_select(lengths.view(T, -1), 0, permute)

        original_segment_lengths = lengths.view(T, -1).sum(dim=1, dtype=torch.int32)
        original_segment_start = [0] + list(
            accumulate(original_segment_lengths.view(-1))
        )

        permuted_indices = []
        permuted_weights = []
        for i in range(permute.size(0)):
            start = original_segment_start[permute[i]]
            end = start + original_segment_lengths[permute[i]]
            permuted_indices.append(indices[start:end])
            if weights is not None:
                permuted_weights.append(weights[start:end])

        permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

        if weights is None:
            permuted_weights = None
        else:
            permuted_weights = torch.cat(permuted_weights, dim=0).flatten()

        return permuted_lengths, permuted_indices, permuted_weights


    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_indices(
        self, B: int, T: int, L: int, long_index: bool, has_weight: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_sparse_data(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = self.permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_allclose(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_allclose(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_allclose(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if torch.cuda.is_available():
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_allclose(
                permuted_indices_gpu.cpu(), permuted_indices_cpu
            )
            torch.testing.assert_allclose(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu
            )
            if has_weight:
                torch.testing.assert_allclose(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_gpu is None

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_indices_with_repeats(
        self, B: int, T: int, L: int, long_index: bool, has_weight: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(T))

        num_repeats = random.randint(0, T)
        for _ in range(num_repeats):
            permute_list.append(random.randint(0, T - 1))

        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_sparse_data(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = self.permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_allclose(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_allclose(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_allclose(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if torch.cuda.is_available():
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_allclose(
                permuted_indices_gpu.cpu(), permuted_indices_cpu
            )
            torch.testing.assert_allclose(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu
            )
            if has_weight:
                torch.testing.assert_allclose(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_cpu is None

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        D=st.integers(min_value=5, max_value=20),
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_indices_multi_dimension(
        self, D: int, B: int, T: int, L: int, long_index: bool, has_weight: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B, D)).type(index_dtype)
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_sparse_data(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = self.permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_allclose(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_allclose(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_allclose(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if torch.cuda.is_available():
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_allclose(
                permuted_indices_gpu.cpu(), permuted_indices_cpu
            )
            torch.testing.assert_allclose(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu
            )
            if has_weight:
                torch.testing.assert_allclose(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_cpu is None

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_embeddings(self, B: int, T: int, L: int, long_index: bool) -> None:
        index_dtype = torch.int32 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        embeddings = torch.rand(lengths.sum().item()).float()
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_embeddings_cpu,
            _,
        ) = torch.ops.fbgemm.permute_sparse_data(permute, lengths, embeddings, None)
        (
            permuted_lengths_ref,
            permuted_embeddings_ref,
            _,
        ) = self.permute_indices_ref_(lengths, embeddings, None, permute.long())
        torch.testing.assert_allclose(permuted_embeddings_cpu, permuted_embeddings_ref)
        torch.testing.assert_allclose(permuted_lengths_cpu, permuted_lengths_ref)

        if torch.cuda.is_available():
            (
                permuted_lengths_gpu,
                permuted_embeddings_gpu,
                _,
            ) = torch.ops.fbgemm.permute_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                embeddings.cuda(),
                None,
            )
            torch.testing.assert_allclose(
                permuted_embeddings_gpu.cpu(), permuted_embeddings_cpu
            )
            torch.testing.assert_allclose(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        long_indices=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_long_indices(
        self,
        long_indices: bool,
    ) -> None:
        has_weight = False
        bucketize_pos = False
        sequence = False
        index_type = torch.long

        # 3 features, 2 batches
        T = 3
        B = 2

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

        print(f"new_lengths_gpu={new_lengths_gpu}")
        print(f"new_indices_gpu={new_indices_gpu}")
        torch.testing.assert_allclose(new_lengths_gpu.cpu(), new_lengths_ref)
        torch.testing.assert_allclose(new_indices_gpu.cpu(), new_indices_ref)

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        offset_type=st.sampled_from([torch.int, torch.long]),
        index_type=st.sampled_from([torch.int, torch.long]),
        has_weight=st.booleans(),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features(
        self,
        offset_type: np_int_types,
        index_type: np_int_types,
        has_weight: bool,
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        T = 4
        B = 2
        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 2nd param but
        #  got `Union[Type[np.int32], Type[np.int64]]`.
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=offset_type)
        indices = torch.tensor(
            # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 2nd param
            #  but got `Union[Type[np.int32], Type[np.int64]]`.
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20], dtype=index_type
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
                # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 2nd
                #  param but got `Type[float]`.
                dtype=float,
            )
            if has_weight
            else None
        )
        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 2nd param but
        #  got `Union[Type[np.int32], Type[np.int64]]`.
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        new_lengths_ref = torch.tensor(
            # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 2nd param
            #  but got `Union[Type[np.int32], Type[np.int64]]`.
            [0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 1, 2, 1, 3, 2, 1], dtype=index_type
        )
        new_indices_ref = torch.tensor(
            # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 2nd param
            #  but got `Union[Type[np.int32], Type[np.int64]]`.
            [3, 4, 11, 1, 11, 0, 13, 14, 0, 1, 2, 3, 2, 0, 0], dtype=index_type
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
            # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 2nd param
            #  but got `Type[float]`.
            dtype=float,
        )
        new_pos_ref = torch.tensor(
            # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 2nd param
            #  but got `Union[Type[np.int32], Type[np.int64]]`.
            [0, 1, 0, 0, 0, 0, 1, 2, 1, 0, 1, 2, 1, 2, 0], dtype=index_type
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
            weights.cuda() if has_weight else None,
        )
        torch.testing.assert_allclose(new_lengths_gpu.cpu(), new_lengths_ref, 0, 0)
        torch.testing.assert_allclose(new_indices_gpu.cpu(), new_indices_ref, 0, 0)
        if has_weight:
            torch.testing.assert_allclose(new_weights_gpu.cpu(), new_weights_ref)
        if bucketize_pos:
            torch.testing.assert_allclose(new_pos_gpu.cpu(), new_pos_ref)
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
            torch.testing.assert_allclose(unbucketized_indices, indices, 0, 0)


if __name__ == "__main__":
    unittest.main()
