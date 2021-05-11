#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import random
from typing import Optional, Tuple
from itertools import accumulate

import hypothesis.strategies as st

import torch
from hypothesis import Verbosity, given, settings

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


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

        # pyre-fixme[28]: Unexpected keyword argument `dtype`.
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
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
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
        ) = torch.ops.fb.permute_sparse_data(permute, lengths, indices, weights)
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
            ) = torch.ops.fb.permute_sparse_data(
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
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
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
        ) = torch.ops.fb.permute_sparse_data(permute, lengths, indices, weights)
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
            ) = torch.ops.fb.permute_sparse_data(
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
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
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
        ) = torch.ops.fb.permute_sparse_data(permute, lengths, indices, weights)
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
            ) = torch.ops.fb.permute_sparse_data(
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
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
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
        ) = torch.ops.fb.permute_sparse_data(permute, lengths, embeddings, None)
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
            ) = torch.ops.fb.permute_sparse_data(
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


if __name__ == "__main__":
    unittest.main()
