#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import random
import unittest
from itertools import accumulate
from typing import List, Optional

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, permute_indices_ref_, permute_scripted


# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_available


class PermuteIndicesTest(unittest.TestCase):
    @given(
        B=st.integers(min_value=0, max_value=20),
        T=st.integers(min_value=0, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
        is_1D=st.booleans(),
        W=st.integers(min_value=4, max_value=8),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_permute_indices(
        self,
        B: int,
        T: int,
        L: int,
        long_index: bool,
        has_weight: bool,
        is_1D: bool,
        W: int,
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        length_splits: Optional[List[torch.Tensor]] = None
        if is_1D:
            if B == 0:
                batch_sizes = [0] * W
            else:
                batch_sizes = [random.randint(a=1, b=B) for i in range(W)]
            length_splits = [
                torch.randint(low=1, high=L, size=(T, batch_sizes[i])).type(index_dtype)
                for i in range(W)
            ]
            lengths = torch.cat(length_splits, dim=1)
        else:
            lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)

        # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        if is_1D:
            permute_list = []
            offset_w = [0] + list(
                # pyre-fixme[16]
                accumulate([length_split.numel() for length_split in length_splits])
            )
            for t in range(T):
                for w in range(W):
                    for b in range(batch_sizes[w]):
                        permute_list.append(offset_w[w] + t * batch_sizes[w] + b)
        else:
            permute_list = list(range(T))
            random.shuffle(permute_list)

        permute = torch.IntTensor(permute_list)

        if is_1D:
            (
                permuted_lengths_cpu,
                permuted_indices_cpu,
                permuted_weights_cpu,
            ) = torch.ops.fbgemm.permute_1D_sparse_data(
                permute, lengths, indices, weights, None
            )
        else:
            (
                permuted_lengths_cpu,
                permuted_indices_cpu,
                permuted_weights_cpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute, lengths, indices, weights, None
            )
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
            # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
        ) = permute_indices_ref_(lengths, indices, weights, permute.long(), is_1D)
        torch.testing.assert_close(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_close(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_close(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if gpu_available:
            if is_1D:
                (
                    permuted_lengths_gpu,
                    permuted_indices_gpu,
                    permuted_weights_gpu,
                ) = torch.ops.fbgemm.permute_1D_sparse_data(
                    permute.cuda(),
                    lengths.cuda(),
                    indices.cuda(),
                    # pyre-fixme[16]: `Optional` has no attribute `cuda`.
                    weights.cuda() if has_weight else None,
                    None,
                )
            else:
                (
                    permuted_lengths_gpu,
                    permuted_indices_gpu,
                    permuted_weights_gpu,
                ) = torch.ops.fbgemm.permute_2D_sparse_data(
                    permute.cuda(),
                    lengths.cuda(),
                    indices.cuda(),
                    weights.cuda() if has_weight else None,
                    None,
                )
            torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_gpu is None

    # TorchScript has different behaviors than eager mode. We can see undefined
    # models returned. So we need to add a unittest to ensure the op return
    # real None, not an undefined tensor.
    def test_permute_indices_scripted_with_none_weights(
        self,
    ) -> None:
        index_dtype = torch.int32
        lengths = torch.randint(low=1, high=2, size=(1, 1)).type(index_dtype)
        weights = None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(1))
        random.shuffle(permute_list)

        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = permute_scripted(permute, lengths, indices)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
            # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
        ) = permute_indices_ref_(lengths, indices, weights, permute.long(), False)
        self.assertTrue(torch.equal(permuted_indices_cpu, permuted_indices_ref))
        self.assertTrue(torch.equal(permuted_lengths_cpu, permuted_lengths_ref))
        self.assertEqual(permuted_weights_cpu, None)
        self.assertEqual(permuted_weights_ref, None)

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
        # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
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
        ) = torch.ops.fbgemm.permute_2D_sparse_data(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
            # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
        ) = permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_close(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_close(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_close(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if gpu_available:
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                # pyre-fixme[16]: `Optional` has no attribute `cuda`.
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_cpu is None


extend_test_class(PermuteIndicesTest)

if __name__ == "__main__":
    unittest.main()
