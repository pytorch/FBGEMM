#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import itertools
import random
import unittest
from itertools import accumulate
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable


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


def get_n_rand_num_summing_to_k(n: int, k: int) -> np.ndarray:
    """Get a list of `n` integers which collectively sum to `k`, drawn
    uniformly from the set of all such lists.

    Args:
        n - The number of integers in the result list
        k - The value they should sum to
    """
    # There are a lot of ways to do this wrong, probably including
    # the ones you've just thought of. I think the following does
    # it correctly, though.
    if n == 0:
        return np.array([])
    return np.random.multinomial(k, np.ones(n) / n, size=1)[0]


class SparseOpsTest(unittest.TestCase):
    @staticmethod
    def permute_indices_ref_(
        lengths: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        permute: torch.LongTensor,
        is_1D: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        T = lengths.size(0)
        if is_1D:
            permuted_lengths = torch.index_select(lengths.view(-1), 0, permute).view(-1)
            original_segment_lengths = lengths.view(-1)
            original_segment_start = [0] + list(accumulate(lengths.view(-1)))

            permuted_indices = []
            permuted_weights = []
            for i in range(permute.numel()):
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
        else:
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
        ) = self.permute_indices_ref_(lengths, indices, weights, permute.long(), is_1D)
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

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        permute_size=st.integers(min_value=30, max_value=1000),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_invert_permute(
        self,
        permute_size: int,
        long_index: bool,
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        permute_list = list(range(permute_size))
        random.shuffle(permute_list)
        inversed_permute_list = [0] * len(permute_list)
        for i in range(permute_size):
            inversed_permute_list[permute_list[i]] = i
        permute = torch.IntTensor(permute_list).type(index_dtype)
        inverse_permute_ref = torch.IntTensor(inversed_permute_list).type(index_dtype)

        inverse_permute_cpu = torch.ops.fbgemm.invert_permute(permute)
        torch.testing.assert_close(inverse_permute_cpu, inverse_permute_ref)

        if gpu_available:
            inverse_permute_gpu = torch.ops.fbgemm.invert_permute(permute.cuda())
            torch.testing.assert_close(inverse_permute_gpu.cpu(), inverse_permute_cpu)

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
        ) = self.permute_indices_ref_(lengths, indices, weights, permute.long())
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

    @staticmethod
    def permute_embeddings_(
        permute_fn: Callable[..., Tuple[torch.Tensor, ...]],
        *args: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if permute_fn == torch.ops.fbgemm.permute_2D_sparse_data:
            permuted_lengths, permuted_embeddings, _ = permute_fn(*args, None)
            return permuted_lengths, permuted_embeddings
        else:
            return permute_fn(*args)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        permute_fn=st.sampled_from(
            [
                torch.ops.fbgemm.permute_2D_sparse_data,
                torch.ops.fbgemm.permute_sequence_embeddings,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_embeddings(
        self,
        B: int,
        T: int,
        L: int,
        long_index: bool,
        permute_fn: Callable[..., Tuple[torch.Tensor, ...]],
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
        embeddings = torch.rand(lengths.sum().item()).float()
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (permuted_lengths_cpu, permuted_embeddings_cpu) = self.permute_embeddings_(
            permute_fn, permute, lengths, embeddings
        )
        (
            permuted_lengths_ref,
            permuted_embeddings_ref,
            _,
            # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
        ) = self.permute_indices_ref_(lengths, embeddings, None, permute.long())
        torch.testing.assert_close(permuted_embeddings_cpu, permuted_embeddings_ref)
        torch.testing.assert_close(permuted_lengths_cpu, permuted_lengths_ref)

        if gpu_available:
            (permuted_lengths_gpu, permuted_embeddings_gpu) = self.permute_embeddings_(
                permute_fn,
                permute.cuda(),
                lengths.cuda(),
                embeddings.cuda(),
            )
            torch.testing.assert_close(
                permuted_embeddings_gpu.cpu(), permuted_embeddings_cpu
            )
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
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

    # pyre-ignore [56]
    @given(
        n=st.integers(min_value=1, max_value=100),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_cumsum(self, n: int, long_index: bool) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        np_index_dtype = np.int64 if long_index else np.int32

        # cpu tests
        x = torch.randint(low=0, high=100, size=(n,)).type(index_dtype)
        ze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
        zi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
        zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        torch.testing.assert_close(
            torch.from_numpy(np.cumsum(x.cpu().numpy()).astype(np_index_dtype)),
            zi.cpu(),
        )
        torch.testing.assert_close(
            torch.from_numpy(
                (np.cumsum([0] + x.cpu().numpy().tolist())[:-1]).astype(np_index_dtype)
            ),
            ze.cpu(),
        )
        torch.testing.assert_close(
            torch.from_numpy(
                (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype)
            ),
            zc.cpu(),
        )

        if gpu_available:
            x = x.cuda()
            ze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
            zi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
            zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
            torch.testing.assert_close(
                torch.from_numpy(np.cumsum(x.cpu().numpy()).astype(np_index_dtype)),
                zi.cpu(),
            )
            torch.testing.assert_close(
                torch.from_numpy(
                    (np.cumsum([0] + x.cpu().numpy().tolist())[:-1]).astype(
                        np_index_dtype
                    )
                ),
                ze.cpu(),
            )
            torch.testing.assert_close(
                torch.from_numpy(
                    (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype)
                ),
                zc.cpu(),
            )

    # pyre-ignore [56]
    @given(
        n=st.integers(min_value=1, max_value=600),
        b=st.integers(min_value=1, max_value=10),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_asynchronous_complete_cumsum_2d(
        self, n: int, b: int, long_index: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32

        def test_asynchronous_complete_cumsum_2d_helper(x: torch.Tensor) -> None:
            np_index_dtype = np.int64 if long_index else np.int32
            zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
            zeros = torch.zeros(b, 1)
            torch.testing.assert_close(
                torch.from_numpy(
                    np.cumsum(
                        torch.concat([zeros, x.cpu()], dim=1).numpy(), axis=1
                    ).astype(np_index_dtype)
                ),
                zc.cpu(),
            )

        x = torch.randint(low=0, high=100, size=(b, n)).type(index_dtype)
        # cpu test
        test_asynchronous_complete_cumsum_2d_helper(x)
        if gpu_available:
            # gpu test
            test_asynchronous_complete_cumsum_2d_helper(x.cuda())

    # pyre-ignore [56]
    @given(
        N=st.integers(min_value=1, max_value=20),
        offsets_type=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_offsets_range(
        self,
        N: int,
        # pyre-fixme[11]: Annotation `int32` is not defined as a type.
        # pyre-fixme[11]: Annotation `int64` is not defined as a type.
        offsets_type: "Union[Type[torch.int32], Type[torch.int64]]",
    ) -> None:
        lengths = np.array([np.random.randint(low=0, high=20) for _ in range(N)])
        offsets = np.cumsum(np.concatenate(([0], lengths)))[:-1]
        range_ref = torch.from_numpy(
            np.concatenate([np.arange(size) for size in lengths])
        )
        output_size = np.sum(lengths)

        offsets_cpu = torch.tensor(offsets, dtype=offsets_type)
        range_cpu = torch.ops.fbgemm.offsets_range(offsets_cpu, output_size)
        range_ref = range_ref.to(range_cpu.dtype)
        torch.testing.assert_close(range_cpu, range_ref, rtol=0, atol=0)

        if gpu_available:
            range_gpu = torch.ops.fbgemm.offsets_range(offsets_cpu.cuda(), output_size)
            range_ref = range_ref.to(range_gpu.dtype)
            torch.testing.assert_close(range_gpu.cpu(), range_ref, rtol=0, atol=0)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
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

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        has_weight=st.booleans(),
        bucketize_pos=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_bucketize_sparse_features(
        self,
        index_type: Type[torch.dtype],
        has_weight: bool,
        bucketize_pos: bool,
    ) -> None:
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3], dtype=index_type)
        # pyre-ignore [6]
        indices = torch.tensor([10, 10, 15, 20, 25, 30], dtype=index_type)
        weights = (
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float)
            if has_weight
            else None
        )

        # pyre-ignore [6]
        new_lengths_ref = torch.tensor([0, 2, 0, 2, 0, 0, 1, 1], dtype=index_type)
        # pyre-ignore [6]
        new_indices_ref = torch.tensor([5, 5, 10, 15, 7, 12], dtype=index_type)
        new_weights_ref = torch.tensor(
            [1.0, 2.0, 4.0, 6.0, 3.0, 5.0], dtype=torch.float
        )
        # pyre-ignore [6]
        new_pos_ref = torch.tensor([0, 1, 0, 2, 0, 1], dtype=index_type)
        (
            new_lengths_cpu,
            new_indices_cpu,
            new_weights_cpu,
            new_pos_cpu,
        ) = torch.ops.fbgemm.bucketize_sparse_features(
            lengths, indices, bucketize_pos, 2, weights
        )
        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref, rtol=0, atol=0)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref, rtol=0, atol=0)
        if has_weight:
            torch.testing.assert_close(new_weights_cpu, new_weights_ref)
        if bucketize_pos:
            torch.testing.assert_close(new_pos_cpu, new_pos_ref)
        if gpu_available:
            (
                new_lengths_gpu,
                new_indices_gpu,
                new_weights_gpu,
                new_pos_gpu,
            ) = torch.ops.fbgemm.bucketize_sparse_features(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                2,
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

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
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

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
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
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
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

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
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

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(data_type=st.sampled_from([torch.bfloat16, torch.half, torch.float32]))
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_histogram_binning_calibration(self, data_type: torch.dtype) -> None:
        num_bins = 5000

        logit = torch.tensor([[-0.0018], [0.0085], [0.0090], [0.0003], [0.0029]]).type(
            data_type
        )

        bin_num_examples = torch.empty([num_bins], dtype=torch.float64).fill_(0.0)
        bin_num_positives = torch.empty([num_bins], dtype=torch.float64).fill_(0.0)

        calibrated_prediction, bin_ids = torch.ops.fbgemm.histogram_binning_calibration(
            logit=logit,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            positive_weight=0.4,
            lower_bound=0.0,
            upper_bound=1.0,
            bin_ctr_in_use_after=10000,
            bin_ctr_weight_value=0.9995,
        )

        expected_calibrated_prediction = torch.tensor(
            [[0.2853], [0.2875], [0.2876], [0.2858], [0.2863]]
        ).type(data_type)
        expected_bin_ids = torch.tensor(
            [1426, 1437, 1437, 1428, 1431], dtype=torch.long
        )

        error_tolerance = 1e-03
        if data_type == torch.bfloat16:
            # Due to smaller significand bits.
            error_tolerance = 1e-02

            expected_bin_ids = torch.tensor(
                [1426, 1438, 1438, 1430, 1430], dtype=torch.long
            )

        torch.testing.assert_close(
            calibrated_prediction,
            expected_calibrated_prediction,
            rtol=error_tolerance,
            atol=error_tolerance,
        )

        self.assertTrue(
            torch.equal(
                bin_ids.long(),
                expected_bin_ids,
            )
        )

        if torch.cuda.is_available():
            (
                calibrated_prediction_gpu,
                bin_ids_gpu,
            ) = torch.ops.fbgemm.histogram_binning_calibration(
                logit=logit.cuda(),
                bin_num_examples=bin_num_examples.cuda(),
                bin_num_positives=bin_num_positives.cuda(),
                positive_weight=0.4,
                lower_bound=0.0,
                upper_bound=1.0,
                bin_ctr_in_use_after=10000,
                bin_ctr_weight_value=0.9995,
            )

            torch.testing.assert_close(
                calibrated_prediction_gpu,
                expected_calibrated_prediction.cuda(),
                rtol=error_tolerance,
                atol=error_tolerance,
            )

            self.assertTrue(
                torch.equal(
                    bin_ids_gpu.long(),
                    expected_bin_ids.cuda(),
                )
            )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        data_type=st.sampled_from([torch.bfloat16, torch.half, torch.float32]),
        segment_value_type=st.sampled_from([torch.int, torch.long]),
        segment_length_type=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_histogram_binning_calibration_by_feature(
        self,
        data_type: torch.dtype,
        segment_value_type: torch.dtype,
        segment_length_type: torch.dtype,
    ) -> None:
        num_bins = 5000
        num_segments = 42

        logit = torch.tensor([-0.0018, 0.0085, 0.0090, 0.0003, 0.0029]).type(data_type)

        segment_value = torch.tensor([40, 31, 32, 13, 31]).type(segment_value_type)
        lengths = torch.tensor([[1], [1], [1], [1], [1]]).type(segment_length_type)

        num_interval = num_bins * (num_segments + 1)
        bin_num_examples = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)
        bin_num_positives = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)

        (
            calibrated_prediction,
            bin_ids,
        ) = torch.ops.fbgemm.histogram_binning_calibration_by_feature(
            logit=logit,
            segment_value=segment_value,
            segment_lengths=lengths,
            num_segments=num_segments,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            num_bins=num_bins,
            positive_weight=0.4,
            lower_bound=0.0,
            upper_bound=1.0,
            bin_ctr_in_use_after=10000,
            bin_ctr_weight_value=0.9995,
        )

        expected_calibrated_prediction = torch.tensor(
            [0.2853, 0.2875, 0.2876, 0.2858, 0.2863]
        ).type(data_type)
        expected_bin_ids = torch.tensor(
            [206426, 161437, 166437, 71428, 161431], dtype=torch.long
        )

        error_tolerance = 1e-03
        if data_type == torch.bfloat16:
            # Due to smaller significand bits.
            error_tolerance = 1e-02

            expected_bin_ids = torch.tensor(
                [206426, 161438, 166438, 71430, 161430], dtype=torch.long
            )

        torch.testing.assert_close(
            calibrated_prediction,
            expected_calibrated_prediction,
            rtol=error_tolerance,
            atol=error_tolerance,
        )

        self.assertTrue(
            torch.equal(
                bin_ids.long(),
                expected_bin_ids,
            )
        )

        if torch.cuda.is_available():
            (
                calibrated_prediction_gpu,
                bin_ids_gpu,
            ) = torch.ops.fbgemm.histogram_binning_calibration_by_feature(
                logit=logit.cuda(),
                segment_value=segment_value.cuda(),
                segment_lengths=lengths.cuda(),
                num_segments=num_segments,
                bin_num_examples=bin_num_examples.cuda(),
                bin_num_positives=bin_num_positives.cuda(),
                num_bins=num_bins,
                positive_weight=0.4,
                lower_bound=0.0,
                upper_bound=1.0,
                bin_ctr_in_use_after=10000,
                bin_ctr_weight_value=0.9995,
            )

            torch.testing.assert_close(
                calibrated_prediction_gpu,
                expected_calibrated_prediction.cuda(),
                rtol=error_tolerance,
                atol=error_tolerance,
            )

            self.assertTrue(
                torch.equal(
                    bin_ids_gpu.long(),
                    expected_bin_ids.cuda(),
                )
            )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        data_type=st.sampled_from([torch.bfloat16, torch.half, torch.float32]),
        segment_value_type=st.sampled_from([torch.int, torch.long]),
        segment_length_type=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_generic_histogram_binning_calibration_by_feature(
        self,
        data_type: torch.dtype,
        segment_value_type: torch.dtype,
        segment_length_type: torch.dtype,
    ) -> None:
        num_bins = 5000
        num_segments = 42

        logit = torch.tensor([-0.0018, 0.0085, 0.0090, 0.0003, 0.0029]).type(data_type)

        segment_value = torch.tensor([40, 31, 32, 13, 31]).type(segment_value_type)
        lengths = torch.tensor([[1], [1], [1], [1], [1]]).type(segment_length_type)

        num_interval = num_bins * (num_segments + 1)
        bin_num_examples = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)
        bin_num_positives = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)

        lower_bound = 0.0
        upper_bound = 1.0
        w = (upper_bound - lower_bound) / num_bins
        bin_boundaries = torch.arange(
            lower_bound + w, upper_bound - w / 2, w, dtype=torch.float64
        )

        (
            calibrated_prediction,
            bin_ids,
        ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            logit=logit,
            segment_value=segment_value,
            segment_lengths=lengths,
            num_segments=num_segments,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            bin_boundaries=bin_boundaries,
            positive_weight=0.4,
            bin_ctr_in_use_after=10000,
            bin_ctr_weight_value=0.9995,
        )

        expected_calibrated_prediction = torch.tensor(
            [0.2853, 0.2875, 0.2876, 0.2858, 0.2863]
        ).type(data_type)
        expected_bin_ids = torch.tensor(
            [206426, 161437, 166437, 71428, 161431], dtype=torch.long
        )

        error_tolerance = 1e-03
        if data_type == torch.bfloat16:
            # Due to smaller significand bits.
            error_tolerance = 1e-02

            expected_bin_ids = torch.tensor(
                [206426, 161438, 166438, 71430, 161430], dtype=torch.long
            )

        torch.testing.assert_close(
            calibrated_prediction,
            expected_calibrated_prediction,
            rtol=error_tolerance,
            atol=error_tolerance,
        )

        self.assertTrue(
            torch.equal(
                bin_ids.long(),
                expected_bin_ids,
            )
        )

        if torch.cuda.is_available():
            (
                calibrated_prediction_gpu,
                bin_ids_gpu,
            ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
                logit=logit.cuda(),
                segment_value=segment_value.cuda(),
                segment_lengths=lengths.cuda(),
                num_segments=num_segments,
                bin_num_examples=bin_num_examples.cuda(),
                bin_num_positives=bin_num_positives.cuda(),
                bin_boundaries=bin_boundaries.cuda(),
                positive_weight=0.4,
                bin_ctr_in_use_after=10000,
                bin_ctr_weight_value=0.9995,
            )

            torch.testing.assert_close(
                calibrated_prediction_gpu,
                expected_calibrated_prediction.cuda(),
                rtol=error_tolerance,
                atol=error_tolerance,
            )

            self.assertTrue(
                torch.equal(
                    bin_ids_gpu.long(),
                    expected_bin_ids.cuda(),
                )
            )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        data_type=st.sampled_from([torch.bfloat16, torch.half, torch.float32]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_generic_histogram_binning_calibration_by_feature_cpu_gpu(
        self,
        data_type: torch.dtype,
    ) -> None:
        num_logits = random.randint(8, 16)
        num_bins = random.randint(3, 8)
        num_segments = random.randint(3, 8)
        positive_weight = random.uniform(0.1, 1.0)
        bin_ctr_in_use_after = random.randint(0, 10)
        bin_ctr_weight_value = random.random()

        logit = torch.randn(num_logits).type(data_type)

        lengths = torch.randint(0, 2, (num_logits,))
        segment_value = torch.randint(-3, num_segments + 3, (sum(lengths),))

        num_interval = num_bins * (num_segments + 1)
        bin_num_positives = torch.randint(0, 10, (num_interval,)).double()
        bin_num_examples = (
            bin_num_positives + torch.randint(0, 10, (num_interval,)).double()
        )

        lower_bound = 0.0
        upper_bound = 1.0
        w = (upper_bound - lower_bound) / num_bins
        bin_boundaries = torch.arange(
            lower_bound + w, upper_bound - w / 2, w, dtype=torch.float64
        )

        (
            calibrated_prediction_cpu,
            bin_ids_cpu,
        ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            logit=logit,
            segment_value=segment_value,
            segment_lengths=lengths,
            num_segments=num_segments,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            bin_boundaries=bin_boundaries,
            positive_weight=positive_weight,
            bin_ctr_in_use_after=bin_ctr_in_use_after,
            bin_ctr_weight_value=bin_ctr_weight_value,
        )

        (
            calibrated_prediction_gpu,
            bin_ids_gpu,
        ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            logit=logit.cuda(),
            segment_value=segment_value.cuda(),
            segment_lengths=lengths.cuda(),
            num_segments=num_segments,
            bin_num_examples=bin_num_examples.cuda(),
            bin_num_positives=bin_num_positives.cuda(),
            bin_boundaries=bin_boundaries.cuda(),
            positive_weight=positive_weight,
            bin_ctr_in_use_after=bin_ctr_in_use_after,
            bin_ctr_weight_value=bin_ctr_weight_value,
        )

        torch.testing.assert_close(
            calibrated_prediction_cpu,
            calibrated_prediction_gpu.cpu(),
            rtol=1e-03,
            atol=1e-03,
        )

        self.assertTrue(
            torch.equal(
                bin_ids_cpu,
                bin_ids_gpu.cpu(),
            )
        )

    def test_segment_sum_csr(self) -> None:
        segment_sum_cpu = torch.ops.fbgemm.segment_sum_csr(
            2,
            torch.IntTensor([0, 2, 3, 5]),
            torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        )
        torch.testing.assert_close(
            segment_sum_cpu, torch.Tensor([10.0, 11.0, 34.0]), rtol=0, atol=0
        )
        if torch.cuda.is_available():
            segment_sum_cuda = torch.ops.fbgemm.segment_sum_csr(
                2,
                torch.IntTensor([0, 2, 3, 5]).cuda(),
                torch.Tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                ).cuda(),
            )
            torch.testing.assert_close(
                segment_sum_cuda.cpu(), torch.Tensor([10.0, 11.0, 34.0]), rtol=0, atol=0
            )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        batch_size=st.just(2),
        m=st.just(3),
        k=st.just(4),
        n=st.just(5),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_permute102_baddbmm_permute102(
        self,
        batch_size: int,
        m: int,
        k: int,
        n: int,
        use_cpu: bool,
    ) -> None:
        # baddbmm doesn't support half
        dtype = torch.float if use_cpu else torch.half
        device = torch.device("cpu" if use_cpu else "cuda")

        A = torch.rand((m, batch_size, k), dtype=dtype, device=device)
        B = torch.rand((batch_size, k, n), dtype=dtype, device=device)
        # bias_permute102 = torch.rand(batch_size, 1, n).half().cuda()
        # bias = bias_permute102.permute(1, 0, 2)

        bias = torch.rand((batch_size, n), dtype=dtype, device=device)
        bias_permute102 = bias.unsqueeze(1)
        # bias = bias_short.unsqueeze(0)

        A_permute102 = A.permute(1, 0, 2)
        C_permute102 = torch.baddbmm(bias_permute102, A_permute102, B)
        C_ref = C_permute102.permute(1, 0, 2)  # (m, batch_size, n)

        C = torch.ops.fbgemm.permute102_baddbmm_permute102(bias, A, B)
        torch.testing.assert_close(C.cpu(), C_ref.cpu())

    def _pack_segments_ref(
        self,
        lengths: torch.Tensor,
        tensor: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        lengths = lengths.numpy()
        sections = np.split(tensor, np.cumsum(lengths))
        max_length = np.max(lengths, initial=0) if max_length is None else max_length
        padded_arrs = []
        for arr in sections[:-1]:  # Last section is always a blank
            arr = arr[: min(max_length, len(arr)), ...]
            padded_arr = np.pad(
                arr,
                [(0, max(max_length - arr.shape[0], 0))]
                + ([(0, 0)] * (len(arr.shape) - 1)),
                constant_values=0,
            )
            padded_arrs.append(padded_arr)

        if len(padded_arrs) == 0:
            padded_arrs = torch.empty((0, 0) + tuple(tensor.shape[1:]))
        else:
            padded_arrs = torch.Tensor(np.stack(padded_arrs))

        # pyre-fixme[7]: Expected `ndarray` but got `Tensor`.
        return padded_arrs

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
    )
    @settings(deadline=None)
    def test_pack_segments(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
    ) -> None:
        input_raw = np.random.rand(batch_size, n, k)
        input_data = torch.tensor(input_raw, dtype=torch.float32, requires_grad=True)
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
        )
        max_length = lengths.max().item()

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data, lengths=lengths, max_length=max_length
        )

        packed_ref = self._pack_segments_ref(lengths, input_raw)

        # pyre-fixme[6]: For 2nd param expected `Tensor` but got `ndarray`.
        self.assertTrue(torch.equal(packed_tensor, packed_ref))

        grad_cpu = torch.tensor(
            np.random.uniform(low=0.01, high=0.5, size=packed_ref.shape).astype(
                np.float32
            )
        )
        # CPU backward
        packed_tensor.backward(grad_cpu)

        if gpu_available:
            packed_cuda = torch.ops.fbgemm.pack_segments(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))

            # GPU backward
            packed_cuda.backward(grad_cpu.cuda())

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        max_length=st.integers(1, 20),
    )
    @settings(deadline=None)
    def test_pack_segments_smaller_max_len(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        max_length: int,
    ) -> None:
        input_data = torch.tensor(np.random.rand(batch_size, n, k), dtype=torch.float32)
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
        )

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data,
            lengths=lengths,
            max_length=max_length,
        )
        self.assertEqual(packed_tensor.shape, (divisions, max_length, n, k))

        packed_ref = self._pack_segments_ref(
            lengths,
            input_data,
            max_length=max_length,
        )
        # pyre-fixme[6]: For 2nd param expected `Tensor` but got `ndarray`.
        self.assertTrue(torch.equal(packed_tensor, packed_ref))

        if gpu_available:
            packed_cuda = torch.ops.fbgemm.pack_segments(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))

    # pyre-ignore [56]
    @given(
        N=st.integers(1, 32),
        shape=st.one_of(
            st.lists(st.integers(1, 128), max_size=1),
            st.lists(st.integers(1, 16), min_size=2, max_size=2),
        ),
        dtype=st.sampled_from([torch.float, torch.half, torch.double]),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        consecutive_indices=st.booleans(),
        skip_indices_sorting_fwd=st.booleans(),
        use_inference_mode=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_index_select_dim0(
        self,
        N: int,
        shape: List[int],
        dtype: torch.dtype,
        use_cpu: bool,
        consecutive_indices: bool,
        skip_indices_sorting_fwd: bool,
        use_inference_mode: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        U = random.randint(0, N + 1)

        kwargs = {}
        if consecutive_indices:
            start = np.random.randint(0, U)
            length = np.random.randint(1, U - start + 1)
            indices = list(range(start, start + length))
            np_arr = np.array(indices)
            for _ in range(N - U):
                indices.append(np.random.randint(start, start + length))
                np_arr = np.array(indices)
                np.random.shuffle(np_arr)
            indices = torch.from_numpy(np_arr).to(torch.int).to(device)
            kwargs["consecutive_range_start"] = start
            kwargs["consecutive_range_length"] = length
        else:
            indices = torch.randint(U, (N,), device=device)

        kwargs["skip_indices_sorting_fwd"] = skip_indices_sorting_fwd

        input = torch.rand((U,) + tuple(shape), dtype=dtype, device=device)

        with torch.inference_mode() if use_inference_mode else contextlib.nullcontext():
            output_ref = torch.ops.fbgemm.index_select_dim0(input, indices, **kwargs)
            output = torch.index_select(input, 0, indices)

            torch.testing.assert_close(output, output_ref)

        if not use_inference_mode:
            gradcheck_args = [
                input.clone().detach().double().requires_grad_(True),
                indices,
            ]
            for k in kwargs:
                gradcheck_args.append(kwargs[k])

            torch.autograd.gradcheck(torch.ops.fbgemm.index_select_dim0, gradcheck_args)

    # pyre-ignore [56]
    @given(
        num_indices=st.integers(1, 32),
        num_input_rows=st.integers(1, 32),
        shape=st.lists(st.integers(1, 32), min_size=1, max_size=2),
        dtype=st.sampled_from([torch.float, torch.half, torch.double]),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        num_groups=st.integers(1, 32),
        use_var_cols=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_group_index_select_dim0(
        self,
        num_indices: int,
        num_input_rows: int,
        shape: List[int],
        dtype: torch.dtype,
        use_cpu: bool,
        num_groups: int,
        use_var_cols: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")

        input_group: List[torch.Tensor] = []
        input_ref_group: List[torch.Tensor] = []
        indices_group: List[torch.Tensor] = []
        grad_group: List[torch.Tensor] = []
        for _ in range(num_groups):
            indices = torch.randint(num_input_rows, (num_indices,), device=device)
            assert indices.max() < num_input_rows

            if use_var_cols:
                var_dim = random.randint(0, len(shape) - 1)
                new_shape = random.randint(1, 32)
                shape[var_dim] = new_shape
            indices_group.append(indices)
            input = torch.rand(
                (num_input_rows,) + tuple(shape), dtype=dtype, device=device
            )
            input_ref = input.clone().detach()

            input.requires_grad = True
            input_ref.requires_grad = True

            input_group.append(input)
            input_ref_group.append(input_ref)

            grad = torch.rand((num_indices,) + tuple(shape), dtype=dtype, device=device)
            grad_group.append(grad)

        # Test forward
        output_ref_group = []
        for input, indices in zip(input_ref_group, indices_group):
            output_ref_group.append(torch.index_select(input, 0, indices))

        output_group = torch.ops.fbgemm.group_index_select_dim0(
            input_group, indices_group
        )

        # Test backward
        for out, grad in zip(output_ref_group, grad_group):
            out.backward(grad)

        cat_output = torch.concat([output.flatten() for output in output_group])
        cat_grad = torch.concat([grad.flatten() for grad in grad_group])
        cat_output.backward(cat_grad)

        def compare_tensor_groups(
            test_group: List[torch.Tensor],
            ref_group: List[torch.Tensor],
            tensor_type: str,
            tols: Dict["str", float],
        ) -> None:
            passed = True
            failure_count = 0
            for i, (test, ref) in enumerate(zip(test_group, ref_group)):
                # pyre-ignore [6]
                if not torch.allclose(test, ref, **tols):
                    passed = False
                    failure_count += 1
                    print(
                        f"FAILED: group {i} {tensor_type} ({dtype}), "
                        f"input shape {input_group[i].shape}, indices "
                        f"{indices_group[i]}, test {test}, ref {ref}"
                    )
            assert (
                passed
            ), f"{failure_count}/{num_groups} groups of {tensor_type} failed"

        compare_tensor_groups(
            output_group, output_ref_group, "activation", {"rtol": 0, "atol": 0}
        )
        compare_tensor_groups(
            # pyre-ignore [6]
            [i.grad for i in input_group],
            # pyre-ignore [6]
            [i.grad for i in input_ref_group],
            "gradient",
            {"rtol": 1e-02, "atol": 1e-02} if dtype == torch.half else {},
        )

    # pyre-ignore [56]
    @given(
        T=st.integers(1, 5),
        B=st.integers(1, 5),
        L=st.integers(1, 5),
    )
    @settings(max_examples=20, deadline=None)
    def test_bottom_unique_k_per_row(
        self,
        T: int,
        B: int,
        L: int,
    ) -> None:
        E = 1000000
        all_indices = (np.random.zipf(a=1.15, size=(T, B, 3 * L)) - 1) % E
        all_indices_deduped = torch.ops.fbgemm.bottom_k_per_row(
            torch.as_tensor(all_indices), torch.tensor([0, L], dtype=torch.long), True
        )
        for index_tuple in itertools.product(range(T), range(B)):
            # sample without replacement from
            # https://stats.stackexchange.com/questions/20590/how-do-i-sample-without-replacement-using-a-sampling-with-replacement-function
            r = set()
            for x in all_indices[index_tuple]:
                if x not in r:
                    r.add(x)
                    if len(r) == L:
                        break
            assert (len(r)) == L, "too skewed distribution (alpha too big)"
            all_indices[index_tuple][:L] = sorted(r)
        all_indices_deduped_ref = torch.as_tensor(all_indices[:, :, :L])
        torch.testing.assert_close(all_indices_deduped, all_indices_deduped_ref)


if __name__ == "__main__":
    unittest.main()
