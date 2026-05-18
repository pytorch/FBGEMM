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
from itertools import accumulate
from typing import Callable, Optional

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import (
    extend_test_class,
    open_source,
    permute_indices_ref_,
    permute_scripted,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable, on_oss_clang
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable, on_oss_clang


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
        length_splits: Optional[list[torch.Tensor]] = None
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
                    # pyre-fixme[61]: `batch_sizes` is undefined, or not always defined.
                    for b in range(batch_sizes[w]):
                        # pyre-fixme[61]: `batch_sizes` is undefined, or not always
                        #  defined.
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
            weights_cuda = (
                weights.cuda() if (has_weight and weights is not None) else None
            )
            if is_1D:
                (
                    permuted_lengths_gpu,
                    permuted_indices_gpu,
                    permuted_weights_gpu,
                ) = torch.ops.fbgemm.permute_1D_sparse_data(
                    permute.cuda(),
                    lengths.cuda(),
                    indices.cuda(),
                    weights_cuda,
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
                    weights_cuda,
                    None,
                )
            torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                self.assertIsNone(permuted_weights_gpu)

    @given(
        B=st.integers(min_value=2, max_value=20),
        T=st.integers(min_value=2, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_indices_non_contiguous(
        self,
        B: int,
        T: int,
        L: int,
        long_index: bool,
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)

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

        def create_non_contiguous(x: torch.Tensor) -> torch.Tensor:
            # Create a diluted tensor with 2x elements, and then take every other element
            # with the value from the original tensor. For example, if x = [1, 2, 3, 4],
            # then the diluted tensor is [1, 0, 2, 0, 3, 0, 4, 0].
            diluted = x.new_zeros(x.numel() * 2).flatten()
            diluted[::2] = x.flatten()
            # Returns the sliced tensor, which is non-contiguous.
            return diluted[::2].view(x.shape)

        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
            # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
        ) = permute_indices_ref_(lengths, indices, None, permute.long())

        permute_gpu = create_non_contiguous(permute.cuda())
        lengths_gpu = create_non_contiguous(lengths.cuda())
        indices_gpu = create_non_contiguous(indices.cuda())
        self.assertFalse(permute_gpu.is_contiguous())
        self.assertFalse(lengths_gpu.is_contiguous())
        self.assertFalse(indices_gpu.is_contiguous())

        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute_gpu,
            lengths_gpu,
            indices_gpu,
            None,
            None,
        )
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_ref)
        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_ref)
        self.assertIsNone(permuted_weights_gpu)

    # TorchScript has different behaviors than eager mode. We can see undefined
    # models returned. So we need to add a unittest to ensure the op return
    # real None, not an undefined tensor.
    @unittest.skipIf(*gpu_unavailable)
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

    @unittest.skipIf(*on_oss_clang)
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
            weights_cuda = (
                weights.cuda() if (has_weight and weights is not None) else None
            )
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights_cuda,
            )
            torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_cpu is None

    @given(
        num_segments=st.integers(min_value=20, max_value=100),
        max_segment_length=st.integers(min_value=100, max_value=1000),
        index_dtype=st.sampled_from([torch.int32, torch.int64, torch.float32]),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=100, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_1D_sparse_data_vec(
        self,
        num_segments: int,
        max_segment_length: int,
        index_dtype: torch.dtype,
        has_weight: bool,
    ) -> None:
        """
        Test vectorized permute_1D_sparse_data kernel with vec4 optimization.

        Validates:
        - Correctness for various segment lengths (tests vec4 path and remainder handling)
        - Alignment-based vectorization (vec4 when aligned, scalar fallback when misaligned)
        - With and without weights (tests weights_vec4_aligned short-circuit logic)
        - Different index types (float vs int64)
        - Edge cases: segment lengths at vec4 boundaries (1, 3, 4, 5, 8, 15, 16, etc.)
        """

        # Generate variable-length segments to test vectorization
        lengths = torch.randint(
            low=max_segment_length // 2,
            high=max_segment_length,
            size=(num_segments,),
            dtype=torch.int32,
        )
        total_indices = int(lengths.sum().item())
        # Generate indices
        if index_dtype == torch.float32:
            indices = torch.rand(total_indices, dtype=index_dtype)
        else:
            indices = torch.randint(
                low=0,
                high=2**31 - 1,
                size=(total_indices,),
                dtype=index_dtype,
            )

        # Generate optional weights
        weights = torch.rand(total_indices, dtype=torch.float32) if has_weight else None

        # Generate random permutation
        permute_list = list(range(num_segments))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # CPU reference (uses scalar kernel)
        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_1D_sparse_data(
            permute, lengths, indices, weights, None
        )

        # GPU vectorized kernel (uses vec4 when aligned)
        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_1D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights.cuda() if has_weight and weights is not None else None,
            None,
        )

        # Validate correctness
        torch.testing.assert_close(
            permuted_lengths_gpu.cpu(),
            permuted_lengths_cpu,
        )
        torch.testing.assert_close(
            permuted_indices_gpu.cpu(),
            permuted_indices_cpu,
        )

        if has_weight:
            torch.testing.assert_close(
                permuted_weights_gpu.cpu(),
                permuted_weights_cpu,
            )
        else:
            self.assertIsNone(permuted_weights_gpu)
            self.assertIsNone(permuted_weights_cpu)

        # Test edge cases with specific segment lengths at vec4 boundaries
        # This validates remainder handling (segment_length % 4 = 0, 1, 2, 3)
        edge_case_lengths = [1, 3, 4, 5, 15, 16, 17, 63, 64, 127, 128]
        for segment_length in edge_case_lengths:
            lengths_edge = torch.tensor([segment_length], dtype=torch.int32)
            if index_dtype == torch.float32:
                indices_edge = torch.rand(segment_length, dtype=index_dtype)
            else:
                indices_edge = torch.randint(
                    0, 2**31 - 1, size=(segment_length,), dtype=index_dtype
                )
            weights_edge = (
                torch.rand(segment_length, dtype=torch.float32) if has_weight else None
            )
            permute_edge = torch.IntTensor([0])

            (
                permuted_lengths_cpu_edge,
                permuted_indices_cpu_edge,
                permuted_weights_cpu_edge,
            ) = torch.ops.fbgemm.permute_1D_sparse_data(
                permute_edge, lengths_edge, indices_edge, weights_edge, None
            )

            weights_edge_cuda = (
                weights_edge.cuda()
                if (has_weight and weights_edge is not None)
                else None
            )
            (
                permuted_lengths_gpu_edge,
                permuted_indices_gpu_edge,
                permuted_weights_gpu_edge,
            ) = torch.ops.fbgemm.permute_1D_sparse_data(
                permute_edge.cuda(),
                lengths_edge.cuda(),
                indices_edge.cuda(),
                weights_edge_cuda,
                None,
            )
            torch.testing.assert_close(
                permuted_lengths_gpu_edge.cpu(),
                permuted_lengths_cpu_edge,
            )
            torch.testing.assert_close(
                permuted_indices_gpu_edge.cpu(),
                permuted_indices_cpu_edge,
            )

            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_gpu_edge.cpu(),
                    permuted_weights_cpu_edge,
                )

    @given(
        num_segments=st.integers(min_value=20, max_value=100),
        max_segment_length=st.integers(min_value=10, max_value=200),
        index_dtype=st.sampled_from([torch.int32, torch.int64, torch.float32]),
        weights_columns=st.sampled_from([2, 3, 4, 7, 8]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_1D_sparse_data_vec_2d_weights(
        self,
        num_segments: int,
        max_segment_length: int,
        index_dtype: torch.dtype,
        weights_columns: int,
    ) -> None:
        """
        Test vectorized permute_1D_sparse_data kernel with 2D weights.

        Validates that the vec kernel correctly handles 2D weights tensors of shape
        [total_indices, W] for various W:
        - W=2, 3, 7: odd widths fall back to scalar path (weights_columns % 4 != 0)
        - W=4, 8:    divisible-by-4 widths use the vec4 path

        Uses a Python reference implementation to avoid dependence on the CPU kernel,
        which may flatten 2D weights differently.
        """

        def permute_1d_ref(
            permute: torch.Tensor,
            lengths: torch.Tensor,
            indices: torch.Tensor,
            weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Python reference: permute segments in 1D sparse data with 2D weights."""
            offsets = torch.cat(
                [torch.zeros(1, dtype=torch.int64), lengths.long().cumsum(0)]
            )
            permuted_lengths = lengths[permute.long()]
            out_offsets = torch.cat(
                [torch.zeros(1, dtype=torch.int64), permuted_lengths.long().cumsum(0)]
            )
            total = int(permuted_lengths.sum().item())
            permuted_indices = torch.empty(total, dtype=indices.dtype)
            permuted_weights = torch.empty(total, weights_columns, dtype=weights.dtype)
            for i, p in enumerate(permute.tolist()):
                src = int(offsets[p].item())
                length = int(lengths[p].item())
                dst = int(out_offsets[i].item())
                permuted_indices[dst : dst + length] = indices[src : src + length]
                permuted_weights[dst : dst + length] = weights[src : src + length]
            return permuted_lengths, permuted_indices, permuted_weights

        lengths = torch.randint(
            low=max_segment_length // 2,
            high=max_segment_length,
            size=(num_segments,),
            dtype=torch.int32,
        )
        total_indices = int(lengths.sum().item())

        if index_dtype == torch.float32:
            indices = torch.rand(total_indices, dtype=index_dtype)
        else:
            indices = torch.randint(
                low=0,
                high=2**31 - 1,
                size=(total_indices,),
                dtype=index_dtype,
            )

        # 2D weights: shape [total_indices, weights_columns]
        weights = torch.rand(total_indices, weights_columns, dtype=torch.float32)

        permute_list = list(range(num_segments))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # Python reference
        ref_lengths, ref_indices, ref_weights = permute_1d_ref(
            permute, lengths, indices, weights
        )

        # GPU vectorized kernel
        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_1D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights.cuda(),
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), ref_lengths)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), ref_indices)
        torch.testing.assert_close(permuted_weights_gpu.cpu(), ref_weights)

        # Verify output shape is [permuted_total, weights_columns]
        self.assertEqual(permuted_weights_gpu.dim(), 2)
        self.assertEqual(permuted_weights_gpu.size(1), weights_columns)

        # Edge cases: specific segment lengths at vec4 boundaries for 2D weights
        # The effective element count per segment is segment_length * weights_columns,
        # so boundary behavior differs from the 1D case.
        for seg_len in [1, 3, 4, 5, 8, 16]:
            lengths_edge = torch.tensor([seg_len], dtype=torch.int32)
            if index_dtype == torch.float32:
                indices_edge = torch.rand(seg_len, dtype=index_dtype)
            else:
                indices_edge = torch.randint(
                    0, 2**31 - 1, size=(seg_len,), dtype=index_dtype
                )
            weights_edge = torch.rand(seg_len, weights_columns, dtype=torch.float32)
            permute_edge = torch.IntTensor([0])

            ref_lengths_edge, ref_indices_edge, ref_weights_edge = permute_1d_ref(
                permute_edge, lengths_edge, indices_edge, weights_edge
            )
            (
                permuted_lengths_gpu_edge,
                permuted_indices_gpu_edge,
                permuted_weights_gpu_edge,
            ) = torch.ops.fbgemm.permute_1D_sparse_data(
                permute_edge.cuda(),
                lengths_edge.cuda(),
                indices_edge.cuda(),
                weights_edge.cuda(),
                None,
            )
            torch.testing.assert_close(
                permuted_lengths_gpu_edge.cpu(), ref_lengths_edge
            )
            torch.testing.assert_close(
                permuted_indices_gpu_edge.cpu(), ref_indices_edge
            )
            torch.testing.assert_close(
                permuted_weights_gpu_edge.cpu(), ref_weights_edge
            )

    @given(
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_vec_remainder(
        self,
        long_index: bool,
        has_weight: bool,
    ) -> None:
        """
        Test vectorized permute_2D_sparse_data kernel vec4 remainder handling.

        Uses hand-crafted segment lengths (0, 1, 2, 3, 4, 5, 7, 8) to explicitly
        exercise the vec4 vectorized path and scalar remainder path in
        permute_2D_data_kernel_vec. Lengths cover:
        - Pure remainder: 0, 1, 2, 3
        - Exact vec4: 4, 8
        - Mixed vec4 + remainder: 5, 7
        """
        index_dtype = torch.int64 if long_index else torch.int32
        T = 4  # number of tables
        # Hand-crafted lengths with specific remainder patterns
        # Shape [T, B=2] — each entry is a segment length
        lengths_list = [
            [0, 1],  # table 0: pure remainder (0 and 1)
            [2, 3],  # table 1: pure remainder (2 and 3)
            [4, 5],  # table 2: exact vec4 (4) and vec4+1 remainder (5)
            [7, 8],  # table 3: vec4+3 remainder (7) and exact 2xvec4 (8)
        ]
        lengths = torch.tensor(lengths_list, dtype=index_dtype)
        total = int(lengths.sum().item())

        indices = torch.randint(
            low=1,
            high=int(1e5),
            size=(total,),
            dtype=index_dtype,
        )
        weights = torch.rand(total, dtype=torch.float32) if has_weight else None

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # CPU reference
        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, None
        )

        # GPU (uses vectorized kernel)
        weights_cuda = weights.cuda() if has_weight and weights is not None else None
        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights_cuda,
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        if has_weight:
            torch.testing.assert_close(permuted_weights_gpu.cpu(), permuted_weights_cpu)
        else:
            self.assertIsNone(permuted_weights_gpu)

    @given(
        index_dtype=st.sampled_from([torch.int32, torch.int64, torch.float32]),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_large_segments(
        self,
        index_dtype: torch.dtype,
        has_weight: bool,
    ) -> None:
        """
        Test vectorized permute_2D_sparse_data kernel with large segments.

        Uses segment lengths in the hundreds to ensure the vec4 vectorized path
        in permute_2D_data_kernel_vec works correctly on realistic workloads.
        Tests all index dtypes (int32, int64, float32) and with/without weights.
        """
        T = 6  # number of tables
        # Large segment lengths to heavily exercise the vec4 loop
        # Shape [T, B=4]
        lengths_list = [
            [256, 512, 1000, 333],
            [128, 257, 513, 100],
            [1024, 750, 64, 999],
            [500, 501, 502, 503],
            [1000, 1, 0, 1000],
            [255, 256, 257, 1023],
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        total = int(lengths.sum().item())

        if index_dtype == torch.float32:
            indices = torch.rand(total, dtype=index_dtype)
        else:
            indices = torch.randint(
                low=0,
                high=2**31 - 1,
                size=(total,),
                dtype=index_dtype,
            )
        weights = torch.rand(total, dtype=torch.float32) if has_weight else None

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # CPU reference
        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, None
        )

        # GPU (uses vectorized kernel)
        weights_cuda = weights.cuda() if has_weight and weights is not None else None
        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights_cuda,
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        if has_weight:
            torch.testing.assert_close(permuted_weights_gpu.cpu(), permuted_weights_cpu)
        else:
            self.assertIsNone(permuted_weights_gpu)

    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_long2_remainder(
        self,
    ) -> None:
        """
        Test new int64 long2 vectorized path remainder handling.

        After the long4 -> long2 change, the vec width for 8-byte types is 2
        elements per thread. Lengths chosen to exercise:
        - Pure remainder (lens 1, 3): len % 2 == 1 stragglers
        - Exact vec width (lens 2, 4, 6, 8): len % 2 == 0
        - Mixed combinations (lens 0..9)

        Uses int64 indices AND torch.double weights (8-byte) to exercise both
        the new indices vec path AND the new weights vec path (weights_vec_t
        = long2 for 8-byte types). int64 weights are not supported by the CPU
        reference dispatch, so double is the only 8-byte weights dtype that
        can be cross-checked CPU vs GPU.
        """
        index_dtype = torch.int64
        weights_dtype = torch.double  # 8-byte, hits long2 weights vec path
        # Shape [T=2, B=10]: each row covers all interesting boundary cases
        # row 0: ascending lens 0..9, row 1: descending to add variety
        lengths_list = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ]
        T = len(lengths_list)
        lengths = torch.tensor(lengths_list, dtype=index_dtype)
        total = int(lengths.sum().item())

        indices = torch.randint(
            low=1,
            high=int(1e9),
            size=(total,),
            dtype=index_dtype,
        )
        # 8-byte (double) weights to exercise weights_vec_t = long2 path
        weights = torch.rand(total, dtype=weights_dtype)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # CPU reference
        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, None
        )

        # GPU (uses vectorized long2 kernel for 8-byte indices and weights)
        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights.cuda(),
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        torch.testing.assert_close(permuted_weights_gpu.cpu(), permuted_weights_cpu)

    @given(
        has_weight=st.booleans(),
        weights_dtype=st.sampled_from([torch.float32, torch.double, torch.float16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_int64_large_segments(
        self,
        has_weight: bool,
        weights_dtype: torch.dtype,
    ) -> None:
        """
        Test the int64 hot path (long2 vec) with long segments.

        This is the path being optimized: int64 indices with segment lengths
        in the thousands. Sweeps weights dtypes (float32, double, float16) to
        cover both 8-byte weights (double -> long2 vec) and 4-byte weights
        (float -> float4 vec) and 2-byte weights (float16 -> float4 vec).
        The CPU reference dispatch (FBGEMM_DISPATCH_FLOAT_HALF_AND_DOUBLE for
        weights) does not support integer weights, so they are excluded here.
        """
        index_dtype = torch.int64
        T = 6
        # Long segments (several thousand) to exercise the vec loop hard.
        lengths_list = [
            [4096, 8192, 2048, 1024],
            [3000, 5000, 7000, 2500],
            [6000, 1500, 4500, 3500],
            [2049, 4097, 8193, 1025],  # odd lengths to hit remainder
            [10000, 5000, 2500, 1250],
            [3333, 6666, 9999, 1111],
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        total = int(lengths.sum().item())

        indices = torch.randint(
            low=0,
            high=2**62,
            size=(total,),
            dtype=index_dtype,
        )

        weights = torch.rand(total, dtype=weights_dtype) if has_weight else None

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # CPU reference
        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, None
        )

        # GPU (uses vectorized long2 kernel)
        weights_cuda = weights.cuda() if has_weight and weights is not None else None
        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights_cuda,
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        if has_weight:
            torch.testing.assert_close(permuted_weights_gpu.cpu(), permuted_weights_cpu)
        else:
            self.assertIsNone(permuted_weights_gpu)

    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_bf16_correctness(
        self,
    ) -> None:
        """
        Test BF16 "indices" (2-byte type) correctness.

        Note: BF16 indices always route through the scalar fallback inside
        permute_2D_data_kernel_vec because the alignment guard at
        sparse_permute_2d.cu:118-123 requires sizeof(indices_t) in {4, 8}.
        This test therefore exercises the scalar-fallback path for 2-byte
        indices, not the vec path. The companion
        test_permute_2D_indices_2byte_weights_correctness covers the actual
        vec-path correctness fix from D105336279, which affects 2-byte
        weights (whose alignment guard has no sizeof restriction).
        """
        T = 4
        # Mix of even/odd-aligned segment lengths.
        # With BF16 (2 bytes), float4 holds 8 elements per vec.
        # Lengths chosen to span < 1 vec, exact vec, and vec + remainder.
        lengths_list = [
            [1, 7, 8, 9],
            [15, 16, 17, 23],
            [64, 100, 256, 333],
            [512, 1000, 2000, 4096],
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        total = int(lengths.sum().item())

        # BF16 "indices" — operator treats them as opaque 2-byte payloads.
        indices = torch.rand(total, dtype=torch.bfloat16)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # CPU reference
        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, None, None
        )

        # GPU (uses vectorized kernel — currently buggy for 2-byte)
        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            None,
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        self.assertIsNone(permuted_weights_gpu)

    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_half_correctness(
        self,
    ) -> None:
        """
        Test Half (float16) "indices" (2-byte type) correctness.

        Sibling of test_permute_2D_indices_bf16_correctness. Covers the OTHER
        2-byte dtype in FBGEMM_DISPATCH_ALL_TYPES dispatch. After the
        sizeof(vec_t)/sizeof(elem_t) fix, kIndicesVecWidth for half is 8
        (float4 / sizeof(half) = 16 / 2 = 8), not the previously-hardcoded 4.
        """
        T = 4
        # Same shape pattern as bf16 test: span < 1 vec, exact vec, vec + remainder.
        lengths_list = [
            [1, 7, 8, 9],
            [15, 16, 17, 23],
            [64, 100, 256, 333],
            [512, 1000, 2000, 4096],
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        total = int(lengths.sum().item())

        # Half "indices" — operator treats them as opaque 2-byte payloads.
        indices = torch.rand(total, dtype=torch.float16)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, None, None
        )

        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            None,
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        self.assertIsNone(permuted_weights_gpu)

    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_double_correctness(
        self,
    ) -> None:
        """
        Test Double (float64) "indices"-style payload via weights (8-byte type).

        Double is only reachable via the weights dispatch (FBGEMM_DISPATCH_ALL_
        TYPES_AND_DOUBLE), not indices. This test pairs int64 indices with
        double weights so BOTH the indices vec path (long2, kWidth=2) and the
        weights vec path (long2, kWidth=2) are exercised on segments chosen
        to cover all len % 2 remainders.
        """
        index_dtype = torch.int64
        weights_dtype = torch.double
        T = 4
        # Mix of segment lengths covering remainders for kWidth=2.
        lengths_list = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [100, 101, 255, 256],
            [1023, 1024, 2049, 4096],
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        total = int(lengths.sum().item())

        indices = torch.randint(low=0, high=2**62, size=(total,), dtype=index_dtype)
        weights = torch.rand(total, dtype=weights_dtype)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, None
        )

        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights.cuda(),
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        torch.testing.assert_close(permuted_weights_gpu.cpu(), permuted_weights_cpu)

    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_2byte_weights_correctness(
        self,
    ) -> None:
        """
        Test 2-byte weights correctness with weights_columns=1.

        Direct regression test for the D89161131 latent bug fixed by
        D105336279. Unlike 2-byte indices (which always route to the scalar
        fallback because of the alignment guard at sparse_permute_2d.cu:118-
        123 requiring sizeof(indices_t) in {4, 8}), 2-byte weights ARE
        reachable via the vec path: the weights_vec_aligned check at
        sparse_permute_2d.cu:125-129 has no sizeof restriction. Pre-fix:
        kWeightsVecWidth was hardcoded to 4 for non-8-byte types, but
        float4 holds 8 elements per 2-byte slot
        (sizeof(float4)/sizeof(half) = 16/2 = 8). The vec loop wrote 2x
        past each segment, silently corrupting adjacent segments' weights.
        Post-fix: width is sizeof(vec_t)/sizeof(elem_t) = 8.

        Uses Half (float16) weights as the testable proxy: the CPU
        dispatch (FBGEMM_DISPATCH_FLOAT_HALF_AND_DOUBLE) does not include
        BFloat16, so CPU cross-validation isn't available for BF16
        weights; the bug mechanics are identical for Half and BFloat16
        (both 2-byte, both routed through float4 vec).
        """
        T = 4
        # Lengths chosen to span < 1 vec, exact vec, vec + remainder
        # at kWeightsVecWidth = 8 (the post-fix width for 2-byte weights).
        lengths_list = [
            [1, 7, 8, 9],
            [15, 16, 17, 23],
            [64, 100, 256, 333],
            [512, 1000, 2000, 4096],
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        total = int(lengths.sum().item())

        # int64 indices + Half weights, weights_columns = 1 (1D weights).
        indices = torch.randint(low=0, high=2**62, size=(total,), dtype=torch.int64)
        weights = torch.rand(total, dtype=torch.float16)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, None
        )

        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights.cuda(),
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        torch.testing.assert_close(permuted_weights_gpu.cpu(), permuted_weights_cpu)

    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_mixed_width_weights(
        self,
    ) -> None:
        """
        Test mixed-width vec kernel: int64 indices (kWidth=2, long2) +
        float32 weights (kWidth=4, float4).

        The indices and weights vec counts/remainders are computed
        independently inside permute_2D_data_kernel_vec. This test verifies
        the two parallel vec loops produce consistent output when widths
        differ, across segment lengths that hit:
        - len % 2 == 1 (indices remainder)
        - len % 4 != 0 (weights remainder)
        - len divisible by both
        """
        T = 4
        # Lengths chosen so vec counts/remainders differ for the two widths.
        lengths_list = [
            [1, 2, 3, 4],  # all in remainder territory
            [5, 6, 7, 8],  # mix: 5 = 2v+1 (idx) / 1v+1 (wt)
            [9, 10, 11, 12],  # 11 = 5v+1 (idx) / 2v+3 (wt)
            [255, 256, 257, 1000],
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        total = int(lengths.sum().item())

        indices = torch.randint(low=0, high=2**62, size=(total,), dtype=torch.int64)
        weights = torch.rand(total, dtype=torch.float32)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, None
        )
        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights.cuda(),
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
        torch.testing.assert_close(permuted_weights_gpu.cpu(), permuted_weights_cpu)

    @given(
        weights_columns=st.sampled_from([2, 4, 8]),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_weights_columns_gt1(
        self,
        weights_columns: int,
        long_index: bool,
    ) -> None:
        """
        Test the scalar fallback dispatch (weights_columns > 1).

        permute_2D_sparse_data routes weights_columns > 1 to the scalar
        permute_2D_data_kernel, which is NOT touched by this diff. This is a
        no-regression smoke test confirming the other dispatch arm still
        works after the vec kernel changes.

        Uses a Python reference for cross-validation: the CPU op flattens 2D
        weights to 1D (different convention from the GPU), so we cannot
        compare GPU vs CPU directly.
        """
        index_dtype = torch.int64 if long_index else torch.int32
        T = 4
        lengths_list = [
            [1, 7, 8, 9],
            [15, 16, 17, 23],
            [64, 100, 256, 333],
            [512, 1000, 2000, 4096],
        ]
        lengths = torch.tensor(lengths_list, dtype=index_dtype)
        total = int(lengths.sum().item())

        indices = torch.randint(low=0, high=int(1e9), size=(total,), dtype=index_dtype)
        # 2D weights -> weights_columns > 1 -> scalar dispatch
        weights = torch.rand(total, weights_columns, dtype=torch.float32)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # Python reference (segment-by-segment copy).
        B = lengths.size(1)
        input_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64), lengths.view(-1).long().cumsum(0)]
        )
        permuted_lengths_ref = torch.index_select(
            lengths.view(T, -1), 0, permute.long()
        )
        out_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64),
                permuted_lengths_ref.view(-1).long().cumsum(0),
            ]
        )
        permuted_indices_ref = torch.empty(total, dtype=indices.dtype)
        permuted_weights_ref = torch.empty(total, weights_columns, dtype=weights.dtype)
        for i in range(permute.numel()):
            src_t = int(permute[i].item())
            for b in range(B):
                src_start = int(input_offsets[src_t * B + b].item())
                seg_len = int(lengths[src_t, b].item())
                dst_start = int(out_offsets[i * B + b].item())
                permuted_indices_ref[dst_start : dst_start + seg_len] = indices[
                    src_start : src_start + seg_len
                ]
                permuted_weights_ref[dst_start : dst_start + seg_len] = weights[
                    src_start : src_start + seg_len
                ]

        (
            permuted_lengths_gpu,
            permuted_indices_gpu,
            permuted_weights_gpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.cuda(),
            lengths.cuda(),
            indices.cuda(),
            weights.cuda(),
            None,
        )

        torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_ref)
        torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_ref)
        torch.testing.assert_close(permuted_weights_gpu.cpu(), permuted_weights_ref)
        # Confirm 2D shape preserved on GPU.
        self.assertEqual(permuted_weights_gpu.dim(), 2)
        self.assertEqual(permuted_weights_gpu.size(1), weights_columns)

    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_misalignment_fallback(
        self,
    ) -> None:
        """
        Test scalar fallback inside permute_2D_data_kernel_vec when offsets are
        not vec-aligned.

        The kernel falls back to scalar copy if the per-segment src/dst
        pointers are not aligned to alignof(vec_t). Using odd-only lengths
        (1, 3, 5, ...) creates cumulative offsets that are odd element counts,
        so for any vec_t whose alignment exceeds the element size, the second
        segment's pointer is misaligned. This forces the scalar branch.

        Covers all 5 indices dtypes in FBGEMM_DISPATCH_ALL_TYPES.
        """
        T = 4
        # Odd-only lengths -> cumulative offsets land on odd element counts.
        # For 8-byte indices (int64) with long2 (alignof 16B = 2 elems), an
        # odd elem-offset is misaligned. For 4-byte (int32/float) with float4
        # (alignof 16B = 4 elems), 1 mod 4 != 0 -> misaligned. For 2-byte
        # (bf16/half) with float4 (8 elems), 1 mod 8 != 0 -> misaligned.
        lengths_list = [
            [1, 3, 5, 7],
            [9, 11, 13, 15],
            [17, 19, 21, 23],
            [25, 27, 29, 31],
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        total = int(lengths.sum().item())

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        for indices_dtype in [
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ]:
            if indices_dtype in (torch.float32, torch.float16, torch.bfloat16):
                indices = torch.rand(total, dtype=indices_dtype)
            else:
                indices = torch.randint(
                    low=0, high=int(1e9), size=(total,), dtype=indices_dtype
                )

            (
                permuted_lengths_cpu,
                permuted_indices_cpu,
                permuted_weights_cpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute, lengths, indices, None, None
            )
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                None,
                None,
            )
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
            torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
            self.assertIsNone(permuted_weights_gpu)

    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_indices_stress_randomized(
        self,
    ) -> None:
        """
        Seeded-random stress test that sweeps dtypes, shapes, permutation
        styles, and has_weight flag.

        Goal is breadth — each iteration keeps shapes modest so the full
        loop finishes well under 60s. All dtypes from FBGEMM_DISPATCH_ALL_
        TYPES are exercised on the indices side; the weights side uses the
        CPU-supported dtypes (float32, float64, float16, bfloat16) so the
        CPU reference can validate.
        """
        rng = random.Random(0xFBE0)
        torch.manual_seed(0xFBE0)

        indices_dtypes = [
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ]
        # CPU weights dispatch supports only FBGEMM_DISPATCH_FLOAT_HALF_AND_DOUBLE
        # -> exclude bfloat16 / int weights so the CPU reference is available.
        weights_dtypes = [torch.float32, torch.float64, torch.float16]
        permutation_styles = ["random", "identity", "inverse", "with_repeats"]

        n_iters = 50
        for it in range(n_iters):
            T = rng.randint(1, 32)
            B = rng.randint(1, 16)
            # Cap max segment length to keep memory and time modest.
            max_seg = rng.randint(0, 200)
            indices_dtype = rng.choice(indices_dtypes)
            has_weight = rng.choice([True, False])
            weights_dtype = rng.choice(weights_dtypes) if has_weight else None
            style = rng.choice(permutation_styles)

            lengths = torch.randint(
                low=0, high=max_seg + 1, size=(T, B), dtype=torch.int32
            )
            total = int(lengths.sum().item())
            if total == 0:
                # Skip degenerate iter — the op behaviour on empty totals is
                # tested separately by the existing T=0/B=0 paths.
                continue

            if indices_dtype in (torch.float32, torch.float16, torch.bfloat16):
                indices = torch.rand(total, dtype=indices_dtype)
            else:
                indices = torch.randint(
                    low=0, high=int(1e9), size=(total,), dtype=indices_dtype
                )

            weights = torch.rand(total, dtype=weights_dtype) if has_weight else None

            if style == "random":
                permute_list = list(range(T))
                rng.shuffle(permute_list)
            elif style == "identity":
                permute_list = list(range(T))
            elif style == "inverse":
                permute_list = list(range(T - 1, -1, -1))
            else:  # with_repeats
                base = list(range(T))
                extra = [rng.randint(0, T - 1) for _ in range(rng.randint(0, T))]
                permute_list = base + extra
                rng.shuffle(permute_list)

            permute = torch.IntTensor(permute_list)

            (
                permuted_lengths_cpu,
                permuted_indices_cpu,
                permuted_weights_cpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute, lengths, indices, weights, None
            )
            weights_cuda = (
                weights.cuda() if has_weight and weights is not None else None
            )
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights_cuda,
                None,
            )

            msg = (
                f"iter={it} T={T} B={B} max_seg={max_seg} "
                f"indices_dtype={indices_dtype} has_weight={has_weight} "
                f"weights_dtype={weights_dtype} style={style}"
            )
            torch.testing.assert_close(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu, msg=msg
            )
            torch.testing.assert_close(
                permuted_indices_gpu.cpu(), permuted_indices_cpu, msg=msg
            )
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu, msg=msg
                )
            else:
                self.assertIsNone(permuted_weights_gpu)


# Skip opcheck wrappers for tests that:
# - exercise 2D weights (pre-existing fake-tensor metadata mismatch in
#   permute_2D_sparse_data — the fake meta impl returns 1D shape regardless
#   of input weights rank, same class of issue tracked elsewhere)
# - run many iterations and would multiply opcheck overhead unproductively
#   (stress test already does CPU-vs-GPU cross-check on every iteration).
# pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
additional_decorators: dict[str, list[Callable]] = {
    "test_aot_dispatch_dynamic__test_permute_2D_indices_weights_columns_gt1": [
        unittest.skip("fake tensor meta returns 1D shape regardless of weights rank")
    ],
    "test_faketensor__test_permute_2D_indices_weights_columns_gt1": [
        unittest.skip("fake tensor meta returns 1D shape regardless of weights rank")
    ],
    "test_schema__test_permute_2D_indices_weights_columns_gt1": [
        unittest.skip("fake tensor meta returns 1D shape regardless of weights rank")
    ],
    "test_autograd_registration__test_permute_2D_indices_weights_columns_gt1": [
        unittest.skip("fake tensor meta returns 1D shape regardless of weights rank")
    ],
    "test_aot_dispatch_dynamic__test_permute_2D_indices_stress_randomized": [
        unittest.skip("real test already covers CPU vs GPU; opcheck overhead too high")
    ],
    "test_faketensor__test_permute_2D_indices_stress_randomized": [
        unittest.skip("real test already covers CPU vs GPU; opcheck overhead too high")
    ],
    "test_schema__test_permute_2D_indices_stress_randomized": [
        unittest.skip("real test already covers CPU vs GPU; opcheck overhead too high")
    ],
    "test_autograd_registration__test_permute_2D_indices_stress_randomized": [
        unittest.skip("real test already covers CPU vs GPU; opcheck overhead too high")
    ],
}


# pyre-ignore[6]: `additional_decorators` type compatibility
extend_test_class(PermuteIndicesTest, additional_decorators)

if __name__ == "__main__":
    unittest.main()
