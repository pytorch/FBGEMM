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
from typing import Optional

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


extend_test_class(PermuteIndicesTest)

if __name__ == "__main__":
    unittest.main()
