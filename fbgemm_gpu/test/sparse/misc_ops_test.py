#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import itertools
import random
import unittest
from typing import Type, Union

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_available


class MiscOpsTest(unittest.TestCase):
    @given(
        permute_size=st.integers(min_value=0, max_value=1000),
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

    def test_segment_sum_csr_empty_input(self) -> None:
        segment_sum_cpu = torch.ops.fbgemm.segment_sum_csr(
            0,
            torch.IntTensor([0]),
            torch.Tensor([]),
        )
        torch.testing.assert_close(segment_sum_cpu.numel(), 0, rtol=0, atol=0)

        if torch.cuda.is_available():
            segment_sum_cuda = torch.ops.fbgemm.segment_sum_csr(
                0,
                torch.IntTensor([0]).cuda(),
                torch.Tensor([]).cuda(),
            )
            torch.testing.assert_close(
                segment_sum_cuda.cpu().numel(), 0, rtol=0, atol=0
            )

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


extend_test_class(MiscOpsTest)

if __name__ == "__main__":
    unittest.main()
