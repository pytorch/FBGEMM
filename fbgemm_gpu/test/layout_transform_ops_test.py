#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity
from torch import Tensor

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_unavailable

except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    from fbgemm_gpu.test.test_utils import gpu_unavailable


MAX_EXAMPLES = 20


class LayoutTransformOpsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        D=st.integers(min_value=2, max_value=20),
        W=st.integers(min_value=1, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_recat_embedding_grad_output(self, B: int, T: int, D: int, W: int) -> None:
        num_features_per_rank = np.random.randint(low=1, high=20, size=(W,)).tolist()
        grad_output = torch.randn(B, sum(num_features_per_rank), D).float().cuda()
        grad_outputs_by_rank = grad_output.split(num_features_per_rank, dim=1)
        sharded_grad_output = torch.cat(
            [
                grad_output_by_rank.contiguous().view(-1)
                for grad_output_by_rank in grad_outputs_by_rank
            ],
            dim=0,
        )
        sharded_grad_output_impl = torch.ops.fbgemm.recat_embedding_grad_output(
            grad_output, num_features_per_rank
        )
        torch.testing.assert_close(
            sharded_grad_output_impl.cpu(), sharded_grad_output.cpu()
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]
    @given(
        B=st.integers(min_value=1, max_value=20),
        W=st.integers(min_value=1, max_value=20),
        cuda=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_recat_embedding_grad_output_mixed_D(
        self, B: int, W: int, cuda: bool
    ) -> None:
        num_features_per_rank = np.random.randint(low=1, high=20, size=(W,)).tolist()
        global_T = sum(num_features_per_rank)
        mixed_D_list = np.random.randint(low=1, high=10, size=(global_T,))
        grad_output = torch.randn(B, sum(mixed_D_list)).float().cuda()
        if cuda:
            grad_output = grad_output.cuda()
        num_feature_offsets_list = torch.tensor(
            [0] + np.cumsum(num_features_per_rank).tolist()
        )
        dim_sum_per_rank = [
            sum(
                mixed_D_list[
                    num_feature_offsets_list[i] : num_feature_offsets_list[i + 1]
                ]
            )
            for i in range(W)
        ]
        grad_outputs_by_rank = grad_output.split(dim_sum_per_rank, dim=1)
        sharded_grad_output = torch.cat(
            [
                grad_output_by_rank.contiguous().view(-1)
                for grad_output_by_rank in grad_outputs_by_rank
            ],
            dim=0,
        )
        sharded_grad_output_impl = torch.ops.fbgemm.recat_embedding_grad_output_mixed_D(
            grad_output, dim_sum_per_rank
        )
        torch.testing.assert_close(
            sharded_grad_output_impl.cpu(), sharded_grad_output.cpu()
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]
    @given(
        B=st.integers(min_value=1, max_value=20),
        W=st.integers(min_value=1, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_recat_embedding_grad_output_mixed_D_batch(self, B: int, W: int) -> None:
        num_features_per_rank = np.random.randint(low=1, high=20, size=(W,)).tolist()
        global_T = sum(num_features_per_rank)
        mixed_D_list = np.random.randint(low=1, high=10, size=(global_T,))
        grad_output = torch.randn(B, sum(mixed_D_list)).float().cuda()
        num_feature_offsets_list = torch.tensor(
            [0] + np.cumsum(num_features_per_rank).tolist()
        )
        dim_sum_per_rank = [
            sum(
                mixed_D_list[
                    num_feature_offsets_list[i] : num_feature_offsets_list[i + 1]
                ]
            )
            for i in range(W)
        ]
        # pyre-fixme[16]: Module `cuda` has no attribute `LongTensor`.
        dim_sum_per_rank_tensor = torch.cuda.LongTensor(dim_sum_per_rank)
        # pyre-fixme[16]: Module `cuda` has no attribute `LongTensor`.
        cumsum_dim_sum_per_rank_tensor = torch.cuda.LongTensor(
            np.cumsum([0] + dim_sum_per_rank)[:-1]
        )

        grad_outputs_by_rank = grad_output.split(dim_sum_per_rank, dim=1)
        sharded_grad_output = torch.cat(
            [
                grad_output_by_rank.contiguous().view(-1)
                for grad_output_by_rank in grad_outputs_by_rank
            ],
            dim=0,
        )
        sharded_grad_output_impl = (
            torch.ops.fbgemm.recat_embedding_grad_output_mixed_D_batch(
                grad_output.cuda(),
                dim_sum_per_rank_tensor.cuda(),
                cumsum_dim_sum_per_rank_tensor.cuda(),
            )
        )
        torch.testing.assert_close(
            sharded_grad_output_impl.cpu(), sharded_grad_output.cpu()
        )
        num_features_per_rank = np.random.randint(low=1, high=20, size=(W,)).tolist()
        global_T = sum(num_features_per_rank)
        mixed_D_list = np.random.randint(low=1, high=10, size=(global_T,))
        grad_output = torch.randn(B, sum(mixed_D_list)).float().cuda()
        num_feature_offsets_list = torch.tensor(
            [0] + np.cumsum(num_features_per_rank).tolist()
        )
        dim_sum_per_rank = [
            sum(
                mixed_D_list[
                    num_feature_offsets_list[i] : num_feature_offsets_list[i + 1]
                ]
            )
            for i in range(W)
        ]
        # pyre-fixme[16]: Module `cuda` has no attribute `LongTensor`.
        dim_sum_per_rank_tensor = torch.cuda.LongTensor(dim_sum_per_rank)
        # pyre-fixme[16]: Module `cuda` has no attribute `LongTensor`.
        cumsum_dim_sum_per_rank_tensor = torch.cuda.LongTensor(
            np.cumsum([0] + dim_sum_per_rank)[:-1]
        )

        grad_outputs_by_rank = grad_output.split(dim_sum_per_rank, dim=1)
        sharded_grad_output = torch.cat(
            [
                grad_output_by_rank.contiguous().view(-1)
                for grad_output_by_rank in grad_outputs_by_rank
            ],
            dim=0,
        )
        sharded_grad_output_impl = (
            torch.ops.fbgemm.recat_embedding_grad_output_mixed_D_batch(
                grad_output, dim_sum_per_rank_tensor, cumsum_dim_sum_per_rank_tensor
            )
        )
        torch.testing.assert_close(
            sharded_grad_output_impl.cpu(), sharded_grad_output.cpu()
        )


class MultiDimSplitTest(unittest.TestCase):
    def _test_multi_dim_split(
        self, grad: Tensor, splits: List[int], split_grad: List[Tensor]
    ) -> None:
        for idx, g in enumerate(torch.ops.fbgemm.multi_dim_split(grad, splits)):
            self.assertTrue(torch.equal(split_grad[idx], g.cpu()))

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(use_cpu=st.just(True) if gpu_unavailable[0] else st.booleans())
    @settings(deadline=None)
    def test_multi_dim_split_1(self, use_cpu: bool) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        grad = torch.arange(10).reshape(5, 2).to(device)
        splits = [3, 2]
        split_grad = [torch.arange(6).reshape(3, 2), torch.arange(6, 10).reshape(2, 2)]
        self._test_multi_dim_split(grad, splits, split_grad)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(use_cpu=st.just(True) if gpu_unavailable[0] else st.booleans())
    @settings(deadline=None)
    def test_multi_dim_split_2(self, use_cpu: bool) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        grad = torch.arange(10).reshape(5, 2).to(device)
        splits = [3, 1]
        split_grad = [
            torch.arange(0, 6, 2).reshape(3, 1),
            torch.arange(0, 6, 2).reshape(3, 1) + 1,
            torch.arange(6, 10, 2).reshape(2, 1),
            torch.arange(6, 10, 2).reshape(2, 1) + 1,
        ]
        self._test_multi_dim_split(grad, splits, split_grad)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(use_cpu=st.just(True) if gpu_unavailable[0] else st.booleans())
    @settings(deadline=None)
    def test_multi_dim_split_3(self, use_cpu: bool) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        grad = torch.arange(30).reshape(5, 2, 3).to(device)
        splits = [3, 1, 2]
        split_grad = [
            torch.tensor([[[0, 1]], [[6, 7]], [[12, 13]]]),
            torch.tensor([[[2]], [[8]], [[14]]]),
            torch.tensor([[[3, 4]], [[9, 10]], [[15, 16]]]),
            torch.tensor([[[5]], [[11]], [[17]]]),
            torch.tensor([[[18, 19]], [[24, 25]]]),
            torch.tensor([[[20]], [[26]]]),
            torch.tensor([[[21, 22]], [[27, 28]]]),
            torch.tensor([[[23]], [[29]]]),
        ]
        self._test_multi_dim_split(grad, splits, split_grad)


class MultiDimCatTest(unittest.TestCase):
    def _test_multi_dim_cat(
        self, grad: Tensor, num_splits: List[int], split_grad: List[Tensor]
    ) -> None:
        self.assertTrue(
            torch.equal(
                torch.ops.fbgemm.multi_dim_cat(split_grad, num_splits), grad.cpu()
            )
        )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(use_cpu=st.just(True) if gpu_unavailable[0] else st.booleans())
    @settings(deadline=None)
    def test_multi_dim_cat_1(self, use_cpu: bool) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        grad = torch.arange(10).reshape(5, 2).to(device)
        num_splits = [2, 1]
        split_grad = [torch.arange(6).reshape(3, 2), torch.arange(6, 10).reshape(2, 2)]
        self._test_multi_dim_cat(grad, num_splits, split_grad)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(use_cpu=st.just(True) if gpu_unavailable[0] else st.booleans())
    @settings(deadline=None)
    def test_multi_dim_cat_2(self, use_cpu: bool) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        grad = torch.arange(10).reshape(5, 2).to(device)
        num_splits = [2, 2]
        split_grad = [
            torch.arange(0, 6, 2).reshape(3, 1),
            torch.arange(0, 6, 2).reshape(3, 1) + 1,
            torch.arange(6, 10, 2).reshape(2, 1),
            torch.arange(6, 10, 2).reshape(2, 1) + 1,
        ]
        self._test_multi_dim_cat(grad, num_splits, split_grad)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(use_cpu=st.just(True) if gpu_unavailable[0] else st.booleans())
    @settings(deadline=None)
    def test_multi_dim_cat_3(self, use_cpu: bool) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        grad = torch.arange(30).reshape(5, 2, 3).to(device)
        num_splits = [2, 2, 2]
        split_grad = [
            torch.tensor([[[0, 1]], [[6, 7]], [[12, 13]]]),
            torch.tensor([[[2]], [[8]], [[14]]]),
            torch.tensor([[[3, 4]], [[9, 10]], [[15, 16]]]),
            torch.tensor([[[5]], [[11]], [[17]]]),
            torch.tensor([[[18, 19]], [[24, 25]]]),
            torch.tensor([[[20]], [[26]]]),
            torch.tensor([[[21, 22]], [[27, 28]]]),
            torch.tensor([[[23]], [[29]]]),
        ]
        self._test_multi_dim_cat(grad, num_splits, split_grad)


if __name__ == "__main__":
    unittest.main()
