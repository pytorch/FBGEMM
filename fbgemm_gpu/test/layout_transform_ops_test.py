#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_memory_lt_gb, gpu_unavailable

except Exception:
    from fbgemm_gpu.test.test_utils import gpu_memory_lt_gb, gpu_unavailable

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


MAX_EXAMPLES = 20


class LayoutTransformOpsTest(unittest.TestCase):
    # pyrefly: ignore [bad-argument-type]
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

    # pyrefly: ignore [bad-argument-type]
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

    # pyrefly: ignore [bad-argument-type]
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

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]: Pyre cannot infer the type of `gpu_memory_lt_gb`
    # through the open-source / non-open-source import branch above.
    @unittest.skipIf(*gpu_memory_lt_gb(8))
    def test_recat_embedding_grad_output_mixed_D_batch_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in recat_copy_async_kernel
        (include/fbgemm_gpu/layout_transform_ops.cuh).

        Block: dim3(kWarpSize=32, kMaxThreads/kWarpSize=32) = 1024 threads.
        Grid: div_round_up(B_local * dim_num, 32). Total threads ~=
        B_local * dim_num * kWarpSize. For B_local * dim_num > 2**27,
        total threads exceed the HIP 2**32 limit, tripping
        KernelLauncher::checkThreadCountNotExceeded on ROCm.

        Pre-fix: AMD launch fails. Post-fix: passes (kernel grid-strides).
        """
        B_local = 8
        dim_num = (1 << 24) + 1  # B_local * dim_num = 2**27 + 8 > 2**27
        # All-zero dim_sum_per_rank means each rank contributes 0 elements,
        # so per-tile copy work inside the kernel is empty. grad_output
        # therefore has width 0 and the output tensor is also empty.
        dim_sum_per_rank_tensor = torch.zeros(
            dim_num, dtype=torch.int64, device=torch.accelerator.current_accelerator()
        )
        cumsum_dim_sum_per_rank_tensor = torch.zeros(
            dim_num, dtype=torch.int64, device=torch.accelerator.current_accelerator()
        )
        grad_output = torch.zeros(
            (B_local, 0),
            dtype=torch.float32,
            device=torch.accelerator.current_accelerator(),
        )
        sharded_grad_output = (
            torch.ops.fbgemm.recat_embedding_grad_output_mixed_D_batch(
                grad_output,
                dim_sum_per_rank_tensor,
                cumsum_dim_sum_per_rank_tensor,
            )
        )
        self.assertEqual(sharded_grad_output.numel(), grad_output.numel())

        # Tier-B Python-reference correctness check at small scale.
        # Sentinel non-zero values force the kernel to do non-trivial copies.
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        small_B = 4
        small_dim_sum_per_rank = [3, 5, 2]  # W = 3 ranks
        small_total_D = sum(small_dim_sum_per_rank)
        small_grad_output = torch.arange(
            small_B * small_total_D, dtype=torch.float32, device=device
        ).reshape(small_B, small_total_D)
        small_dim_sum_tensor = torch.tensor(
            small_dim_sum_per_rank, dtype=torch.int64, device=device
        )
        small_cumsum_tensor = torch.tensor(
            [0] + list(np.cumsum(small_dim_sum_per_rank)[:-1]),
            dtype=torch.int64,
            device=device,
        )

        # Python reference: split grad_output column-wise per rank, then
        # concatenate the per-rank flattened views.
        ref_outputs_by_rank = small_grad_output.split(small_dim_sum_per_rank, dim=1)
        ref_sharded = torch.cat(
            [g.contiguous().view(-1) for g in ref_outputs_by_rank], dim=0
        )

        small_sharded = torch.ops.fbgemm.recat_embedding_grad_output_mixed_D_batch(
            small_grad_output,
            small_dim_sum_tensor,
            small_cumsum_tensor,
        )
        torch.testing.assert_close(small_sharded, ref_sharded)


if __name__ == "__main__":
    unittest.main()
