# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16,21,53,56]

import logging
import unittest
from typing import Tuple

import torch
import triton  # noqa: F401

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        triton_quantize_fp8_row,
    )
    from fbgemm_gpu.experimental.gen_ai.moe import (
        gather_scale_dense_tokens,
        gather_scale_quant_dense_tokens,
        scatter_add_dense_tokens,
        scatter_add_padded_tokens,
    )

from hypothesis import given, settings, strategies as st, Verbosity

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source

    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu.docs.version import __version__  # noqa: F401
except Exception:
    open_source: bool = False


logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

_MAX_SAMPLES: int = 100


@unittest.skipIf(open_source, "Tests currently fail in open source")
@unittest.skipIf(
    not torch.cuda.is_available()
    or (torch.version.hip is None and torch.version.cuda < "12.4"),
    "Skip when no GPU is available or CUDA version is older than `12.4`.",
)
class GatherScatterTests(unittest.TestCase):
    """Test gather/scatter kernels."""

    @given(
        E=st.sampled_from([2, 4, 8]),
        T=st.sampled_from([1, 128, 2048, 4096, 16384]),
        D=st.sampled_from([5120, 7168]),
        rowmajor=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_gather_scale_dense_tokens(
        self,
        E: int,
        T: int,
        D: int,
        rowmajor: bool,
        compiled: bool,
    ) -> None:
        if T == 16384 and (E > 2 or D > 5120):
            logger.info(f"Skipping test for E={E}, T={T} because it will lead to OOM")
            return
        x: torch.Tensor = torch.randn((T, D), dtype=torch.bfloat16, device="cuda").abs()
        expert_indices: torch.Tensor = torch.randint(0, E, (T,), device="cuda")
        token_indices: torch.Tensor = torch.randperm(T, device="cuda").to(torch.int32)
        scores: torch.Tensor = torch.rand((E, T), dtype=torch.bfloat16, device="cuda")

        def torch_fn() -> torch.Tensor:
            shuffled_x = torch.index_select(x, dim=0, index=token_indices)
            shuffled_scores = torch.index_select(scores, dim=1, index=token_indices)
            shuffled_selected_scores = torch.gather(
                shuffled_scores, dim=0, index=expert_indices.view(1, T)
            )
            ref_output = shuffled_x * shuffled_selected_scores.view(-1, 1)
            return ref_output

        torch_output = torch_fn()

        def triton_fn() -> torch.Tensor:
            scores_ = scores.contiguous().transpose(0, 1)
            if rowmajor:
                scores_ = scores_.contiguous()
            op = gather_scale_dense_tokens
            if compiled:
                op = torch.compile(op)
            test_output = op(x, token_indices, expert_indices, scores_)
            return test_output

        test_output = triton_fn()

        torch.testing.assert_close(torch_output, test_output)

    @given(
        E=st.sampled_from([2, 4, 8]),
        T=st.sampled_from([1, 128, 2048, 4096, 16384]),
        D=st.sampled_from([5120, 7168]),
        rowmajor=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_gather_scale_quant_dense_tokens(
        self,
        E: int,
        T: int,
        D: int,
        rowmajor: bool,
        compiled: bool,
    ) -> None:
        if T == 16384 and (E > 2 or D > 5120):
            logger.info(f"Skipping test for E={E}, T={T} because it will lead to OOM")
            return
        x: torch.Tensor = torch.randn((T, D), dtype=torch.bfloat16, device="cuda")
        expert_indices: torch.Tensor = torch.randint(0, E, (T,), device="cuda")
        token_indices: torch.Tensor = torch.randperm(T, device="cuda").to(torch.int32)
        scores: torch.Tensor = torch.randn((E, T), dtype=torch.bfloat16, device="cuda")

        scale_ub = torch.tensor([1200], dtype=torch.float, device="cuda")

        def torch_fn() -> Tuple[torch.Tensor, torch.Tensor]:
            shuffled_x = torch.index_select(x, dim=0, index=token_indices)
            shuffled_scores = torch.index_select(scores, dim=1, index=token_indices)
            shuffled_selected_scores = torch.gather(
                shuffled_scores, dim=0, index=expert_indices.view(1, T)
            )
            ref_output = shuffled_x.to(torch.float32) * shuffled_selected_scores.view(
                -1, 1
            ).to(torch.float32)
            ref_output_q, ref_output_scales = triton_quantize_fp8_row(
                ref_output, scale_ub
            )
            return ref_output_q, ref_output_scales

        torch_output_q, torch_output_scales = torch_fn()
        torch_output = torch_output_q.to(torch.float32) * torch_output_scales.view(
            -1, 1
        )

        def triton_fn() -> Tuple[torch.Tensor, torch.Tensor]:
            scores_ = scores.contiguous().transpose(0, 1)
            if rowmajor:
                scores_ = scores_.contiguous()
            op = gather_scale_quant_dense_tokens
            if compiled:
                op = torch.compile(op)
            test_output_q, test_output_scales = op(
                x, token_indices, expert_indices, scores_, scale_ub
            )
            return test_output_q, test_output_scales

        test_output_q, test_output_scales = triton_fn()
        test_output = test_output_q.to(torch.float32) * test_output_scales.view(-1, 1)

        torch.testing.assert_close(torch_output, test_output, atol=1e-3, rtol=1.6e-2)

    @given(
        num_tokens=st.sampled_from([1, 128, 2048, 4096, 16384]),
        dim=st.sampled_from([5120]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_scatter_add_dense_tokens(
        self,
        num_tokens: int,
        dim: int,
        compiled: bool,
    ) -> None:
        torch.manual_seed(0)

        in_tokens: torch.Tensor = torch.randn(
            num_tokens, dim, device="cuda", dtype=torch.bfloat16
        )
        out_tokens: torch.Tensor = torch.randn(
            num_tokens, dim, device="cuda", dtype=torch.bfloat16
        )

        token_indices: torch.Tensor = torch.randperm(num_tokens, device="cuda").to(
            torch.int32
        )

        test_out_tokens: torch.Tensor = out_tokens.clone()
        ref_out_tokens: torch.Tensor = out_tokens.clone()

        def fn() -> None:
            op = scatter_add_dense_tokens
            if compiled:
                op = torch.compile(op)
            op(
                test_out_tokens,
                in_tokens,
                token_indices,
            )

        fn()

        token_indices: torch.Tensor = token_indices.to(torch.int64)

        def ref_fn() -> None:
            ref_out_tokens.scatter_add_(
                dim=0,
                index=token_indices.view(-1, 1).expand(-1, dim),
                src=in_tokens.view(-1, dim),
            )

        ref_fn()

        torch.testing.assert_close(
            test_out_tokens, ref_out_tokens, atol=1e-3, rtol=1.6e-2
        )

    @given(
        num_tokens=st.sampled_from([64, 128, 256]),
        num_experts=st.sampled_from([16, 128]),
        ep_size=st.sampled_from([2, 4, 8, 16]),
        dim=st.sampled_from([5120]),
        balanced=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_scatter_add_padded_tokens(
        self,
        num_tokens: int,
        num_experts: int,
        ep_size: int,
        dim: int,
        balanced: bool,
        compiled: bool,
    ) -> None:
        torch.manual_seed(0)

        in_tokens: torch.Tensor = torch.randn(
            ep_size, num_tokens, dim, device="cuda", dtype=torch.bfloat16
        )
        out_tokens: torch.Tensor = torch.randn(
            num_tokens, dim, device="cuda", dtype=torch.bfloat16
        )

        if balanced:
            if num_tokens < num_experts:
                logger.info("Skipping test as num_tokens must be >= num_experts")
                return
            num_tokens_per_expert = num_tokens // num_experts
            token_counts: torch.Tensor = torch.tensor(
                [num_tokens_per_expert] * num_experts, device="cuda"
            ).to(torch.int32)
        else:
            token_choices = torch.randint(0, num_experts, (num_tokens,), device="cuda")
            token_counts = torch.bincount(token_choices, minlength=num_experts)

        token_cumsums = torch.cumsum(token_counts, dim=0)

        token_indices: torch.Tensor = torch.randperm(num_tokens, device="cuda").to(
            torch.int32
        )

        test_out_tokens: torch.Tensor = out_tokens.clone()
        ref_out_tokens: torch.Tensor = out_tokens.clone()

        def fn() -> None:
            op = scatter_add_padded_tokens
            if compiled:
                op = torch.compile(op)
            op(
                in_tokens,
                token_counts,
                token_indices,
                test_out_tokens,
            )

        fn()

        token_indices: torch.Tensor = token_indices.to(torch.int64)
        token_cumsums_list: list[int] = [0] + token_cumsums.tolist()
        num_experts_per_rank: int = num_experts // ep_size

        def ref_fn() -> None:
            for rank in range(ep_size):
                start_index = token_cumsums_list[num_experts_per_rank * rank]
                end_index = token_cumsums_list[num_experts_per_rank * (rank + 1)]
                if start_index == end_index:
                    continue
                ref_out_tokens.scatter_add_(
                    dim=0,
                    index=token_indices[start_index:end_index]
                    .view(-1, 1)
                    .expand(-1, dim),
                    src=in_tokens[rank, start_index:end_index, :].view(-1, dim),
                )

        ref_fn()

        torch.testing.assert_close(
            test_out_tokens, ref_out_tokens, atol=1e-3, rtol=1.6e-2
        )


if __name__ == "__main__":

    unittest.main()
