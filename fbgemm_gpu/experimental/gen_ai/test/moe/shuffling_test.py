# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16,21,56]

import functools
import itertools
import logging
import unittest
from typing import List, Optional, Tuple

import torch
import triton  # noqa: F401

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gen_ai.moe import (
        combine_shuffling,
        index_shuffling,
        split_shuffling,
    )

from hypothesis import given, settings, strategies as st, Verbosity

try:
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source

    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu.docs.version import __version__  # noqa: F401
except Exception:
    open_source: bool = False

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch._dynamo.config.cache_size_limit = 128

_MAX_SAMPLES: int = 100


@unittest.skipIf(open_source, "Tests currently fail in open source")
@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when no GPU is available.",
)
class ShufflingTests(unittest.TestCase):
    """Test shuffling kernels."""

    @given(
        num_tokens=st.sampled_from(
            [1, 3, 123, 128, 1234, 2048, 4567, 4096, 8192, 16384]
        ),
        num_experts=st.sampled_from([16, 32, 128, 320]),
        num_local_experts=st.sampled_from([None, 8]),
        top_k=st.sampled_from([1, 2, 4] if torch.version.cuda else [1]),
        padded=st.sampled_from([True, False]),
        rowmajor=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
        routing_score_dtype=st.sampled_from([torch.float, torch.bfloat16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_topk_index_shuffling(
        self,
        num_tokens: int,
        num_experts: int,
        num_local_experts: Optional[int],
        top_k: int,
        padded: bool,
        rowmajor: bool,
        compiled: bool,
        routing_score_dtype: torch.dtype,
    ) -> None:
        if (
            routing_score_dtype == torch.float
            and num_experts > 16
            and torch.version.hip
        ):
            self.skipTest(
                f"Skipping test for {routing_score_dtype=}, {num_experts=} due to torch.AcceleratorError: HIP error: out of memory"
            )

        torch.manual_seed(0)

        expert_index_start: int = 0
        expert_index_end: int = num_experts
        if num_local_experts is not None:
            expert_index_start = 1
            expert_index_end = expert_index_start + num_local_experts

        num_total_tokens: int = num_tokens
        num_valid_tokens: int = num_tokens
        valid_token_counts: Optional[torch.Tensor] = None
        if padded:
            num_total_tokens = num_tokens * 2
            num_valid_tokens = num_tokens
            valid_token_counts = torch.tensor([num_tokens], device="cuda")

        routing_scores: torch.Tensor = _get_scores_without_ties(
            num_total_tokens, num_experts, routing_score_dtype
        )
        if not rowmajor:
            routing_scores = routing_scores.transpose(0, 1).contiguous().transpose(0, 1)

        def fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            op = index_shuffling
            if compiled:
                op = torch.compile(op, backend="inductor", fullgraph=True)
            if num_local_experts is None and valid_token_counts is None:
                return op(routing_scores, top_k=top_k)
            else:
                return op(
                    routing_scores,
                    expert_index_start,
                    expert_index_end,
                    valid_token_counts,
                    top_k,
                )

        def ref_fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            valid_routing_scores = routing_scores[:num_valid_tokens].contiguous()
            selected_expert_indices = torch.topk(valid_routing_scores, top_k, dim=1)[1]
            expert_indices, flattened_position_indices = torch.sort(
                selected_expert_indices.flatten(), dim=0
            )
            token_indices = flattened_position_indices // top_k
            expert_ids = torch.arange(num_experts, device=expert_indices.device)
            token_counts_per_expert = (
                expert_indices[:, None] == expert_ids[None, :]
            ).sum(dim=0)
            return (
                token_counts_per_expert.flatten(),
                expert_indices.flatten(),
                token_indices.flatten(),
            )

        token_counts_per_expert, expert_indices, token_indices = fn()
        ref_token_counts_per_expert, ref_expert_indices, ref_token_indices = ref_fn()

        # Correctness check
        # 1. Checks `token_counts_per_expert` return is correct.
        self.assertEqual(token_counts_per_expert.shape, (num_experts + 2,))
        self.assertTrue(
            token_counts_per_expert[expert_index_start:expert_index_end].equal(
                ref_token_counts_per_expert[expert_index_start:expert_index_end]
            )
        )
        self.assertEqual(token_counts_per_expert[-2].item(), num_total_tokens)
        ref_num_sorted_tokens = torch.sum(
            ref_token_counts_per_expert[expert_index_start:expert_index_end]
        )
        self.assertEqual(token_counts_per_expert[-1].item(), ref_num_sorted_tokens)

        # 2. Checks `expert_indices` and `token_indices` returns are correct.
        self.assertEqual(expert_indices.shape, (num_total_tokens * top_k,))
        self.assertEqual(token_indices.shape, (num_total_tokens * top_k,))

        # Test behavior assertions
        if padded:
            assert valid_token_counts is not None
            assert valid_token_counts.item() < num_total_tokens
            assert valid_token_counts.item() == num_valid_tokens
        else:
            assert valid_token_counts is None
            assert num_valid_tokens == num_total_tokens

        # No invalid `expert_indices` or `token_indices` values.
        self.assertTrue(
            (expert_indices[:ref_num_sorted_tokens] >= 0)
            .logical_and(expert_indices[:ref_num_sorted_tokens] < num_experts)
            .all()
        )
        self.assertTrue(
            (token_indices[:ref_num_sorted_tokens] >= 0)
            .logical_and(token_indices[:ref_num_sorted_tokens] < num_total_tokens)
            .all()
        )

        def _assert_indices_equal(
            indices1: torch.Tensor, indices2: torch.Tensor
        ) -> None:
            if indices1.numel() == 0 and indices2.numel() == 0:
                return
            indices1 = torch.sort(indices1)[0]
            indices2 = torch.sort(indices2)[0]
            self.assertTrue(
                indices1.equal(indices2),
                f"indices1={indices1}, indices2={indices2}",
            )

        ref_start_index, start_index = 0, 0
        for i in range(num_experts):
            ref_end_index = ref_start_index + ref_token_counts_per_expert[i]
            if i >= expert_index_start and i < expert_index_end:
                end_index = start_index + token_counts_per_expert[i]
                self.assertTrue(
                    torch.equal(
                        token_counts_per_expert[i],
                        ref_token_counts_per_expert[i],
                    )
                )
                _assert_indices_equal(
                    expert_indices[start_index:end_index] + expert_index_start,
                    ref_expert_indices[ref_start_index:ref_end_index],
                )
                _assert_indices_equal(
                    token_indices[start_index:end_index],
                    ref_token_indices[ref_start_index:ref_end_index],
                )
                start_index = end_index
            ref_start_index = ref_end_index

    @given(
        num_tokens=st.sampled_from(
            [1, 3, 123, 128, 1234, 2048, 4567, 4096, 8192, 16384]
        ),
        num_experts=st.sampled_from([16, 80, 128]),
        ep_size=st.sampled_from([4, 5, 8, 11]),
        dim=st.sampled_from([5120]),
        top_k=st.sampled_from([1, 4]),
        sparse=st.sampled_from([True, False]),
        balanced=st.sampled_from([False]),
        target_fn=st.sampled_from(["combine_shuffling", "split_shuffling"]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_combine_or_split_shuffling(
        self,
        num_tokens: int,
        num_experts: int,
        ep_size: int,
        dim: int,
        top_k: int,
        sparse: bool,
        balanced: bool,
        target_fn: str,
    ) -> None:
        torch.manual_seed(0)
        device = device = torch.accelerator.current_accelerator()

        is_combine_shuffling: bool = target_fn == "combine_shuffling"

        if sparse:
            num_tokens *= ep_size
            num_local_experts: int = num_experts
            expert_start: int = 1
            expert_end: int = 1 + (num_experts // ep_size)
        else:
            num_local_experts: int = num_experts // ep_size
            expert_start: int = 0
            expert_end: int = num_local_experts

        expert_group_size: int = expert_end - expert_start

        tokens: torch.Tensor = torch.randn(
            num_tokens * ep_size * top_k, dim, device=device, dtype=torch.bfloat16
        )

        if balanced:
            assert num_tokens % (ep_size * num_local_experts) == 0
            num_tokens_per_expert = num_tokens // (ep_size * num_local_experts)
            token_counts: torch.Tensor = (
                torch.ones(
                    [ep_size, num_local_experts], dtype=torch.int32, device=device
                )
                * num_tokens_per_expert
            )
        else:
            token_cumsums, _ = torch.sort(
                torch.randint(
                    low=0,
                    high=num_tokens,
                    size=(num_local_experts * ep_size + 1,),
                    device=device,
                    dtype=torch.int32,
                )
            )
            token_cumsums[0] = 0
            token_cumsums[-1] = num_tokens
            token_counts: torch.Tensor = token_cumsums[1:] - token_cumsums[:-1]
            token_counts = token_counts.reshape(ep_size, num_local_experts)

        assert token_counts.sum().item() == num_tokens

        token_counts_list: List[List[int]] = token_counts.tolist()
        token_counts_t_list: List[List[int]] = token_counts.T.tolist()

        def slice_tokens(
            tokens: torch.Tensor,
            combine: bool,
        ) -> Tuple[torch.Tensor, ...]:
            offset = 0
            if combine:
                reshuffled_chunks = [[] for _ in range(num_local_experts)]
                # token_counts: [EP, E]
                for counts_per_rank in token_counts_list:
                    for expert, chunk_size in enumerate(counts_per_rank):
                        if expert >= expert_start and expert < expert_end:
                            reshuffled_chunks[expert].append(
                                tokens[offset : offset + chunk_size]
                            )
                        offset += chunk_size
            else:
                reshuffled_chunks = [[] for _ in range(ep_size)]
                # token_counts: [EP, E]
                for expert, counts_per_expert in enumerate(token_counts_t_list):
                    for rank, chunk_size in enumerate(counts_per_expert):
                        if expert >= expert_start and expert < expert_end:
                            reshuffled_chunks[rank].append(
                                tokens[offset : offset + chunk_size]
                            )
                        offset += chunk_size
            return tuple(itertools.chain(*reshuffled_chunks))

        prepare_ref_fn = functools.partial(
            slice_tokens, tokens=tokens, combine=is_combine_shuffling
        )
        reshuffled_chunks: Tuple[torch.Tensor, ...] = prepare_ref_fn()

        def ref_fn() -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            cat_tokens = torch.cat(reshuffled_chunks)
            if is_combine_shuffling:
                counts = token_counts[:, expert_start:expert_end].sum(dim=0)
            else:
                counts = None
            return cat_tokens, counts

        ref_output_tokens, ref_output_token_counts = ref_fn()

        def fn() -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            if is_combine_shuffling:
                return combine_shuffling(
                    tokens,
                    token_counts,
                    expert_start=expert_start,
                    expert_end=expert_end,
                    is_balanced=balanced,
                )
            else:
                return (
                    split_shuffling(
                        tokens,
                        token_counts,
                        expert_start=expert_start,
                        expert_end=expert_end,
                        is_balanced=balanced,
                    ),
                    None,
                )

        output_tokens, output_token_counts = fn()

        if is_combine_shuffling:
            assert output_token_counts is not None
            assert ref_output_token_counts is not None
            self.assertEqual(tuple(output_token_counts.shape), (expert_group_size + 1,))
            self.assertTrue(output_token_counts[:-1].equal(ref_output_token_counts))
            self.assertEqual(
                output_token_counts[:-1].sum(), output_token_counts[-1].item()
            )

        num_valid_tokens = ref_output_tokens.shape[0]
        self.assertEqual(
            tuple(output_tokens.shape), (num_tokens * ep_size * top_k, dim)
        )
        if not is_combine_shuffling and sparse:
            reverse_input_tokens = torch.concat(
                slice_tokens(output_tokens, combine=True)
            )
            self.assertTrue(tuple(reverse_input_tokens.shape), (num_valid_tokens, dim))
            self.assertTrue(reverse_input_tokens.equal(tokens[:num_valid_tokens]))
        else:
            self.assertTrue(output_tokens[:num_valid_tokens].equal(ref_output_tokens))


def _get_scores_without_ties(
    num_total_tokens: int, num_experts: int, routing_score_dtype: torch.dtype
) -> torch.Tensor:
    """
    Generate routing scores without ties in each row - ties are harmless in a real run, but difficult to test.
    """
    diffs = (
        torch.randn(
            num_total_tokens,
            num_experts,
            device=torch.accelerator.current_accelerator(),
            dtype=routing_score_dtype,
        ).abs()
        + 0.1
    )
    rand_shifts = torch.randn(
        num_total_tokens,
        1,
        device=torch.accelerator.current_accelerator(),
        dtype=routing_score_dtype,
    )
    # Cumulative sums are all positive, so add random shifts to make some score values negative for testing.
    routing_scores_sorted = diffs.cumsum(dim=1) - rand_shifts
    random_indices = torch.argsort(
        torch.rand(num_total_tokens, num_experts, device=diffs.device), dim=1
    )
    return routing_scores_sorted.gather(1, random_indices).contiguous()


if __name__ == "__main__":

    unittest.main()
