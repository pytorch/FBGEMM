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
        num_experts=st.sampled_from([16, 128]),
        rowmajor=st.sampled_from([True, False]),
        padded=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_top1_index_shuffling(
        self,
        num_tokens: int,
        num_experts: int,
        rowmajor: bool,
        padded: bool,
        compiled: bool,
    ) -> None:
        torch.manual_seed(0)

        num_valid_tokens_tensor: Optional[torch.Tensor] = None
        num_valid_tokens_scalar: int = num_tokens
        num_total_tokens = num_tokens
        if padded:
            num_valid_tokens_scalar = num_tokens
            num_valid_tokens_tensor: Optional[torch.Tensor] = torch.tensor(
                [num_tokens], device="cuda"
            )
            num_total_tokens = num_tokens * 2

        scores: torch.Tensor = torch.randn(
            num_total_tokens, num_experts, device="cuda", dtype=torch.bfloat16
        )
        scores = scores.contiguous()

        if not rowmajor:
            scores = scores.transpose(0, 1).contiguous().transpose(0, 1)

        def fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            op = index_shuffling
            if compiled:
                op = torch.compile(op, backend="inductor", fullgraph=True)
            return op(scores, num_valid_tokens_tensor)

        def ref_fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            valid_scores = scores[:num_valid_tokens_scalar].contiguous()
            selected_scores, selected_expert_indices = torch.topk(
                valid_scores, 1, dim=1
            )
            expert_indices, token_indices = torch.sort(selected_expert_indices, dim=0)
            router_counts = (
                expert_indices[:, None]
                == torch.arange(num_experts, device=expert_indices.device)[None, :]
            ).sum(dim=0)
            return (
                router_counts.flatten(),
                expert_indices.flatten(),
                token_indices.flatten(),
            )

        router_counts, expert_indices, token_indices = fn()
        ref_router_counts, ref_expert_indices, ref_token_indices = ref_fn()

        # Correctness check
        self.assertTrue(router_counts[:-1].equal(ref_router_counts))
        self.assertEqual(router_counts[:-1].sum().item(), num_valid_tokens_scalar)
        self.assertEqual(expert_indices.shape, (num_total_tokens,))
        self.assertEqual(token_indices.shape, (num_total_tokens,))

        if padded:
            assert num_valid_tokens_tensor is not None
            assert num_valid_tokens_tensor.item() == num_valid_tokens_scalar
            assert num_valid_tokens_tensor.item() < num_total_tokens
        else:
            assert num_valid_tokens_tensor is None
            assert num_valid_tokens_scalar == num_total_tokens

        # No invalid indices
        self.assertTrue(
            (expert_indices >= 0).logical_and(expert_indices < num_experts).all()
        )
        self.assertTrue(
            (token_indices >= 0).logical_and(token_indices < num_total_tokens).all()
        )

        def _assert_indices_equal(
            indices1: torch.Tensor, indices2: torch.Tensor
        ) -> None:
            if indices1.numel() == 0 and indices2.numel() == 0:
                return
            indices1 = torch.sort(indices1)[0]
            indices2 = torch.sort(indices2)[0]
            self.assertTrue(
                torch.equal(
                    indices1,
                    indices2,
                ),
                f"indices1={indices1}, indices2={indices2}",
            )

        start_index = 0
        for i in range(num_experts):
            end_index = start_index + router_counts[i]
            _assert_indices_equal(
                expert_indices[start_index:end_index],
                ref_expert_indices[start_index:end_index],
            )
            _assert_indices_equal(
                token_indices[start_index:end_index],
                ref_token_indices[start_index:end_index],
            )
            start_index = end_index
        token_indices_unshuffling = torch.sort(
            token_indices[:num_valid_tokens_tensor], dim=0
        )[1]
        ref_token_indices_unshuffling = torch.sort(ref_token_indices, dim=0)[1]
        self.assertTrue(
            expert_indices[token_indices_unshuffling].equal(
                ref_expert_indices[ref_token_indices_unshuffling]
            )
        )

    @given(
        num_tokens=st.sampled_from(
            [1, 3, 123, 128, 1234, 2048, 4567, 4096, 8192, 16384]
        ),
        num_experts=st.sampled_from([16, 128]),
        ep_size=st.sampled_from([4, 8]),
        dim=st.sampled_from([5120]),
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
        sparse: bool,
        balanced: bool,
        target_fn: str,
    ) -> None:
        torch.manual_seed(0)

        is_combine_shuffling: bool = target_fn == "combine_shuffling"
        assert num_experts % ep_size == 0

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
            num_tokens, dim, device="cuda", dtype=torch.bfloat16
        )

        if balanced:
            assert num_tokens % (ep_size * num_local_experts) == 0
            num_tokens_per_expert = num_tokens // (ep_size * num_local_experts)
            token_counts: torch.Tensor = (
                torch.ones(
                    [ep_size, num_local_experts], dtype=torch.int32, device="cuda"
                )
                * num_tokens_per_expert
            )
        else:
            token_cumsums, _ = torch.sort(
                torch.randint(
                    low=0,
                    high=num_tokens,
                    size=(num_local_experts * ep_size + 1,),
                    device="cuda",
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
        self.assertEqual(tuple(output_tokens.shape), (num_tokens, dim))
        if not is_combine_shuffling and sparse:
            reverse_input_tokens = torch.concat(
                slice_tokens(output_tokens, combine=True)
            )
            self.assertTrue(tuple(reverse_input_tokens.shape), (num_valid_tokens, dim))
            self.assertTrue(reverse_input_tokens.equal(tokens[:num_valid_tokens]))
        else:
            self.assertTrue(output_tokens[:num_valid_tokens].equal(ref_output_tokens))


if __name__ == "__main__":

    unittest.main()
