# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


# pyre-strict
# pyre-ignore-all-errors[56]

import logging
import sys
import unittest
from typing import Tuple

import fbgemm_gpu.experimental.fb.gen_ai  # noqa: F401

import torch
import triton  # noqa: F401
from llm_inference.utils import profiler_or_nullcontext, record_function_or_nullcontext

from triton.testing import do_bench

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when no Hopper GPU is available.",
)
class IndexShufflingTests(unittest.TestCase):
    """Test IndexShuffling."""

    def test_top1_index_shuffling(self) -> None:
        def _test_top1_index_shuffling(
            num_tokens: int,
            num_experts: int,
            benchmark: bool = False,
            compile: bool = False,
        ) -> None:
            print(
                f"num_tokens={num_tokens}, num_experts={num_experts}, ",
                file=sys.stderr,
            )
            torch.manual_seed(0)

            scores = torch.randn(
                num_tokens, num_experts, device="cuda", dtype=torch.bfloat16
            )

            def fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                op = torch.ops.fbgemm.index_shuffling
                if compile:
                    op = torch.compile(op, backend="inductor", fullgraph=True)
                return op(scores)

            def ref_fn():
                selected_scores, selected_expert_indices = torch.topk(scores, 1, dim=1)
                expert_indices, token_indices = torch.sort(
                    selected_expert_indices, dim=0
                )
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
            token_indices_unshuffling = torch.sort(token_indices, dim=0)[1]
            ref_token_indices_unshuffling = torch.sort(ref_token_indices, dim=0)[1]
            self.assertTrue(
                expert_indices[token_indices_unshuffling].equal(
                    ref_expert_indices[ref_token_indices_unshuffling]
                )
            )

            # Performance check
            if benchmark:
                with profiler_or_nullcontext(
                    enabled=True,
                    rank=0,
                    seq_no=0,
                    with_stack=True,
                    record_shapes=True,
                    with_cpu=True,
                    upload_to_manifold=True,
                ):
                    with record_function_or_nullcontext(
                        f"{num_tokens=},{num_experts=}", benchmark
                    ):
                        with record_function_or_nullcontext("fbgemm", benchmark):
                            fbgemm_time = do_bench(fn) * 1e3
                        with record_function_or_nullcontext("torch", benchmark):
                            torch_time = do_bench(ref_fn) * 1e3
                print(
                    f"num_tokens={num_tokens:4}, num_experts={num_experts:4}, "
                    f"fbgemm_time={fbgemm_time:7.3f}us, torch_time={torch_time:7.3f}us",
                    file=sys.stderr,
                )

        # Correctness check on random shapes
        for num_tokens in [3, 123, 1234, 4567, 7891]:
            for num_experts in [16, 128]:
                _test_top1_index_shuffling(num_tokens, num_experts, benchmark=False)

        # Performance check on regular shapes
        for num_tokens in [1, 128, 2048, 4096, 8192]:
            for num_experts in [16, 128]:
                _test_top1_index_shuffling(num_tokens, num_experts, benchmark=True)


if __name__ == "__main__":
    unittest.main()
