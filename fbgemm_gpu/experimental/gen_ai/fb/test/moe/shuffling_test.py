# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


# pyre-strict
# pyre-ignore-all-errors[56]

import inspect
import itertools
import logging
import unittest
from typing import List, Tuple

import fbgemm_gpu.experimental.fb.gen_ai  # noqa: F401

import torch
import triton  # noqa: F401
from fbgemm_gpu.experimental.fb.gen_ai.moe.shuffling import combine_shuffling
from llm_inference.utils import profiler_or_nullcontext, record_function_or_nullcontext
from parameterized import param, parameterized

from triton.testing import do_bench, do_bench_cudagraph

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch._dynamo.config.cache_size_limit = 128


# pyre-ignore
def _name_test_func(fn, _, p) -> str:
    name = fn.__name__
    args = inspect.getfullargspec(fn).args
    for arg_name in args[1:]:
        name += f"_{arg_name}={p.kwargs[arg_name]}"
    return name


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when no Hopper GPU is available.",
)
class ShufflingTests(unittest.TestCase):
    """Test shuffling kernels."""

    @parameterized.expand(
        [
            param(
                num_tokens=num_tokens,
                num_experts=num_experts,
                rowmajor=rowmajor,
                compile=compile_,
                benchmark=False,
            )
            for num_tokens in [3, 123, 1234, 4567, 7891]
            for num_experts in [16, 128]
            for rowmajor in [True, False]
            for compile_ in [True, False]
        ]
        + [
            param(
                num_tokens=num_tokens,
                num_experts=num_experts,
                rowmajor=rowmajor,
                compile=False,
                benchmark=True,
            )
            for num_tokens in [1, 128, 2048, 4096, 8192]
            for num_experts in [16, 128]
            for rowmajor in [True, False]
        ],
        name_func=_name_test_func,
    )
    def test_top1_index_shuffling(
        self,
        num_tokens: int,
        num_experts: int,
        rowmajor: bool,
        benchmark: bool = False,
        compile: bool = False,
    ) -> None:
        torch.manual_seed(0)

        scores: torch.Tensor = torch.randn(
            num_tokens, num_experts, device="cuda", dtype=torch.bfloat16
        )
        scores = scores.contiguous()
        if not rowmajor:
            scores = scores.transpose(0, 1).contiguous().transpose(0, 1)

        def fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            op = torch.ops.fbgemm.index_shuffling
            if compile:
                op = torch.compile(op, backend="inductor", fullgraph=True)
            return op(scores)

        def ref_fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            selected_scores, selected_expert_indices = torch.topk(scores, 1, dim=1)
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
                enabled=False,
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
                        fbgemm_time = do_bench_cudagraph(fn) * 1e3
                    with record_function_or_nullcontext("torch", benchmark):
                        torch_time = do_bench_cudagraph(ref_fn) * 1e3
            logger.info(
                f"num_tokens={num_tokens:4}, num_experts={num_experts:4}, rowmajor={int(rowmajor)}, "
                f"fbgemm_time={fbgemm_time:7.3f}us, torch_time={torch_time:7.3f}us"
            )

    @parameterized.expand(
        [
            param(
                num_tokens=num_tokens,
                num_experts=num_experts,
                ep_size=ep_size,
                dim=dim,
                balanced=balanced,
                benchmark=benchmark,
            )
            for num_tokens in [123, 456, 789, 1234, 4567, 7891]
            for num_experts in [16, 128]
            for ep_size in [4, 8]
            for dim in [5120]
            for balanced in [False]
            for benchmark in [False]
        ]
        + [
            param(
                num_tokens=num_tokens,
                num_experts=num_experts,
                ep_size=ep_size,
                dim=dim,
                balanced=balanced,
                benchmark=benchmark,
            )
            for num_tokens in [128, 2048, 4096, 8192]
            for num_experts in [16, 128]
            for ep_size in [4, 8]
            for dim in [5120]
            for balanced in [True, False]
            for benchmark in [True, False]
        ],
        name_func=_name_test_func,
    )
    def test_combine_shuffling(
        self,
        num_tokens: int,
        num_experts: int,
        ep_size: int,
        dim: int,
        balanced: bool = False,
        benchmark: bool = False,
    ) -> None:
        torch.manual_seed(0)

        tokens: torch.Tensor = torch.randn(
            num_tokens, dim, device="cuda", dtype=torch.bfloat16
        )
        assert num_experts % ep_size == 0
        num_local_experts: int = num_experts // ep_size
        if balanced:
            assert num_tokens % num_experts == 0
            num_tokens_per_expert = num_tokens // num_experts
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

        def prepare_ref_fn() -> Tuple[torch.Tensor, ...]:
            offset = 0
            reshuffled_chunks = [[] for _ in range(num_local_experts)]
            # token_counts: [EP, E]
            for counts_per_rank in token_counts_list:
                for expert, chunk_size in enumerate(counts_per_rank):
                    reshuffled_chunks[expert].append(
                        tokens[offset : offset + chunk_size]
                    )
                    offset += chunk_size
            return tuple(itertools.chain(*reshuffled_chunks))

        reshuffled_chunks: Tuple[torch.Tensor, ...] = prepare_ref_fn()

        def ref_fn() -> Tuple[torch.Tensor, torch.Tensor]:
            tokens = torch.cat(reshuffled_chunks)
            counts = token_counts.sum(dim=0)
            return tokens, counts

        ref_output_tokens, ref_output_token_counts = ref_fn()

        def fn() -> Tuple[torch.Tensor, torch.Tensor]:
            return combine_shuffling(tokens, token_counts, balanced=balanced)

        output_tokens, output_token_counts = fn()

        self.assertEqual(tuple(output_tokens.shape), (num_tokens, dim))
        self.assertEqual(tuple(output_token_counts.shape), (num_local_experts,))
        self.assertEqual(output_token_counts.sum().item(), num_tokens)

        self.assertTrue(output_token_counts.equal(ref_output_token_counts.cuda()))
        self.assertTrue(output_tokens.equal(ref_output_tokens))

        if benchmark:
            mem_bytes = num_tokens * dim * 2 * 2
            fbgemm_time = do_bench(fn) * 1e3
            fbgemm_bw = mem_bytes * 1e-9 / (fbgemm_time * 1e-6)
            # We don't benchmark counting on CPU
            torch_time = do_bench(ref_fn) * 1e3
            torch_bw = mem_bytes * 1e-9 / (torch_time * 1e-6)

            logger.info(
                f"num_tokens={num_tokens:4}, ep_size={ep_size:4}, num_local_experts={num_local_experts:4}, balanced={int(balanced)}, "
                f"fbgemm_time={fbgemm_time:7.3f}us, fbgemm_bw={fbgemm_bw:8.3f}GBytes/s,  "
                f"torch_time={torch_time:7.3f}us, torch_bw={torch_bw:8.3f}GBytes/s, speedup={torch_time / fbgemm_time:7.3f}x",
            )


if __name__ == "__main__":

    unittest.main()
