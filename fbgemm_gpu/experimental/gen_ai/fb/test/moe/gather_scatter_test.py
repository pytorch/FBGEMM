# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
# pyre-ignore-all-errors[56]

import logging
import unittest

import torch
import triton  # noqa: F401
from fbgemm_gpu.experimental.fb.gen_ai.moe import scatter_add_padded_tokens

from parameterized import param, parameterized
from triton.testing import do_bench

from .utils import do_bench_cudagraph_and_clear_cache, name_test_func

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.version.cuda < "12.4",
    "Skip when no GPU is available or CUDA version is older than `12.4`.",
)
class GatherScatterTests(unittest.TestCase):
    """Test shuffling kernels."""

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
            for num_tokens in [64, 128, 256]
            for num_experts in [16, 128]
            for ep_size in [2, 4, 8, 16]
            for dim in [5120]
            for balanced in [True, False]
            for benchmark in [True]
        ],
        name_func=name_test_func,
    )
    def test_scatter_add_padded_tokens(
        self,
        num_tokens: int,
        num_experts: int,
        ep_size: int,
        dim: int,
        balanced: bool,
        benchmark: bool,
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
                self.skipTest("num_tokens must be >= num_experts")
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
            scatter_add_padded_tokens(
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

        bench_fn = do_bench_cudagraph_and_clear_cache if torch.version.hip else do_bench
        if benchmark:
            mem_bytes = ref_out_tokens.numel() * 2 * 3
            fbgemm_time = bench_fn(fn) * 1e3
            fbgemm_bw = mem_bytes * 1e-9 / (fbgemm_time * 1e-6)
            # We don't benchmark counting on CPU
            torch_time = bench_fn(ref_fn) * 1e3
            torch_bw = mem_bytes * 1e-9 / (torch_time * 1e-6)

            logger.info(
                f"\nnum_tokens={num_tokens:4}, ep_size={ep_size:4}, num_experts={num_experts:4}, balanced={int(balanced)}, "
                f"fbgemm_time={fbgemm_time:7.3f}us, fbgemm_bw={fbgemm_bw:8.3f}GBytes/s,  "
                f"torch_time={torch_time:7.3f}us, torch_bw={torch_bw:8.3f}GBytes/s, speedup={torch_time / fbgemm_time:7.3f}x",
            )


if __name__ == "__main__":

    unittest.main()
