# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import functools
import itertools
from typing import List, Optional, Tuple

import torch
import triton  # noqa: F401
from fbgemm_gpu.experimental.gen_ai.moe import (
    combine_shuffling,
    gather_along_first_dim,
    gather_scale_dense_tokens,
    gather_scale_quant_dense_tokens,
    index_shuffling,
    scatter_add_along_first_dim,
    scatter_add_dense_tokens,
    split_shuffling,
)
from triton.testing import do_bench, do_bench_cudagraph

_ACCELERATOR_TAG = torch.accelerator.current_accelerator()


def bench_gather_along_first_dim(M: int, N: int, K: int) -> None:
    src = torch.randn([M, K], device=_ACCELERATOR_TAG, dtype=torch.bfloat16).abs()
    if M == N:
        indices = torch.randperm(N, device=_ACCELERATOR_TAG, dtype=torch.int32)
    else:
        indices = torch.randint(0, M, [N], device=_ACCELERATOR_TAG, dtype=torch.int32)

    def fn():
        return gather_along_first_dim(src, indices)

    def ref_fn():
        return torch.index_select(src, 0, indices)

    # Load src, store dst. x2.
    data_size_in_gigabytes = N * K * 2 * 2 / 1e9

    time_in_us = triton.testing.do_bench(fn) * 1e3
    time_in_second = time_in_us / 1e6
    gigabytes_per_second = data_size_in_gigabytes / time_in_second

    ref_time_in_us = triton.testing.do_bench(ref_fn) * 1e3
    ref_time_in_second = ref_time_in_us / 1e6
    ref_gigabytes_per_second = data_size_in_gigabytes / ref_time_in_second

    print(
        f"Benchmark gather_along_first_dim: {M=:5d}, {N=:5d}, {K=:5d}, "
        f"FBGEMM time: {time_in_us:10.3f} us. Bandwidth: {gigabytes_per_second:10.3f} GB/s, "
        f"Torch time: {ref_time_in_us:10.3f} us. Bandwidth: {ref_gigabytes_per_second:10.3f} GB/s"
    )


def bench_scatter_add_along_first_dim_(op, M: int, N: int, K: int) -> None:
    src = torch.randn([M, K], device=_ACCELERATOR_TAG, dtype=torch.bfloat16).abs()
    dst = torch.randn([N, K], device=_ACCELERATOR_TAG, dtype=torch.bfloat16).abs()
    if M == N:
        indices_1d = torch.randperm(N, device=_ACCELERATOR_TAG, dtype=torch.int64)
    else:
        indices_1d = torch.randint(
            0, N, [M], device=_ACCELERATOR_TAG, dtype=torch.int64
        )

    indices_2d = indices_1d.to(torch.int64).unsqueeze(1).expand(-1, K)

    test_dst = dst.clone()
    ref_dst = dst.clone()

    def fn():
        op(test_dst, src, indices_1d)

    def ref_fn():
        ref_dst.scatter_add_(0, indices_2d, src)

    # Load src, load dst, store dst. x3.
    data_size_in_gigabytes = N * K * 2 * 3 / 1e9

    time_in_us = triton.testing.do_bench(fn) * 1e3
    time_in_second = time_in_us / 1e6
    gigabytes_per_second = data_size_in_gigabytes / time_in_second

    ref_time_in_us = triton.testing.do_bench(ref_fn) * 1e3
    ref_time_in_second = ref_time_in_us / 1e6
    ref_gigabytes_per_second = data_size_in_gigabytes / ref_time_in_second

    print(
        f"Benchmark {op.__name__}: {M=:5d}, {N=:5d}, {K=:5d}, "
        f"FBGEMM time: {time_in_us:10.3f} us. Bandwidth: {gigabytes_per_second:10.3f} GB/s, "
        f"Torch time: {ref_time_in_us:10.3f} us. Bandwidth: {ref_gigabytes_per_second:10.3f} GB/s"
    )


bench_scatter_add_along_first_dim = functools.partial(
    bench_scatter_add_along_first_dim_, scatter_add_along_first_dim
)

bench_scatter_add_dense_tokens = functools.partial(
    bench_scatter_add_along_first_dim_, scatter_add_dense_tokens
)


def bench_gather_scale_dense_tokens(E: int, T: int, D: int, quantize: bool):
    x = torch.randn((T, D), dtype=torch.bfloat16, device=_ACCELERATOR_TAG).abs()
    expert_indices = torch.randint(0, E, (T,), device=_ACCELERATOR_TAG)
    token_indices = torch.randperm(T, device=_ACCELERATOR_TAG)
    scores = torch.rand((E, T), dtype=torch.bfloat16, device=_ACCELERATOR_TAG)

    def torch_fn():
        shuffled_x = torch.index_select(x, dim=0, index=token_indices)
        shuffled_scores = torch.index_select(scores, dim=1, index=token_indices)
        shuffled_selected_scores = torch.gather(
            shuffled_scores, dim=0, index=expert_indices.view(1, T)
        )
        ref_output = shuffled_x * shuffled_selected_scores.view(-1, 1)
        return ref_output

    torch_fn()

    scores_TE = scores.transpose(0, 1).contiguous()

    fbgemm_fn = (
        gather_scale_quant_dense_tokens if quantize else gather_scale_dense_tokens
    )

    def triton_fn():
        test_output = fbgemm_fn(x, token_indices, expert_indices, scores_TE)
        return test_output

    triton_fn()

    # Run benchmark
    if quantize:
        data_size_in_gigabytes = T * D * 3 / 1e9
    else:
        data_size_in_gigabytes = T * D * 4 / 1e9

    fbgemm_time = do_bench(triton_fn, rep=1000) * 1e3
    fbgemm_bw = data_size_in_gigabytes / (fbgemm_time / 1e6)

    torch_time = do_bench(torch_fn, rep=1000) * 1e3
    torch_bw = data_size_in_gigabytes / (torch_time / 1e6)
    print(
        f"Benchmark gather_scale_dense_tokens({quantize=}), {E=:3d}, {T=:5d}, {D=:5d}, "
        f"FBGEMM time: {fbgemm_time:10.3f} us. Bandwidth: {fbgemm_bw:10.3f} GB/s, "
        f"Torch time: {torch_time:10.3f} us. Bandwidth: {torch_bw:10.3f} GB/s"
    )


def bench_topk_index_shuffling(T: int, E: int, K: int) -> None:
    torch.manual_seed(0)

    num_rotating_buffers = min(max(2, triton.cdiv(1024 * 1024 * 1024, T * E * 2)), 1000)
    scores_list: List[torch.Tensor] = [
        torch.randn(T, E, device=_ACCELERATOR_TAG, dtype=torch.bfloat16)
        for i in range(num_rotating_buffers)
    ]

    def fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for scores in scores_list:
            index_shuffling(scores, top_k=K)

    def ref_fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for scores in scores_list:
            _, selected_expert_indices = torch.topk(scores, K, dim=1)
            expert_indices, _ = torch.sort(
                selected_expert_indices.flatten(), dim=0, stable=True
            )
            _ = (
                expert_indices[:, None]
                == torch.arange(E, device=expert_indices.device)[None, :]
            ).sum(dim=0)

    fbgemm_time = do_bench_cudagraph(fn) * 1e3 / num_rotating_buffers
    torch_time = do_bench_cudagraph(ref_fn) * 1e3 / num_rotating_buffers
    print(
        f"Benchmark index_shuffling, num_tokens={T:4}, num_experts={E:4}, top_k={K:4}, "
        f"fbgemm_time={fbgemm_time:7.3f}us, torch_time={torch_time:7.3f}us"
    )


def bench_combine_or_split_shuffling(
    T: int,
    D: int,
    E: int,
    EP: bool,
    is_padded: bool,
    is_balanced: bool,
    is_combine_shuffling: bool,
):
    torch.manual_seed(0)

    assert E % EP == 0
    if is_padded:
        # graph. allgather
        input_num_tokens: int = EP * T
        input_num_experts: int = E
        output_num_experts: int = E // EP
        start_expert_index: int = 1
        end_expert_index: int = 1 + output_num_experts
    else:
        # eager. all2all
        input_num_tokens: int = T
        input_num_experts: int = E // EP
        output_num_experts: int = E // EP
        start_expert_index: int = 0
        end_expert_index: int = output_num_experts

    tokens = torch.randn(
        input_num_tokens, D, device=_ACCELERATOR_TAG, dtype=torch.bfloat16
    )

    if input_num_tokens < (EP * input_num_experts) != 0:
        return

    input_num_tokens_per_expert: int = input_num_tokens // (EP * input_num_experts)
    token_counts: torch.Tensor = (
        torch.ones(
            [EP, input_num_experts],
            dtype=torch.int32,
            device=_ACCELERATOR_TAG,
        )
        * input_num_tokens_per_expert
    )
    if not is_balanced:
        for i in range(EP):
            token_counts[i, start_expert_index] -= input_num_tokens_per_expert
            token_counts[i, end_expert_index - 1] += input_num_tokens_per_expert

    assert token_counts.sum().item() == input_num_tokens

    num_rotating_buffers = triton.cdiv(1024 * 1024 * 1024, tokens.numel() * 2)
    token_list: List[torch.Tensor] = [
        tokens.clone() for _ in range(num_rotating_buffers)
    ]
    token_count_list: List[torch.Tensor] = [
        token_counts.clone() for _ in range(num_rotating_buffers)
    ]

    def fn() -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for tokens, token_counts in zip(token_list, token_count_list):
            if is_combine_shuffling:
                combine_shuffling(
                    tokens,
                    token_counts,
                    expert_start=start_expert_index,
                    expert_end=end_expert_index,
                    is_balanced=is_balanced,
                )
            else:
                split_shuffling(
                    tokens,
                    token_counts,
                    expert_start=start_expert_index,
                    expert_end=end_expert_index,
                    is_balanced=is_balanced,
                )

    fn()

    output_num_tokens = 0
    for per_rank_counts in token_counts.tolist():
        for expert_index, per_expert_counts in enumerate(per_rank_counts):
            if expert_index >= start_expert_index and expert_index < end_expert_index:
                output_num_tokens += per_expert_counts

    mem_bytes = output_num_tokens * D * 2 * 2
    fbgemm_time = do_bench_cudagraph(fn) * 1e3 / num_rotating_buffers
    fbgemm_bw = mem_bytes * 1e-9 / (fbgemm_time * 1e-6)

    print(
        f"Benchmark {'combine_shuffling' if is_combine_shuffling else 'split_shuffling'}, "
        f"num_tokens={T:4}, dim={D:4}, num_experts={E:4}, expert_parallelism={EP:4}, output_num_tokens={output_num_tokens:4}, "
        f"{is_balanced=}, {is_padded=}, "
        f"fbgemm_time={fbgemm_time:7.3f}us, fbgemm_bw={fbgemm_bw:8.3f}GBytes/s."
    )


def main(kernels: Optional[str]):
    if kernels is not None:
        kernels = kernels.split(",")

    def should_bench_kernel(fn):
        return (fn is not None) and (kernels is None or fn.__name__ in kernels)

    Es = [16, 128]
    Ts = [1, 128, 2048, 4096, 8192, 16384]
    Ds = [5120]

    # Gather/Scatter
    if should_bench_kernel(gather_scale_dense_tokens):
        for E, T, D in itertools.product(Es, Ts, Ds):
            bench_gather_scale_dense_tokens(E, T, D, quantize=False)

    if should_bench_kernel(gather_scale_quant_dense_tokens):
        for E, T, D in itertools.product(Es, Ts, Ds):
            bench_gather_scale_dense_tokens(E, T, D, quantize=True)

    if should_bench_kernel(gather_along_first_dim):
        for T, D in itertools.product(Ts, Ds):
            bench_gather_along_first_dim(T, T, D)

    if should_bench_kernel(scatter_add_along_first_dim):
        for T, D in itertools.product(Ts, Ds):
            bench_scatter_add_along_first_dim(T, T, D)

    if should_bench_kernel(scatter_add_dense_tokens):
        for T, D in itertools.product(Ts, Ds):
            bench_scatter_add_dense_tokens(T, T, D)

    Ks = [1, 2, 4]
    Es = [16, 32, 128, 320]
    # Shuffling
    if should_bench_kernel(index_shuffling):
        for T, E, K in itertools.product(Ts, Es, Ks):
            bench_topk_index_shuffling(T, E, K)

    EPs = [2, 16]
    Ts = [32, 128, 2048, 4096, 8192, 16384]
    padded = [True, False]
    balanced = [True, False]

    if should_bench_kernel(combine_shuffling):
        for T, D, E, EP, p, b in itertools.product(Ts, Ds, Es, EPs, padded, balanced):
            bench_combine_or_split_shuffling(
                T, D, E, EP, p, b, is_combine_shuffling=True
            )

    if should_bench_kernel(split_shuffling):
        for T, D, E, EP, p, b in itertools.product(Ts, Ds, Es, EPs, padded, balanced):
            bench_combine_or_split_shuffling(
                T, D, E, EP, p, b, is_combine_shuffling=False
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernels",
        default=None,
        help="Comma separated list of kernels to benchmark. Defaults to all kernels.",
    )
    args = parser.parse_args()
    main(args.kernels)
