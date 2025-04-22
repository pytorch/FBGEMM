# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import List, Tuple

import torch
import triton  # noqa: F401
from fbgemm_gpu.experimental.gen_ai.moe import (
    gather_along_first_dim,
    gather_scale_dense_tokens,
    gather_scale_quant_dense_tokens,
    index_shuffling,
    scatter_add_along_first_dim,
)
from triton.testing import do_bench, do_bench_cudagraph


def bench_gather_along_first_dim(M: int, N: int, K: int) -> None:
    src = torch.randn([M, K], device="cuda", dtype=torch.bfloat16).abs()
    if M == N:
        indices = torch.randperm(N, device="cuda", dtype=torch.int32)
    else:
        indices = torch.randint(0, M, [N], device="cuda", dtype=torch.int32)

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


def bench_scatter_add_along_first_dim(M: int, N: int, K: int) -> None:
    src = torch.randn([M, K], device="cuda", dtype=torch.bfloat16).abs()
    dst = torch.randn([N, K], device="cuda", dtype=torch.bfloat16).abs()
    if M == N:
        indices_1d = torch.randperm(N, device="cuda", dtype=torch.int64)
    else:
        indices_1d = torch.randint(0, N, [M], device="cuda", dtype=torch.int64)

    indices_2d = indices_1d.to(torch.int64).unsqueeze(1).expand(-1, K)

    test_dst = dst.clone()
    ref_dst = dst.clone()

    def fn():
        scatter_add_along_first_dim(test_dst, src, indices_1d)

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
        f"Benchmark scatter_add_along_first_dim: {M=:5d}, {N=:5d}, {K=:5d}, "
        f"FBGEMM time: {time_in_us:10.3f} us. Bandwidth: {gigabytes_per_second:10.3f} GB/s, "
        f"Torch time: {ref_time_in_us:10.3f} us. Bandwidth: {ref_gigabytes_per_second:10.3f} GB/s"
    )


def bench_gather_scale_dense_tokens(E: int, T: int, D: int, quantize: bool):
    x = torch.randn((T, D), dtype=torch.bfloat16, device="cuda").abs()
    expert_indices = torch.randint(0, E, (T,), device="cuda")
    token_indices = torch.randperm(T, device="cuda")
    scores = torch.rand((E, T), dtype=torch.bfloat16, device="cuda")

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


def bench_top1_index_shuffling(num_tokens: int, num_experts: int) -> None:
    torch.manual_seed(0)

    scores_list: List[torch.Tensor] = [
        torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.bfloat16)
        for i in range(100)
    ]

    def fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for scores in scores_list:
            index_shuffling(scores)

    def ref_fn() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for scores in scores_list:
            _, selected_expert_indices = torch.topk(scores, 1, dim=1)
            expert_indices, _ = torch.sort(selected_expert_indices, dim=0)
            _ = (
                expert_indices[:, None]
                == torch.arange(num_experts, device=expert_indices.device)[None, :]
            ).sum(dim=0)

    fbgemm_time = do_bench_cudagraph(fn) * 1e3 / 100
    torch_time = do_bench_cudagraph(ref_fn) * 1e3 / 100
    print(
        f"Benchmark index_shuffling, num_tokens={num_tokens:4}, num_experts={num_experts:4}, "
        f"fbgemm_time={fbgemm_time:7.3f}us, torch_time={torch_time:7.3f}us"
    )


def main():
    Es = [16, 128]
    Ts = [1, 128, 2048, 4096, 8192, 16384]
    Ds = [5120]

    for E, T, D in itertools.product(Es, Ts, Ds):
        bench_gather_scale_dense_tokens(E, T, D, quantize=False)

    for E, T, D in itertools.product(Es, Ts, Ds):
        bench_gather_scale_dense_tokens(E, T, D, quantize=True)

    if gather_along_first_dim is not None:
        for T, D in itertools.product(Ts, Ds):
            bench_gather_along_first_dim(T, T, D)

    if scatter_add_along_first_dim is not None:
        for T, D in itertools.product(Ts, Ds):
            bench_scatter_add_along_first_dim(T, T, D)

    for T, E in itertools.product(Ts, Es):
        bench_top1_index_shuffling(T, E)


if __name__ == "__main__":
    main()
