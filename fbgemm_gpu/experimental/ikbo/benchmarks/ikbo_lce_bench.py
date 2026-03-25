# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from functools import partial

import torch
import triton
from ikbo.ops.tlx_ikbo_lce import create_user_flag, tlx_ikbo_lce
from ikbo.ops.torch_lce import torch_decomposed_lce, torch_lce
from ikbo.ops.triton_ikbo_lce import triton_ikbo_lce
from torch._inductor.utils import do_bench_using_profiling

DEVICE = "cuda"
DTYPE = torch.float16
PAD_UNIT = 8  # for fp16/bf16

# Representative realistic dimensions.
# M is non-round because torch.compile fuses multiple LCE modules (with output
# sizes like 128, 64, 32, ...) into one batched matmul; M is their sum.
M, N, K_USER, K_CAND = 433, 256, 1178, 866
DEFAULT_B = 1024
DEFAULT_CAND_TO_USER_RATIO = 70

PROVIDERS = [
    "baseline",
    "opt_1_decomposition",
    "opt_2_k_dim_alignment",
    "opt_3_kernel_fusion",
    "opt_4_tlx",
]
PROVIDER_NAMES = [
    "Baseline",
    "Opt 1 Decomposition",
    "Opt 2 K-dim Alignment",
    "Opt 3 Kernel Fusion",
    "Opt 4 TLX",
]


def prepare_inputs_by_config(
    B: int,
    M: int,
    N: int,
    K_USER: int,
    K_CAND: int,
    low_num_cands_per_user: int = DEFAULT_CAND_TO_USER_RATIO,
    high_num_cands_per_user: int = DEFAULT_CAND_TO_USER_RATIO,
    pad_k: bool = False,
):
    def _generate_num_cands_per_user():
        res = []
        cum_sum = 0
        while True:
            cur = random.randint(low_num_cands_per_user, high_num_cands_per_user)
            if cum_sum + cur >= B:
                res.append(B - cum_sum)
                break
            cum_sum += cur
            res.append(cur)
        return res

    if pad_k:
        K_USER = ((K_USER + PAD_UNIT - 1) // PAD_UNIT) * PAD_UNIT
        K_CAND = ((K_CAND + PAD_UNIT - 1) // PAD_UNIT) * PAD_UNIT

    num_cands_per_user_tensor = torch.tensor(_generate_num_cands_per_user())
    user_batch_size = num_cands_per_user_tensor.size(0)
    cand_to_user_index = (
        torch.repeat_interleave(
            torch.arange(user_batch_size),
            num_cands_per_user_tensor,
        )
        .int()
        .to(DEVICE)
    )
    compression_w_cand = torch.randn(
        (M, K_CAND), device=DEVICE, dtype=DTYPE
    ).requires_grad_(False)
    compression_w_user = torch.randn(
        (M, K_USER), device=DEVICE, dtype=DTYPE
    ).requires_grad_(False)
    embeddings_cand = torch.randn(
        (B, K_CAND, N), device=DEVICE, dtype=DTYPE
    ).requires_grad_(False)
    embeddings_user = torch.randn(
        (user_batch_size, K_USER, N), device=DEVICE, dtype=DTYPE
    ).requires_grad_(False)

    # Concatenated inputs for baseline LCE
    compression_w = torch.cat((compression_w_user, compression_w_cand), dim=1)
    embeddings_user_broadcast = torch.index_select(
        embeddings_user, 0, cand_to_user_index
    )
    embeddings = torch.cat((embeddings_user_broadcast, embeddings_cand), dim=1)

    return (
        compression_w_cand,
        compression_w_user,
        embeddings_cand,
        embeddings_user,
        cand_to_user_index,
        compression_w,
        embeddings,
    )


def prepare_default_inputs(pad_k: bool = False):
    return prepare_inputs_by_config(DEFAULT_B, M, N, K_USER, K_CAND, pad_k=pad_k)


def _run_provider(provider, B, num_cands_per_user):
    torch.manual_seed(0)
    # Fixed ratio ensures fair comparison across providers
    kwargs = dict(
        low_num_cands_per_user=num_cands_per_user,
        high_num_cands_per_user=num_cands_per_user,
    )

    if provider == "baseline":
        *_, compression_w, embeddings = prepare_inputs_by_config(
            B, M, N, K_USER, K_CAND, **kwargs
        )
        fn = partial(torch_lce, compression_w, embeddings)
    elif provider == "opt_1_decomposition":
        cw_c, cw_u, e_c, e_u, idx, _, _ = prepare_inputs_by_config(
            B, M, N, K_USER, K_CAND, **kwargs
        )
        fn = partial(torch_decomposed_lce, cw_c, cw_u, e_c, e_u, idx)
    elif provider == "opt_2_k_dim_alignment":
        cw_c, cw_u, e_c, e_u, idx, _, _ = prepare_inputs_by_config(
            B, M, N, K_USER, K_CAND, **kwargs, pad_k=True
        )
        fn = partial(torch_decomposed_lce, cw_c, cw_u, e_c, e_u, idx)
    elif provider == "opt_3_kernel_fusion":
        cw_c, cw_u, e_c, e_u, idx, _, _ = prepare_inputs_by_config(
            B, M, N, K_USER, K_CAND, **kwargs, pad_k=True
        )
        fn = partial(triton_ikbo_lce, cw_c, cw_u, e_c, e_u, idx)
    elif provider == "opt_4_tlx":
        cw_c, cw_u, e_c, e_u, idx, _, _ = prepare_inputs_by_config(
            B, M, N, K_USER, K_CAND, **kwargs, pad_k=True
        )
        user_flag = create_user_flag(cw_u, e_u)
        fn = partial(tlx_ikbo_lce, cw_c, cw_u, e_c, e_u, idx, user_flag)
    else:
        return 100

    return do_bench_using_profiling(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[b * 512 for b in range(1, 7)],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        ylabel="Latency (ms)",
        plot_name="IKBO LCE - Vary Batch Size",
        args={},
    )
)
def benchmark_vary_batch(B, provider):
    return _run_provider(provider, B, DEFAULT_CAND_TO_USER_RATIO)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["cand_to_user_ratio"],
        x_vals=[2, 5, 10, 50, 100, 1000],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        ylabel="Latency (ms)",
        plot_name="IKBO LCE - Vary Candidate-to-User Ratio",
        args={},
    )
)
def benchmark_vary_ratio(cand_to_user_ratio, provider):
    return _run_provider(provider, DEFAULT_B, cand_to_user_ratio)


def main():
    benchmark_vary_batch.run(show_plots=False, print_data=True)
    benchmark_vary_ratio.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
