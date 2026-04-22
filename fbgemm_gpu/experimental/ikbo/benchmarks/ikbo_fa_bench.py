# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from functools import partial

import torch
import triton
from ikbo.ops.tlx_ikbo_fa_ws import tlx_flash_attn_ikbo_tma_persistent
from ikbo.ops.triton_ikbo_fa import triton_flash_attn_ikbo_tma
from torch._inductor.utils import do_bench_using_profiling

num_heads, n_seed, d_head = 2, 64, 128
DEFAULT_B = 2048
DEFAULT_CAND_TO_USER_RATIO = 64
DEVICE = "cuda"
DTYPE = torch.float16

PROVIDERS = [
    "Inductor SDPA",
    "Broadcast + inductor SDPA",
    "Triton IKBO FA2",
    "TLX IKBO FA3 persistence generalized",
]
PROVIDER_NAMES = [
    "Inductor SDPA",
    "Broadcast + inductor SDPA",
    "Triton IKBO FA2",
    "TLX IKBO FA3 persistence generalized",
]


def pytorch_sdpa(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    )


def broadcast_sdpa(
    query, key, value, cand_to_user_index, n_seed, num_heads, d_head, max_seq_len
):
    # for accuracy check
    query_sdpa = query.view(-1, n_seed, num_heads, d_head).permute(0, 2, 1, 3)
    key_sdpa = key.view(-1, max_seq_len, num_heads, d_head)
    key_sdpa_broadcast = torch.index_select(
        key_sdpa, dim=0, index=cand_to_user_index
    ).permute(0, 2, 1, 3)
    value_sdpa = value.view(-1, max_seq_len, num_heads, d_head)
    value_sdpa_broadcast = torch.index_select(
        value_sdpa, dim=0, index=cand_to_user_index
    ).permute(0, 2, 1, 3)
    return pytorch_sdpa(query_sdpa, key_sdpa_broadcast, value_sdpa_broadcast)


def prepare_inputs_by_config(
    B: int,
    n_seed: int,
    num_heads: int,
    d_head: int,
    max_seq_len: int,
    low_num_cands_per_user: int = DEFAULT_CAND_TO_USER_RATIO,
    high_num_cands_per_user: int = DEFAULT_CAND_TO_USER_RATIO,
):
    def _generate_num_cands_per_user():
        res = []
        cum_sum = 0
        cand_grid = []
        while True:
            # Odd and even number of candidates per user got even chance
            cur = random.randint(
                low_num_cands_per_user, high_num_cands_per_user
            ) + random.randint(0, 1)
            for grid in range(cum_sum, min(cum_sum + cur, B), 2):
                cand_grid.append(grid)
            if cum_sum + cur >= B:
                res.append(B - cum_sum)
                break
            cum_sum += cur
            res.append(cur)
        return res, cand_grid

    res = _generate_num_cands_per_user()
    num_cands_per_user_tensor = torch.tensor(res[0])
    cand_grid = torch.tensor(res[1], dtype=torch.int32, device=DEVICE)

    cand_to_user_index = torch.repeat_interleave(
        torch.arange(num_cands_per_user_tensor.size(0)),
        num_cands_per_user_tensor,
    ).to(dtype=torch.int32, device=DEVICE)
    Bu = num_cands_per_user_tensor.size(0)

    query = torch.randn((B * n_seed, num_heads, d_head), device=DEVICE, dtype=DTYPE)
    key = torch.randn((Bu * max_seq_len, num_heads, d_head), device=DEVICE, dtype=DTYPE)
    value = torch.randn(
        (Bu * max_seq_len, num_heads, d_head), device=DEVICE, dtype=DTYPE
    )
    return (
        query,
        key,
        value,
        cand_to_user_index,
        cand_grid,
    )


def _run_provider(provider, seq_len):
    torch.manual_seed(0)
    q, k, v, cand_to_user_index, cand_grid = prepare_inputs_by_config(
        B=DEFAULT_B,
        n_seed=n_seed,
        num_heads=num_heads,
        d_head=d_head,
        max_seq_len=seq_len,
        low_num_cands_per_user=DEFAULT_CAND_TO_USER_RATIO,
        high_num_cands_per_user=DEFAULT_CAND_TO_USER_RATIO,
    )
    q_sdpa = q.view(-1, n_seed, num_heads, d_head).permute(0, 2, 1, 3)
    k_sdpa = k.view(-1, seq_len, num_heads, d_head)
    k_broadcast = torch.index_select(k_sdpa, dim=0, index=cand_to_user_index).permute(
        0, 2, 1, 3
    )
    v_sdpa = v.view(-1, seq_len, num_heads, d_head)
    v_broadcast = torch.index_select(v_sdpa, dim=0, index=cand_to_user_index).permute(
        0, 2, 1, 3
    )

    def flops(ms):
        return (DEFAULT_B * num_heads * n_seed * d_head * seq_len * 4) / ms * 1e-9

    if provider == "Inductor SDPA":
        eager_fn = partial(pytorch_sdpa, q_sdpa, k_broadcast, v_broadcast)
        fn = torch.compile(
            eager_fn,
            backend="inductor",
            options={"max_autotune": True},
        )
    elif provider == "Broadcast + inductor SDPA":
        eager_fn = partial(
            broadcast_sdpa,
            q,
            k,
            v,
            cand_to_user_index,
            n_seed,
            num_heads,
            d_head,
            seq_len,
        )
        fn = torch.compile(
            eager_fn,
            backend="inductor",
            options={"max_autotune": True},
        )
    elif provider == "Triton IKBO FA2":
        fn = partial(
            triton_flash_attn_ikbo_tma, q, k, v, cand_to_user_index, n_seed, seq_len
        )
    elif provider == "TLX IKBO FA3 persistence generalized":
        fn = partial(
            tlx_flash_attn_ikbo_tma_persistent,
            q,
            k,
            v,
            cand_to_user_index,
            n_seed,
            seq_len,
            cand_grid,
        )
    else:
        return 100

    return flops(do_bench_using_profiling(fn))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[512, 1024, 2048, 4096, 8192, 16384],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        ylabel="Latency (ms)",
        plot_name="IKBO FA latency - Sequence Length",
        args={},
    )
)
def benchmark_vary_seq(seq_len, provider):
    return _run_provider(provider, seq_len)


def main():
    benchmark_vary_seq.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
