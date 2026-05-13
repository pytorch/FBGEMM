# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Dict, List

import click
import pandas as pd
import torch
import triton  # @manual

from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
    grouped_gemm,
    grouped_gemm_bias_scale,
)


def triton_fused_bench(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    bias: torch.Tensor,
    token_weights: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """Factory for Triton fused grouped_gemm + bias + token_weights."""

    def run() -> torch.Tensor:
        return grouped_gemm_bias_scale(
            x, w, m_sizes, bias=bias, token_weights=token_weights
        )

    return run


@torch.compile(mode="reduce-overhead")
def _torch_bmm_bias_scale(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    token_weights: torch.Tensor,
    G: int,
    M_per_group: int,
) -> torch.Tensor:
    """Compiled torch baseline: bmm + bias + scale."""
    N = w.shape[0] // G
    K = w.shape[1]
    x_3d = x.view(G, M_per_group, K)
    w_3d = w.view(G, N, K)
    out = torch.bmm(x_3d, w_3d.transpose(-1, -2))
    out = out + bias.unsqueeze(1)
    out = out * token_weights.view(G, M_per_group, 1)
    return out.view(-1, N)


def torch_baseline_bench(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    token_weights: torch.Tensor,
    G: int,
    M_per_group: int,
) -> Callable[[], torch.Tensor]:
    """Factory for torch.compile'd batched matmul baseline."""

    def run() -> torch.Tensor:
        return _torch_bmm_bias_scale(x, w, bias, token_weights, G, M_per_group)

    return run


def triton_gemm_torch_bias_scale_bench(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    bias: torch.Tensor,
    token_weights: torch.Tensor,
    G: int,
    M_per_group: int,
) -> Callable[[], torch.Tensor]:
    """Factory for Triton grouped_gemm + torch bias + torch token_weights."""

    def run() -> torch.Tensor:
        out = grouped_gemm(x, w, m_sizes)
        out_3d = out.view(G, M_per_group, -1)
        out_3d = out_3d + bias.unsqueeze(1)
        out_3d = out_3d * token_weights.view(G, M_per_group, 1)
        return out_3d.view(-1, out.shape[-1])

    return run


@click.command()
@click.option("--warmup", type=int, default=25, help="Warmup iterations")
@click.option("--rep", type=int, default=10, help="Benchmark repetitions")
def bench(warmup: int, rep: int) -> None:
    """Benchmark grouped_gemm_bias_scale vs torch baseline."""
    device = torch.accelerator.current_accelerator()
    dtype = torch.bfloat16

    # G: Number of experts/groups in the MoE layer
    # M: Total number of tokens across all groups
    # N: Output dimension (hidden size of expert output)
    # K: Input dimension (hidden size of expert input)
    configs = [
        {"G": 4, "M": 512, "N": 256, "K": 256, "name": "Small"},
        {"G": 16, "M": 4096, "N": 512, "K": 512, "name": "Medium"},
        {"G": 64, "M": 16384, "N": 512, "K": 512, "name": "Large"},
    ]

    # Print configuration table
    config_df = pd.DataFrame(configs).rename(
        columns={
            "name": "Config",
            "G": "G (experts)",
            "M": "M (tokens)",
            "N": "N (out_dim)",
            "K": "K (in_dim)",
        }
    )[["Config", "G (experts)", "M (tokens)", "N (out_dim)", "K (in_dim)"]]
    print("\nBenchmark Configurations:")
    print(config_df.to_string(index=False))
    print()

    results: List[Dict[str, str]] = []

    for idx, cfg in enumerate(configs):
        G: int = cfg["G"]  # pyre-ignore[9]
        M: int = cfg["M"]  # pyre-ignore[9]
        N: int = cfg["N"]  # pyre-ignore[9]
        K: int = cfg["K"]  # pyre-ignore[9]
        name: str = cfg["name"]  # pyre-ignore[9]
        M_per_group = M // G

        print(f"Processing config {idx + 1}/{len(configs)}: {name}...")

        # Create tensors
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(G * N, K, dtype=dtype, device=device)
        bias = torch.randn(G, N, dtype=dtype, device=device)
        token_weights = torch.rand(M, dtype=dtype, device=device) + 0.5
        m_sizes = torch.full((G,), M_per_group, dtype=torch.int32, device=device)

        # Create benchmark functions
        triton_fn = triton_fused_bench(x, w, m_sizes, bias, token_weights)
        triton_torch_fn = triton_gemm_torch_bias_scale_bench(
            x, w, m_sizes, bias, token_weights, G, M_per_group
        )
        torch_fn = torch_baseline_bench(x, w, bias, token_weights, G, M_per_group)

        # Warmup torch.compile
        for _ in range(3):
            torch_fn()
        torch.cuda.synchronize()

        # Benchmark
        fused_ms = triton.testing.do_bench(triton_fn, warmup=warmup, rep=rep)
        triton_torch_ms = triton.testing.do_bench(
            triton_torch_fn, warmup=warmup, rep=rep
        )
        torch_ms = triton.testing.do_bench(torch_fn, warmup=warmup, rep=rep)

        results.append(
            {
                "Config": name,
                "fused (ms)": f"{fused_ms:.3f}",
                "triton+torch (ms)": f"{triton_torch_ms:.3f}",
                "torch (ms)": f"{torch_ms:.3f}",
                "Speedup vs torch": f"{torch_ms / fused_ms:.2f}x",
                "Speedup vs triton+torch": f"{triton_torch_ms / fused_ms:.2f}x",
            }
        )

    print("\nBenchmark Results:")
    print(pd.DataFrame(results).to_string(index=False))
    print()


if __name__ == "__main__":
    bench()
