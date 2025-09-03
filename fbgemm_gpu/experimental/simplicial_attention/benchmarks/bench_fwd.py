# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
import triton

try:
    from flash_attn_interface import flash_attn_func as fa3

    HAS_FA3 = True
except ImportError:
    print("Flash Attention 3 not found, skipping benchmarking")
    fa3 = None
    HAS_FA3 = False

from simplicial.ops.tlx.fwd_ws import tlx_fwd_ws
from simplicial.ops.tlx.fwd_ws_pingpong import tlx_fwd_ws_pingpong
from simplicial.ops.triton.fwd import triton_fwd
from simplicial.utils import get_simplicial_tensor_core_tflops

B = 4
W1 = 32
Hk = 1
D = 128

Ns = [1024 * i for i in [1, 2, 4, 8, 16]]
Hqs = [64, 128]
W2s = [256, 512, 1024]

N_W2_Hq_SHAPES = [(n, w2, hq) for hq in Hqs for w2 in W2s for n in Ns]


@triton.testing.perf_report(
    [
        # columns
        triton.testing.Benchmark(
            x_names=["N", "w2", "Hq"],  # printed along X-axis
            x_vals=N_W2_Hq_SHAPES,  # KV window sizes to test
            x_log=False,
            line_arg="provider",  # what are we comparing?
            line_vals=[
                "triton_fwd",
                "tlx_fwd_ws",
                "tlx_fwd_ws_pingpong",
                "fa3_fwd",
            ],
            line_names=[
                "triton_fwd",
                "tlx_fwd_ws",
                "tlx_fwd_ws_pingpong",
                "fa3_fwd",
            ],
            styles=[
                ("red", "-"),
                ("yellow", "-"),
                ("blue", "-"),
                ("green", "-"),
            ],
            ylabel="Latency (ms)",
            plot_name=f"simplicial-benchmark-{B=}-{Hk=}-{D=}-{W1=}-tflops",
            args={
                "B": B,  # batch
                "Hk": Hk,  # kv_heads
                "D": D,  # head_dim
                "w1": W1,
            },
        )
    ]
)
def _benchmark(B, N, w2, Hq, Hk, D, w1, provider):
    torch.manual_seed(7)
    device = torch.accelerator.current_accelerator()

    Q = torch.randn(B, N, Hq, D, device=device, dtype=torch.bfloat16)

    IS_CAUSAL = False
    KV_LEN_2D = w1 * w2

    if provider == "tlx_fwd_ws_pingpong" and Hq != 128:
        # `pingpong` only supports Hq=128 for 2 consumer warpgroups
        return -1.0

    if provider == "fa3_fwd":
        if not HAS_FA3:
            return -1.0

        K1, K2, V1, V2 = None, None, None, None
        K = torch.randn(B, KV_LEN_2D, Hk, D, dtype=torch.bfloat16, device=device)
        V = torch.randn_like(K)
    else:
        K1 = torch.randn(B, N, Hk, D, device=device, dtype=torch.bfloat16)
        K2 = torch.randn_like(K1)
        V1 = torch.randn_like(K1)
        V2 = torch.randn_like(K1)
        K, V = None, None

    def _total_2d_tflops():
        # Original: 1 multiply + 1 add = 2
        ratio = 0.5 if IS_CAUSAL else 1.0
        kv_len = K.shape[1]
        QK_GEMM = B * Hq * N * D * kv_len * 2
        PV_GEMM = B * Hq * N * D * kv_len * 2
        return (QK_GEMM + PV_GEMM) * ratio / 1e12

    total_tflops = (
        get_simplicial_tensor_core_tflops(B, N, Hq, Hk, D, w1, w2)
        if not provider == "fa3_fwd"
        else _total_2d_tflops()
    )

    def _fa3_fwd():
        return fa3(Q, K, V, causal=IS_CAUSAL)

    def _triton_fwd():
        return triton_fwd(Q, K1, K2, V1, V2, w1=w1, w2=w2)

    def _tlx_fwd_ws():
        return tlx_fwd_ws(Q, K1, K2, V1, V2, w1=w1, w2=w2)

    def _tlx_fwd_ws_pingpong():
        return tlx_fwd_ws_pingpong(Q, K1, K2, V1, V2, w1=w1, w2=w2)

    NAME_TO_FN = {
        "fa3_fwd": _fa3_fwd if HAS_FA3 else None,
        "triton_fwd": _triton_fwd,
        "tlx_fwd_ws": _tlx_fwd_ws,
        "tlx_fwd_ws_pingpong": _tlx_fwd_ws_pingpong,
    }

    ms = triton.testing.do_bench_cudagraph(NAME_TO_FN[provider])
    secs = ms / 1e3
    return total_tflops / secs


def main():
    _benchmark.run(save_path=".", show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
