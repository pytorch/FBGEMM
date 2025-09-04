# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

from simplicial.utils import get_simplicial_tensor_core_tflops


def _computed_tflops_tiling_sequence(block_m, block_kv, n, w1, w2, d):
    """
    Args:
        block_m (int): Block size M
        block_kv (int): Block size KV
        n (int): Total sequence length
        w1 (int): Window size parameter 1
        w2 (int): Window size parameter 2
        d (int): Depth/dimension parameter
    """
    total_computed_tflops = 0

    # Process each block in the sequence
    for i in range(0, n, block_m):
        k1_start_idx = max(i - w1 + 1, 0)
        k1_end_idx = min(i + block_m, n)

        k2_start_idx = max(i - w2 + 1, 0)
        k2_end_idx = min(i + block_m, n)

        num_w1_loops = k1_end_idx - k1_start_idx
        num_w2_loops = math.ceil((k2_end_idx - k2_start_idx) / block_kv)
        total_loops = num_w1_loops * num_w2_loops

        # Calculate TFLOPs for this block
        qk_gemm_tflops = block_m * block_kv * d * 2
        pv_gemm_tflops = block_m * block_kv * d * 2
        computed_tflops = (qk_gemm_tflops + pv_gemm_tflops) * total_loops

        # Accumulate totals
        total_computed_tflops += computed_tflops

    return total_computed_tflops / 1e12


def _computed_tflops_tiling_heads(block_m, block_kv, n, w1, w2, d):
    """
    Args:
        block_m (int): Block size M
        block_kv (int): Block size KV
        n (int): Total sequence length
        w1 (int): Window size parameter 1
        w2 (int): Window size parameter 2
        d (int): Depth/dimension parameter
    """
    total_computed_tflops = 0

    # Process each block in the sequence
    for i in range(0, n):
        # K1 (Key1) block parameters
        k1_start_idx = max(i - w1 + 1, 0)
        k1_end_idx = i + 1

        k2_start_idx = max(i - w2 + 1, 0)
        k2_end_idx = i + 1

        num_w1_loops = k1_end_idx - k1_start_idx
        num_w2_loops = math.ceil((k2_end_idx - k2_start_idx) / block_kv)

        total_loops = num_w1_loops * num_w2_loops

        # Calculate TFLOPs for this block
        qk_gemm_tflops = block_m * block_kv * d * 2
        pv_gemm_tflops = block_m * block_kv * d * 2
        computed_tflops = (qk_gemm_tflops + pv_gemm_tflops) * total_loops

        # Accumulate totals
        total_computed_tflops += computed_tflops

    return total_computed_tflops / 1e12


def quick_calculate(block_m=64, block_kv=128, n=8192, w1=32, w2=512, d=128):
    hq = block_m

    # Use centralized utility function (assumes num_kv_heads=1 for analysis)
    total_valid_tflops = get_simplicial_tensor_core_tflops(1, n, hq, 1, d, w1, w2)

    computed_tflops_tiling_sequence = (
        _computed_tflops_tiling_sequence(block_m, block_kv, n, w1, w2, d)
    ) * hq
    computed_tflops_tiling_heads = _computed_tflops_tiling_heads(
        block_m, block_kv, n, w1, w2, d
    )
    waste_pct = 1 - total_valid_tflops / computed_tflops_tiling_sequence
    waste_pct_heads = 1 - total_valid_tflops / computed_tflops_tiling_heads

    print("-" * 30)
    print(f"Parameters: M={block_m}, KV={block_kv}, N={n}, W1={w1}, W2={w2}, D={d}")
    print("Tiling Sequence:")
    print(f"Efficiency: {(1 - waste_pct) * 100:.2f}%")
    print(f"Waste: {waste_pct * 100:.2f}%")
    print("Tiling Heads:")
    print(f"Efficiency: {(1 - waste_pct_heads) * 100:.2f}%")
    print(f"Waste: {waste_pct_heads * 100:.2f}%")
    print("\n" * 2)

    return waste_pct, waste_pct_heads


"""
python3 swa_wasted_tflops.py

Simplicial Attention with 2D Sliding Window Wasted TFLOPs Analysis:
------------------------------
Parameters: M=64, KV=128, N=8192, W1=32, W2=512, D=128
Tiling Sequence:
Efficiency: 26.80%
Waste: 73.20%
Tiling Heads:
Efficiency: 98.65%
Waste: 1.35%

------------------------------
Parameters: M=128, KV=128, N=8192, W1=32, W2=512, D=128
Tiling Sequence:
Efficiency: 16.01%
Waste: 83.99%
Tiling Heads:
Efficiency: 98.65%
Waste: 1.35%
"""


if __name__ == "__main__":
    print("Simplicial Attention with 2D Sliding Window Wasted TFLOPs Analysis:")
    quick_calculate(block_m=64, block_kv=128, n=8192, w1=32, w2=512, d=128)
    quick_calculate(block_m=128, block_kv=128, n=8192, w1=32, w2=512, d=128)
