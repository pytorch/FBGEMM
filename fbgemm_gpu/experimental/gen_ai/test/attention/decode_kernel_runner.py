#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone script to run the decode kernel for Blackwell FMHA.

This script runs the decode (generation) kernel for attention, which is used
during inference when generating tokens one at a time (seqlen_q = 1).

Usage:
    buck run fbcode//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:decode_kernel_entry -- \
        --batch_size 2 --seqlen_k 128 --q_heads 8 --head_dim 128
"""

import argparse

import torch
from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha.cutlass_blackwell_fmha_interface import (
    _cutlass_blackwell_fmha_gen,
    GenKernelType,
)


def run_decode_kernel(
    batch_size: int,
    seqlen_k: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> None:
    """Run the decode kernel with specified parameters."""
    device = torch.accelerator.current_accelerator()
    assert device is not None, "No GPU device available"

    # Decode kernel always has seqlen_q = 1 (generating one token at a time)
    seqlen_q = 1

    print(f"Running decode kernel with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length (K/V): {seqlen_k}")
    print(f"  Query heads: {q_heads}")
    print(f"  KV heads: {kv_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Data type: {dtype}")
    print(f"  Device: {device}")

    # Generate random Q, K, V tensors
    q = torch.randn(
        batch_size,
        seqlen_q,
        q_heads,
        head_dim,
        dtype=torch.float if dtype == torch.float8_e4m3fn else dtype,
        device=device,
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        kv_heads,
        head_dim,
        dtype=torch.float if dtype == torch.float8_e4m3fn else dtype,
        device=device,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        kv_heads,
        head_dim,
        dtype=torch.float if dtype == torch.float8_e4m3fn else dtype,
        device=device,
    )

    # Convert to FP8 if needed
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)

    # Make tensors contiguous as required by _cutlass_blackwell_fmha_gen
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Initialize seqlen_kv for generation phase
    seqlen_kv = torch.full(
        (batch_size,),
        seqlen_k,
        dtype=torch.int32,
        device=device,
    )

    # Create batch_idx tensor
    batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)

    print("\nRunning decode kernel (_cutlass_blackwell_fmha_gen)...")
    print(f"  Kernel Type: GenKernelType.UMMA_I")

    # Run the decode kernel directly
    out = _cutlass_blackwell_fmha_gen(
        q,
        k,
        v,
        seqlen_kv,
        batch_idx,
        kernel_type=GenKernelType.UMMA_I,
    )

    print(f"Decode kernel completed successfully!")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output device: {out.device}")

    # Basic sanity checks
    assert out.shape == (batch_size, seqlen_q, q_heads, head_dim)
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"

    print("\nAll sanity checks passed!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the decode kernel for Blackwell FMHA"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )
    parser.add_argument(
        "--seqlen_k",
        type=int,
        default=128,
        help="Sequence length for K/V (default: 128)",
    )
    parser.add_argument(
        "--q_heads",
        type=int,
        default=8,
        help="Number of query heads (default: 8)",
    )
    parser.add_argument(
        "--kv_heads",
        type=int,
        default=1,
        help="Number of KV heads, use 1 for MQA (default: 1)",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
        help="Head dimension (default: 128)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp8",
        choices=["fp8", "fp16", "bf16"],
        help="Data type (default: fp8)",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "fp8": torch.float8_e4m3fn,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available")
        return

    compute_capability = torch.cuda.get_device_capability("cuda")
    if compute_capability < (10, 0):
        print(
            f"ERROR: Decode kernel requires SM100+ (Blackwell), found SM{compute_capability[0]}{compute_capability[1]}"
        )
        return

    # Run the decode kernel
    run_decode_kernel(
        batch_size=args.batch_size,
        seqlen_k=args.seqlen_k,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
