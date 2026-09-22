# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure the host-side cost of the NFP8 fn-to-fnuz dispatch view on ROCm."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from typing import cast

import torch
from fbgemm_gpu.split_embedding_configs import nfp8_dtype, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    ComputeDevice,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)

T = 8
E = 100_000
D = 256
B = 1024
L = 32
WARMUP = 50
ITERS = 500
VIEW_WARMUP = 1_000
VIEW_CALLS_PER_SAMPLE = 100


def _pct(vals: Sequence[float], p: float) -> float:
    sorted_vals = sorted(vals)
    return sorted_vals[
        min(max(math.ceil(len(sorted_vals) * p) - 1, 0), len(sorted_vals) - 1)
    ]


def _sample_call_cost_ns(operation: Callable[[], object]) -> list[float]:
    for _ in range(VIEW_WARMUP):
        operation()

    samples: list[float] = []
    for _ in range(ITERS):
        start_ns = time.perf_counter_ns()
        for _ in range(VIEW_CALLS_PER_SAMPLE):
            operation()
        samples.append((time.perf_counter_ns() - start_ns) / VIEW_CALLS_PER_SAMPLE)
    return samples


def bench_view(src: torch.Tensor) -> None:
    if src.dtype is not torch.float8_e4m3fn:
        print(f"  view(dtype)                 : inactive for native {src.dtype}")
        return

    view_samples = _sample_call_cost_ns(lambda: src.view(dtype=torch.float8_e4m3fnuz))
    baseline_samples = _sample_call_cost_ns(lambda: None)
    view_p50 = statistics.median(view_samples)
    baseline_p50 = statistics.median(baseline_samples)

    print(
        f"  view(dtype) on {src.numel():>12,} real weight elems: "
        f"raw p50={view_p50:7.1f} ns  "
        f"raw p90={_pct(view_samples, 0.90):7.1f} ns  "
        f"baseline p50={baseline_p50:7.1f} ns  "
        f"adjusted p50={view_p50 - baseline_p50:7.1f} ns"
    )


@torch.inference_mode()
def bench_tbe_forward(device: torch.device) -> tuple[float, torch.Tensor]:
    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        [(E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)] * T,
        weights_precision=SparseType.NFP8,
        output_dtype=SparseType.FP32,
        device=device,
    )
    indices = torch.randint(
        0,
        E,
        (B * T * L,),
        device=device,
        dtype=torch.int64,
    )
    offsets = torch.arange(
        0,
        B * T * L + 1,
        L,
        device=device,
        dtype=torch.int64,
    )

    for _ in range(WARMUP):
        tbe(indices, offsets)
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize(device)
        start_ns = time.perf_counter_ns()
        tbe(indices, offsets)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter_ns() - start_ns)

    p50 = statistics.median(samples)
    print(
        f"  TBE NFP8 forward           : "
        f"p50={p50 / 1000.0:9.1f} us  "
        f"p90={_pct(samples, 0.90) / 1000.0:9.1f} us  "
        f"min={min(samples) / 1000.0:9.1f} us"
    )
    return p50, cast(torch.Tensor, tbe.weights_dev)


def main() -> None:
    if torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm build of PyTorch")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an available ROCm GPU")

    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise RuntimeError("This benchmark requires an available ROCm GPU")
    device = accelerator
    arch = torch.cuda.get_device_properties(device).gcnArchName
    native_dtype = nfp8_dtype(device)
    print(f"arch        = {arch}")
    print(f"nfp8_dtype  = {native_dtype}")
    print(f"fn-to-fnuz dispatch view active = {native_dtype is torch.float8_e4m3fn}")
    print(f"config      = T={T} E={E:,} D={D} B={B} L={L} iters={ITERS}\n")

    p50_fwd, weights = bench_tbe_forward(device)
    print()
    bench_view(weights)

    print(f"\n  forward p50 = {p50_fwd / 1000.0:.1f} us")
    print("  NOTE: forward does 2 relabels (dev_weights, uvm_weights);")
    print("        uvm_weights is empty here, so only 1 is a real view.")


if __name__ == "__main__":
    main()
