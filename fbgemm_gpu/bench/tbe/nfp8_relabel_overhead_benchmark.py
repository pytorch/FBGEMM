# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure the host-side cost of the NFP8 fn->fnuz boundary relabel on ROCm.

Two measurements:
  1. Isolated cost of Tensor.view(dtype) at TBE weight-buffer scale. This is an
     UPPER BOUND on the C++ relabel_nfp8_for_dispatch() cost, since the Python
     call additionally pays pybind/dispatch overhead the C++ path does not.
  2. End-to-end TBE NFP8 forward wall time, for context. Run this file against
     both the relabel build and a forced-fnuz-label build to A/B the delta.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence

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


def _pct(vals: Sequence[float], p: float) -> float:
    s = sorted(vals)
    return s[min(int(len(s) * p), len(s) - 1)]


def bench_view(numel: int) -> None:
    src = torch.empty(
        numel, dtype=torch.float8_e4m3fn, device=torch.accelerator.current_accelerator()
    )
    for _ in range(1000):
        src.view(dtype=torch.float8_e4m3fnuz)

    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        for _ in range(100):
            src.view(dtype=torch.float8_e4m3fnuz)
        samples.append((time.perf_counter_ns() - t0) / 100.0)

    print(
        f"  view(dtype) on {numel:>12,} elems: "
        f"p50={statistics.median(samples):7.1f} ns  "
        f"p90={_pct(samples, 0.90):7.1f} ns  "
        f"min={min(samples):7.1f} ns"
    )


def bench_tbe_forward() -> tuple[float, int, int]:
    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        [(E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)] * T,
        weights_precision=SparseType.NFP8,
        output_dtype=SparseType.FP32,
        device=torch.cuda.current_device(),
    )
    weights = tbe.split_embedding_weights()[0]
    total_numel = sum(w.numel() for w in tbe.split_embedding_weights())

    indices = torch.randint(
        0,
        E,
        (B * T * L,),
        device=torch.accelerator.current_accelerator(),
        dtype=torch.int64,
    )
    offsets = torch.arange(
        0,
        B * T * L + 1,
        L,
        device=torch.accelerator.current_accelerator(),
        dtype=torch.int64,
    )

    for _ in range(WARMUP):
        tbe(indices, offsets)
    torch.cuda.synchronize()

    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        tbe(indices, offsets)
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - t0)

    p50 = statistics.median(samples)
    print(
        f"  TBE NFP8 forward           : "
        f"p50={p50 / 1000.0:9.1f} us  "
        f"p90={_pct(samples, 0.90) / 1000.0:9.1f} us  "
        f"min={min(samples) / 1000.0:9.1f} us"
    )
    return p50, weights.numel(), total_numel


def main() -> None:
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    print(f"arch        = {arch}")
    print(f"nfp8_dtype  = {nfp8_dtype()}")
    print(f"relabel active on this build = {nfp8_dtype() is torch.float8_e4m3fn}")
    print(f"config      = T={T} E={E:,} D={D} B={B} L={L} iters={ITERS}\n")

    p50_fwd, per_table, total = bench_tbe_forward()
    print()
    bench_view(per_table)
    bench_view(total)

    print(f"\n  forward p50 = {p50_fwd / 1000.0:.1f} us")
    print("  NOTE: forward does 2 relabels (dev_weights, uvm_weights);")
    print("        uvm_weights is empty here, so only 1 is a real view.")


if __name__ == "__main__":
    main()
