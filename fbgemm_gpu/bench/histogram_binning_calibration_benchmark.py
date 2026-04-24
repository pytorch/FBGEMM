# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Performance benchmark for histogram_binning_calibration operators.

This benchmark covers 3 FBGEMM histogram-binning calibration ops:
  1. histogram_binning_calibration (hbc) — 1 CUDA kernel
  2. histogram_binning_calibration_by_feature (hbc_by_feature) — 2 CUDA kernels
  3. generic_histogram_binning_calibration_by_feature (generic) — 2 CUDA kernels

Usage:
    buck2 run //deeplearning/fbgemm/fbgemm_gpu:histogram_binning_calibration_benchmark

Hardened changes (matching tritonbench port —
``HistogramBinningCalibrationBenchmark``):
  - Added --device flag to select cpu, cuda, or both (default: both for
    backward compatibility).
  - Added --export-trace flag for Kineto trace export.
  - Added input validation (num_logits > 0, num_bins > 0, CUDA grid limits).
  - Fixed bin arrays: old code initialized bin_num_examples to 0 with
    bin_ctr_in_use_after=0, so the calibrated branch was never taken.  Now
    bin arrays are filled with small positive counts.
  - Refactored into reusable functions with clear per-variant timing using
    ``benchmark_torch_function`` from bench_utils.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, TypeAlias

import click
import torch
from torch import Tensor

_HbcFn: TypeAlias = Callable[[Tensor], tuple[Tensor, Tensor]]

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    open_source_fbgemm: bool = True
except Exception:
    open_source_fbgemm: bool = False
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

if open_source_fbgemm:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function


# ──────────────────────────────────────────────────────────────────────
# Constants (matching tritonbench port)
# ──────────────────────────────────────────────────────────────────────

_NUM_SEGMENTS: int = 42
_POSITIVE_WEIGHT: float = 0.4
_LOWER_BOUND: float = 0.0
_UPPER_BOUND: float = 1.0
_BIN_CTR_IN_USE_AFTER: int = 0
_BIN_CTR_WEIGHT_VALUE: float = 0.9995


# ──────────────────────────────────────────────────────────────────────
# Input validation (backported from tritonbench
# HistogramBinningCalibrationBenchmark._validate_inputs)
# ──────────────────────────────────────────────────────────────────────


def _validate_inputs(num_logits: int, num_bins: int, num_segments: int) -> None:
    """Validate inputs before allocation.

    Checks:
      1. num_logits, num_bins, and num_segments are positive.
      2. CUDA grid dimension limit: ceil(num_logits / 1024) < 2^31.
      3. by_feature bin array size fits in int64.
    """
    if num_logits < 1:
        raise ValueError(f"num_logits must be >= 1, got {num_logits}.")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
    if num_segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {num_segments}.")

    kMaxThreads = 1024
    num_blocks = math.ceil(num_logits / kMaxThreads)
    if num_blocks >= 2**31:
        raise ValueError(
            f"num_logits = {num_logits} requires {num_blocks} CUDA blocks, "
            f"which exceeds the CUDA grid limit (2^31 - 1). "
            f"Reduce num_logits."
        )

    by_feature_size = num_bins * (num_segments + 1)
    if by_feature_size >= 2**63:
        raise ValueError(
            f"by_feature bin array size = num_bins * (num_segments + 1) = "
            f"{num_bins} * {num_segments + 1} = {by_feature_size} "
            f"exceeds int64 max. Reduce num_bins or num_segments."
        )


# ──────────────────────────────────────────────────────────────────────
# Benchmark implementation
# ──────────────────────────────────────────────────────────────────────


def _benchmark_all_variants(
    num_logits: int,
    num_bins: int,
    data_type: torch.dtype,
    device: str,
    iters: int,
    num_warmups: int,
    export_trace: bool,
) -> None:
    """Benchmark all 3 HBC variants for a given config."""

    num_segments = _NUM_SEGMENTS

    # Validate inputs (backported from tritonbench port)
    _validate_inputs(num_logits, num_bins, num_segments)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU benchmark")
        return

    # ── Input generation ──
    input_data = torch.rand(num_logits, dtype=torch.float, device=device).to(data_type)

    # Fix: old benchmark used zeros for bin arrays, which meant the calibrated
    # branch was never exercised (bin_ctr_in_use_after=0 but examples=0 so
    # 0 > 0 is false).  Now fill with small positive counts so the calibration
    # path IS tested.  (Matching tritonbench port.)
    bin_num_examples: Tensor = torch.rand(
        num_bins, dtype=torch.float64, device=device
    ).add_(1.0)
    bin_num_positives: Tensor = torch.rand(num_bins, dtype=torch.float64, device=device)

    segment_lengths: Tensor = torch.randint(
        0, 2, (num_logits,), dtype=torch.int64, device=device
    )
    num_values: int = int(torch.sum(segment_lengths).item())
    segment_values: Tensor = torch.randint(
        0, num_segments, (num_values,), dtype=torch.int64, device=device
    )

    w: float = (_UPPER_BOUND - _LOWER_BOUND) / num_bins
    bin_boundaries: Tensor = torch.arange(
        _LOWER_BOUND + w, _UPPER_BOUND - w / 2, w, dtype=torch.float64, device=device
    )

    by_feature_size = num_bins * (num_segments + 1)
    by_feature_bin_num_examples: Tensor = torch.rand(
        by_feature_size, dtype=torch.float64, device=device
    ).add_(1.0)
    by_feature_bin_num_positives: Tensor = torch.rand(
        by_feature_size, dtype=torch.float64, device=device
    )

    # ── Variant closures ──
    def fbgemm_hbc(logit: Tensor) -> tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.histogram_binning_calibration(
            logit,
            bin_num_examples,
            bin_num_positives,
            _POSITIVE_WEIGHT,
            _LOWER_BOUND,
            _UPPER_BOUND,
            _BIN_CTR_IN_USE_AFTER,
            _BIN_CTR_WEIGHT_VALUE,
        )

    _num_segments: int = num_segments

    def fbgemm_hbc_by_feature(logit: Tensor) -> tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.histogram_binning_calibration_by_feature(
            logit,
            segment_values,
            segment_lengths,
            _num_segments,
            by_feature_bin_num_examples,
            by_feature_bin_num_positives,
            num_bins,
            _POSITIVE_WEIGHT,
            _LOWER_BOUND,
            _UPPER_BOUND,
            _BIN_CTR_IN_USE_AFTER,
            _BIN_CTR_WEIGHT_VALUE,
        )

    def fbgemm_generic_hbc_by_feature(logit: Tensor) -> tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            logit,
            segment_values,
            segment_lengths,
            _num_segments,
            by_feature_bin_num_examples,
            by_feature_bin_num_positives,
            bin_boundaries,
            _POSITIVE_WEIGHT,
            _BIN_CTR_IN_USE_AFTER,
            _BIN_CTR_WEIGHT_VALUE,
        )

    variants: list[tuple[str, _HbcFn]] = [
        ("hbc", fbgemm_hbc),
        ("hbc_by_feature", fbgemm_hbc_by_feature),
        ("generic_hbc_by_feature", fbgemm_generic_hbc_by_feature),
    ]

    print(f"\n{'=' * 70}")
    print(
        f"Config: num_logits={num_logits}, num_bins={num_bins}, "
        f"dtype={data_type}, device={device}"
    )
    print(f"{'=' * 70}")

    for name, fn in variants:
        elapsed_time, _ = benchmark_torch_function(
            fn,
            (input_data,),
            iters=iters,
            num_warmups=num_warmups,
            device=device,
            name=f"{name}_{device}_{data_type}",
        )
        print(f"  {name:40s} {elapsed_time * 1.0e6:.0f} us")

    print(f"{'=' * 70}\n")

    # Export Kineto trace if requested
    if export_trace:
        trace_name = f"hbc_N{num_logits}_B{num_bins}_{data_type}"
        _export_kineto_trace(input_data, variants, trace_name, device)


def _export_kineto_trace(
    input_data: Tensor,
    variants: list[tuple[str, _HbcFn]],
    trace_name: str,
    device: str,
) -> None:
    """Export a Kineto trace for each variant.

    Runs each variant under torch.profiler and saves to JSON files
    that can be loaded in chrome://tracing or Perfetto.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]  # pyre-fixme[16]
    if device != "cpu" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)  # pyre-fixme[16]

    for impl_name, impl_fn in variants:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
        ) as prof:
            for _ in range(3):
                impl_fn(input_data)
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()

        filename = f"{trace_name}_{impl_name}.json"
        prof.export_chrome_trace(filename)
        print(f"Exported Kineto trace: {filename}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--iters", default=100, help="Number of timed iterations.")
@click.option(
    "--warmup-runs",
    default=2,
    help="Number of warmup iterations (passed to bench_utils).",
)
@click.option(
    "--manual-seed/--skip-manual-seed",
    default=False,
    help="Use manual seed for reproduction.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "both"]),
    default="both",
    help=(
        "Device to benchmark on. 'both' (default) matches original behavior: "
        "run CPU then GPU if available."
    ),
)
@click.option(
    "--num-logits",
    type=click.IntRange(min=1),
    default=5000,
    help="Number of input logit values (also used as num_bins).",
)
@click.option(
    "--export-trace/--no-export-trace",
    default=False,
    help="Export Kineto profiler traces for each variant.",
)
def cli(
    iters: int,
    warmup_runs: int,
    manual_seed: bool,
    device: str,
    num_logits: int,
    export_trace: bool,
) -> None:
    """Benchmark FBGEMM histogram-binning calibration operators."""
    # set manual seed for reproducibility
    if manual_seed:
        torch.manual_seed(42)

    data_types = [torch.half, torch.float]
    num_bins = num_logits

    # Determine device list
    if device == "both":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
    else:
        devices = [device]

    print("\n" + "=" * 70)
    print("HISTOGRAM BINNING CALIBRATION BENCHMARK")
    print("=" * 70)
    print(f"Devices: {devices}")
    print(f"Iterations: {iters}")
    print(f"num_logits: {num_logits}, num_bins: {num_bins}")
    print("=" * 70 + "\n")

    for dev in devices:
        for data_type in data_types:
            _benchmark_all_variants(
                num_logits=num_logits,
                num_bins=num_bins,
                data_type=data_type,
                device=dev,
                iters=iters,
                num_warmups=warmup_runs,
                export_trace=export_trace,
            )


if __name__ == "__main__":
    cli()
