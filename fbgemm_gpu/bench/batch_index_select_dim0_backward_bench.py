#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Standalone benchmark for batch_index_select_dim0 backward, used to
# characterize the perf impact of the BATCH_INDEX_SELECT_DIM0_WARP_ONLY_BACKWARD
# feature gate. The gate, when on, forces every dedup run through the warp
# kernel and skips the find_long_segments + CTA path. This benchmark sweeps
# input shapes and index collision rates so the cost of that bypass can be
# quantified across realistic workloads.
#
# Two entry points share the same sweep logic:
#   1. CLI (click): for ad-hoc local runs
#        buck2 run @mode/opt //deeplearning/fbgemm/fbgemm_gpu/bench:batch_index_select_dim0_backward_bench
#   2. unittest.TestCase: lets python_unittest_remote_gpu run it on a
#      specific subplatform (A100 / H100 / B200) with the gate set via env.
#        buck2 test //deeplearning/fbgemm/fbgemm_gpu/bench:batch_index_select_dim0_backward_bench_<hw>
#
# Compare gate off vs on by diffing the tables produced by the
# `_warp_only_<hw>` variant against the `_<hw>` variant.

import logging
import os
import time
import unittest

import click
import fbgemm_gpu.sparse_ops  # noqa: F401, E402
import torch

logger: logging.Logger = logging.getLogger(__name__)


# Default sweep used by both the CLI and the unittest entrypoint.
# (num_tables, rows_per_table, embedding_dim, indices_per_table, hot_frac)
#
# hot_frac controls what fraction of rows are "hot" (eligible to be indexed)
# *within each table independently*. Each table gets its own random hot-pool.
# 1.0 = all rows are hot, uniform random (mostly rc=1, gate has minimal effect).
# 0.10 = only 10% of rows are hot, heavy concentration (lots of rc>=32, gate
# branch dominates).
#
# total_queries = num_tables * indices_per_table
_DEFAULT_SWEEP: list[tuple[int, int, int, int, float]] = [
    # --- Small-scale: characterize gate overhead ---
    # hot_frac sweep: 1.0 (uniform), 0.50 (moderate), 0.10 (skewed)
    (32, 1024, 64, 1024, 1.0),
    (32, 1024, 64, 1024, 0.50),
    (32, 1024, 64, 1024, 0.10),
    (256, 1024, 64, 1024, 1.0),
    (256, 1024, 64, 1024, 0.50),
    (256, 1024, 64, 1024, 0.10),
    (32, 16384, 256, 8192, 1.0),
    (32, 16384, 256, 8192, 0.50),
    (32, 16384, 256, 8192, 0.10),
    # --- Larger embedding dims ---
    (32, 1024, 128, 1024, 1.0),
    (32, 1024, 128, 1024, 0.50),
    (32, 1024, 128, 1024, 0.10),
    (32, 1024, 256, 1024, 1.0),
    (32, 1024, 256, 1024, 0.50),
    (32, 1024, 256, 1024, 0.10),
    (256, 1024, 128, 1024, 1.0),
    (256, 1024, 128, 1024, 0.50),
    (256, 1024, 128, 1024, 0.10),
    (256, 1024, 256, 1024, 1.0),
    (256, 1024, 256, 1024, 0.50),
    (256, 1024, 256, 1024, 0.10),
    # --- Production-scale: ~21M total queries ---
    # 256 tables × 82,404 indices = 21,095,424 total
    # rows_per_table kept at 16384 to avoid GPU OOM on smaller GPUs (A100 MIG)
    (256, 16384, 64, 82404, 1.0),
    (256, 16384, 64, 82404, 0.50),
    (256, 16384, 64, 82404, 0.10),
    (256, 16384, 128, 82404, 1.0),
    (256, 16384, 128, 82404, 0.50),
    (256, 16384, 128, 82404, 0.10),
    (256, 16384, 256, 82404, 1.0),
    (256, 16384, 256, 82404, 0.50),
    (256, 16384, 256, 82404, 0.10),
]


def _build_inputs(
    num_tables: int,
    rows_per_table: int,
    embedding_dim: int,
    indices_per_table: int,
    hot_frac: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[int]]:
    input_rows = [rows_per_table] * num_tables
    input_columns = [embedding_dim] * num_tables
    input_num_indices = [indices_per_table] * num_tables

    indices_list: list[torch.Tensor] = []
    hot_count = max(1, int(rows_per_table * hot_frac))
    for _ in range(num_tables):
        hot_pool = torch.randint(
            low=0,
            high=rows_per_table,
            size=(hot_count,),
            dtype=torch.long,
            device=device,
        )
        pick = torch.randint(
            low=0,
            high=hot_count,
            size=(indices_per_table,),
            dtype=torch.long,
            device=device,
        )
        indices_list.append(hot_pool[pick])
    concat_indices = torch.cat(indices_list)

    inputs = [
        torch.rand(rows_per_table, embedding_dim, dtype=torch.float, device=device)
        for _ in range(num_tables)
    ]
    concat_inputs = torch.cat([t.flatten() for t in inputs])
    concat_inputs.requires_grad = True
    return concat_inputs, concat_indices, input_num_indices, input_rows, input_columns


def _time_one(
    concat_inputs: torch.Tensor,
    concat_indices: torch.Tensor,
    input_num_indices: list[int],
    input_rows: list[int],
    input_columns: list[int],
    warmup: int,
    iters: int,
) -> float:
    output = torch.ops.fbgemm.batch_index_select_dim0(
        concat_inputs,
        concat_indices,
        input_num_indices,
        input_rows,
        input_columns,
        False,
    )
    grad = torch.rand_like(output)

    for _ in range(warmup):
        concat_inputs.grad = None
        output = torch.ops.fbgemm.batch_index_select_dim0(
            concat_inputs,
            concat_indices,
            input_num_indices,
            input_rows,
            input_columns,
            False,
        )
        output.backward(grad)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        concat_inputs.grad = None
        output = torch.ops.fbgemm.batch_index_select_dim0(
            concat_inputs,
            concat_indices,
            input_num_indices,
            input_rows,
            input_columns,
            False,
        )
        output.backward(grad)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def run_sweep(
    sweep: list[tuple[int, int, int, int, float]],
    warmup: int = 5,
    iters: int = 50,
    seed: int = 0,
) -> list[tuple[tuple[int, int, int, int, float], float]]:
    # lint-fixme: TorchDeviceCuda, TorchFunctionCallCudaDevice
    # CUDA specifically required: benchmarking GPU-only FBGEMM sparse op
    device = torch.device("cuda")
    torch.manual_seed(seed)
    results: list[tuple[tuple[int, int, int, int, float], float]] = []
    for cfg in sweep:
        num_tables, rows_per_table, embedding_dim, indices_per_table, hot_frac = cfg
        (
            concat_inputs,
            concat_indices,
            input_num_indices,
            input_rows,
            input_columns,
        ) = _build_inputs(
            num_tables,
            rows_per_table,
            embedding_dim,
            indices_per_table,
            hot_frac,
            device,
        )
        elapsed_ms = _time_one(
            concat_inputs,
            concat_indices,
            input_num_indices,
            input_rows,
            input_columns,
            warmup,
            iters,
        )
        results.append((cfg, elapsed_ms))
    return results


def format_results(
    results: list[tuple[tuple[int, int, int, int, float], float]],
    warmup: int,
    iters: int,
) -> str:
    gate_on = os.environ.get("FBGEMM_BATCH_INDEX_SELECT_DIM0_WARP_ONLY_BACKWARD") == "1"
    gate_label = "ON" if gate_on else "OFF"
    device_name = torch.cuda.get_device_name(0)
    header = (
        "\n=== batch_index_select_dim0 backward perf sweep ==="
        f"\nDevice: {device_name}"
        f"\nGate (BATCH_INDEX_SELECT_DIM0_WARP_ONLY_BACKWARD): {gate_label}"
        f"\nWarmup={warmup} iters={iters}"
        "\n"
        f"\n{'num_tables':>10} {'rows/tbl':>8} {'emb_dim':>7} "
        f"{'idx/tbl':>7} {'hot_frac':>8} {'total_queries':>13} {'fwd+bwd_ms':>11}"
    )
    lines = [header]
    for (
        num_tables,
        rows_per_table,
        embedding_dim,
        indices_per_table,
        hot_frac,
    ), ms in results:
        total_queries = num_tables * indices_per_table
        lines.append(
            f"{num_tables:>10} {rows_per_table:>8} {embedding_dim:>7} "
            f"{indices_per_table:>7} {hot_frac:>8.2f} {total_queries:>13} {ms:>11.4f}"
        )
    return "\n".join(lines)


@click.command()
@click.option("--warmup", default=5, show_default=True, type=int)
@click.option("--iters", default=50, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
def cli(warmup: int, iters: int, seed: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_sweep(_DEFAULT_SWEEP, warmup=warmup, iters=iters, seed=seed)
    print(format_results(results, warmup, iters))


class BatchIndexSelectDim0BackwardBenchTest(unittest.TestCase):
    # Bench-as-test wrapper. Runs the same sweep as the CLI under the
    # python_unittest_remote_gpu execution model so we can pick subplatform
    # (A100 / H100 / B200) and gate state via BUCK target env. Always passes
    # unless the kernel itself crashes.
    def test_run_sweep(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU required")
        warmup, iters = 5, 50
        results = run_sweep(_DEFAULT_SWEEP, warmup=warmup, iters=iters)
        print(format_results(results, warmup, iters))


if __name__ == "__main__":
    cli()
