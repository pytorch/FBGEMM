#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import json
import logging
import math
import os
import random
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import click
import numpy as np

import torch

from fbgemm_gpu.split_embedding_configs import SparseType

from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import (
    bench_warmup,
    benchmark_cpu_requests,
    benchmark_pipelined_requests,
    benchmark_requests,
    benchmark_requests_refer,
    BenchmarkReporter,
    fill_random_scale_bias,
)
from fbgemm_gpu.tbe.utils import generate_requests, round_up, TBERequest
from torch.profiler import profile

logging.basicConfig(level=logging.DEBUG)


def kineto_trace_profiler(p: profile, trace_info: tuple[str, str, str, str]) -> float:
    phase, trace_url, tbe_type, kern_name = trace_info
    p.export_chrome_trace(
        trace_url.format(tbe_type=tbe_type, phase=phase, ospid=os.getpid())
    )
    kernel_time = 0
    for event in p.key_averages():
        # Sum the total time of forward kernel runs
        if kern_name in event.key:
            kernel_time += event.device_time
    assert kernel_time > 0
    print(f"Total CUDA time: {kernel_time:.2f} ")
    return kernel_time


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option("--managed", default="device")
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--index-remapping", is_flag=True, default=False)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
@click.option("--pooling", type=str, default="sum")
def nbit_cpu(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
    warmup_runs: int,
    managed: str,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    row_wise: bool,
    weighted: bool,
    index_remapping: bool,
    requests_data_file: Optional[str],
    tables: Optional[str],
    output_dtype: SparseType,
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
    pooling: str,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    if mixed:
        Ds = [
            # int4 table batched emb op can only handle mixed D where D is multiple of 8
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 8)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    if pooling is None or pooling == "sum":
        pooling = "sum"
        pooling_mode = PoolingMode.SUM
        do_pooling = True
    elif pooling == "mean":
        pooling_mode = PoolingMode.MEAN
        do_pooling = True
    else:  # "none"
        pooling_mode = PoolingMode.NONE
        do_pooling = False

    emb = IntNBitTableBatchedEmbeddingBagsCodegen(
        [("", E, d, weights_precision, EmbeddingLocation.HOST) for d in Ds],
        device="cpu",
        index_remapping=[torch.arange(E) for _ in Ds] if index_remapping else None,
        output_dtype=output_dtype,
        pooling_mode=pooling_mode,
        fp8_exponent_bits=fp8_exponent_bits,
        fp8_exponent_bias=fp8_exponent_bias,
    ).cpu()
    emb.fill_random_weights()
    fill_random_scale_bias(emb, T, weights_precision)

    nparams_byte = sum(w.numel() for (w, _) in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    if do_pooling:
        read_write_bytes = (
            output_size_multiplier * B * T * D + param_size_multiplier * B * T * L * D
        )
    else:
        read_write_bytes = (
            output_size_multiplier * B * T * L * D
            + param_size_multiplier * B * T * L * D
        )

    logging.info(
        f"{weights_precision} Embedding tables: {E * T} rows, {nparams_byte / param_size_multiplier / 1.0e9: .2f} GParam, "
        f"{nparams_byte / 1.0e9: .2f} GB"  # IntN TBE use byte for storage
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        # pyre-fixme[58]: `*` is not supported for operand types `int` and
        #  `Union[np.floating[typing.Any], int]`.
        f"{B * T * L * D * param_size_multiplier / 1.0e9: .2f} GB"
    )

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weighted=weighted,
        requests_data_file=requests_data_file,
        tables=tables,
        use_cpu=True,
    )
    requests = [
        TBERequest(
            req.indices.cpu().int(),
            req.offsets.cpu().int(),
            req.per_sample_weights.cpu() if req.per_sample_weights else None,
        )
        for req in requests
    ]

    time_per_iter = benchmark_cpu_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb.forward(
            indices,
            offsets,
            per_sample_weights,
        ),
        num_warmups=warmup_runs,
    )

    logging.info(
        f"{weights_precision} Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--managed", default="device")
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--pooling", type=str, default="sum")
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.NONE.value)
@click.option("--pruning-ratio", type=float, default=None)
@click.option("--pruning-hash-load-factor", default=0.75)
@click.option("--use-array-for-index-remapping", is_flag=True, default=True)
@click.option("--check-median", is_flag=True, default=True)
@click.option("--iters", default=100)
@click.option("--runs-of-iters", default=5)
@click.option("--warmup-runs", default=2)
@click.option("--warmup-ms", type=int, default=None)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--report-aibench", is_flag=True)
@click.option("--run-reference", is_flag=True, default=False)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
@click.option("--export-trace", is_flag=True, default=False)
@click.option(
    "--trace-url",
    type=str,
    default="{tbe_type}_tbe_{phase}_trace_{ospid}.json",
)
@click.option(
    "--warmup-runs",
    default=2,
    help="Number of warmup runs. Ignored if --warmup-ms is set.",
)
@click.option(
    "--warmup-ms",
    type=int,
    default=None,
    help="Warmup duration in milliseconds. Disables the --run-nums option.",
)
def nbit_device(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    managed: str,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    weighted: bool,
    pooling: str,
    bounds_check_mode: int,
    pruning_ratio: Optional[float],
    pruning_hash_load_factor: float,
    use_array_for_index_remapping: bool,
    check_median: bool,
    iters: int,
    runs_of_iters: int,
    output_dtype: SparseType,
    report_aibench: bool,
    run_reference: bool,
    requests_data_file: Optional[str],
    tables: Optional[str],
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
    export_trace: bool,
    trace_url: str,
    warmup_runs: int,
    warmup_ms: Optional[int],
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    reporter = BenchmarkReporter(report_aibench)

    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    original_E = E
    T = num_tables
    index_remapping = None
    if mixed:
        # int4 table batched emb op can only handle mixed D where D is multiple of 8
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 8)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    mem_for_pruning = 0
    if pruning_ratio:
        assert pruning_ratio < 1 and pruning_ratio >= 0
        E = math.ceil(E * (1.0 - pruning_ratio))
        index_remapping = []
        for _ in range(T):
            mapping = torch.tensor([-1] * original_E, dtype=torch.int32)
            selected_indices = random.sample(range(original_E), E)
            for i, idx in enumerate(selected_indices):
                mapping[idx] = i
            index_remapping.append(mapping)
            if use_array_for_index_remapping:
                mem_for_pruning += mapping.numel() * 4
            else:
                mem_for_pruning += E / pruning_hash_load_factor * 2 * 4

    if managed == "device":
        managed_option = EmbeddingLocation.DEVICE
    else:
        managed_option = EmbeddingLocation.MANAGED

    if pooling is None or pooling == "sum":
        pooling = "sum"
        pooling_mode = PoolingMode.SUM
        do_pooling = True
    elif pooling == "mean":
        pooling_mode = PoolingMode.MEAN
        do_pooling = True
    else:  # "none"
        pooling_mode = PoolingMode.NONE
        do_pooling = False

    emb = IntNBitTableBatchedEmbeddingBagsCodegen(
        [("", E, d, weights_precision, managed_option) for d in Ds],
        bounds_check_mode=BoundsCheckMode(bounds_check_mode),
        index_remapping=index_remapping,
        pruning_hash_load_factor=pruning_hash_load_factor,
        use_array_for_index_remapping=use_array_for_index_remapping,
        output_dtype=output_dtype,
        pooling_mode=pooling_mode,
        fp8_exponent_bits=fp8_exponent_bits,
        fp8_exponent_bias=fp8_exponent_bias,
    ).cuda()
    emb.fill_random_weights()
    fill_random_scale_bias(emb, T, weights_precision)

    nparams_byte = sum(w.numel() for (w, _) in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    if do_pooling:
        read_write_bytes = (
            output_size_multiplier * B * T * D + param_size_multiplier * B * T * L * D
        )
    else:
        read_write_bytes = (
            output_size_multiplier * B * T * L * D
            + param_size_multiplier * B * T * L * D
        )
    logging.info(
        f"{weights_precision} Embedding tables: {E * T} rows, {nparams_byte / param_size_multiplier / 1.0e9: .2f} GParam, "
        f"{nparams_byte / 1.0e9: .2f} GB"  # IntN TBE use byte for storage
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        # pyre-fixme[58]: `*` is not supported for operand types `int` and
        #  `Union[np.floating[typing.Any], int]`.
        f"{B * T * L * D * param_size_multiplier / 1.0e9: .2f} GB"
    )

    times = []
    for i in range(runs_of_iters):
        requests = generate_requests(
            iters,
            B,
            T,
            L,
            E,
            reuse=reuse,
            alpha=alpha,
            weighted=weighted,
            requests_data_file=requests_data_file,
            tables=tables,
        )
        requests = [
            TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
            for req in requests
        ]

        # forward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb.forward(
                indices.int(),
                offsets.int(),
                per_sample_weights,
            ),
            check_median=check_median,
            warmup_ms=warmup_ms,
        )

        # free up GPU memory
        del requests

        logging.info(
            f"Iteration {i}: "
            f"{weights_precision} Forward, B: {B}, "
            f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
            f"Time: {time_per_iter * 1.0e6:.0f}us, "
            f"Memory Usage For Pruning: {mem_for_pruning / 1.0e9:.0f} GB"
        )

        if i >= warmup_runs:
            times.append(time_per_iter)

    time_per_iter = statistics.mean(times)
    bandwidth = read_write_bytes / time_per_iter / 1.0e9

    logging.info(
        f"Average of all iterations: "
        f"{weights_precision} Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {bandwidth: .2f} GB/s, "  # noqa: B950
        f"Time: {time_per_iter * 1.0e6:.0f}us, "
        f"Memory Usage For Pruning: {mem_for_pruning / 1.0e9:.0f} GB"
    )

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weighted=weighted,
        requests_data_file=requests_data_file,
        tables=tables,
    )
    requests = [
        TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
        for req in requests
    ]

    # pyre-ignore[3]
    def context_factory(on_trace_ready: Callable[[profile], None]):
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    # Get trace for one run of iter
    tbe_type: str = "split"
    # input of the kineto_trace_profiler
    trace_info = ("fwd", trace_url, tbe_type, "embedding_codegen_forward")
    time_dict = {"kernel_time": None}  # dict to hold the kernel time

    # warm-up right before profiling
    # warmup_ms prioritized over warmup_runs
    if warmup_ms or warmup_runs:
        bench_warmup(
            requests[0],
            # pyre-ignore[6]
            warmup_ms,
            warmup_runs,
            lambda indices, offsets, per_sample_weights: emb.forward(
                indices.int(),
                offsets.int(),
                per_sample_weights,
            ),
        )

    with context_factory(
        # pyre-ignore[6]
        lambda p: time_dict.update(kernel_time=kineto_trace_profiler(p, trace_info))
    ):
        # forward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb.forward(
                indices.int(),
                offsets.int(),
                per_sample_weights,
            ),
            check_median=check_median,
        )

    if export_trace:
        kernel_time = time_dict["kernel_time"]
        # pyre-ignore[58]
        bandwidth = read_write_bytes / kernel_time / 1.0e3

        logging.info(
            f"kineto profiled stats: "
            f"{weights_precision} Forward, B: {B}, "
            f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {bandwidth: .2f} GB/s, "  # noqa: B950
            f"Time: {kernel_time:.0f}us, "
            f"Memory Usage For Pruning: {mem_for_pruning / 1.0e9:.0f} GB"
        )

    # free up GPU memory
    del requests

    reporter.emit_metric(
        type="NET",
        metric=f"bandwidth_{weights_precision}",
        unit="scalar",
        value=str(bandwidth),
    )
    reporter.emit_metric(
        type="NET",
        metric=f"time_per_iter_{weights_precision}",
        unit="scalar",
        value=str(time_per_iter * 1.0e6),
    )

    if run_reference:
        times = []
        for i in range(runs_of_iters):
            requests = generate_requests(
                iters,
                B,
                T,
                L,
                E,
                reuse=reuse,
                alpha=alpha,
                weighted=weighted,
                requests_data_file=requests_data_file,
                tables=tables,
            )
            requests = [
                TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
                for req in requests
            ]

            # forward
            time_per_iter_refer = benchmark_requests_refer(
                requests,
                T,
                B,
                L,
                E,
                # pyre-fixme[6]: For 6th argument expected `int` but got
                #  `Union[floating[typing.Any], int]`.
                D,
                pooling,
                weighted,
                check_median=check_median,
            )

            # free up GPU memory
            del requests

            logging.info(
                f"Reference (nn.Embedding(Bag)) Iteration {i}: "
                f"Forward, B: {B}, "
                f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
                f"BW: {read_write_bytes / time_per_iter_refer / 1.0e9: .2f} GB/s, "  # noqa: B950
                f"Time: {time_per_iter_refer * 1.0e6:.0f}us "
            )

            if i >= warmup_runs:
                times.append(time_per_iter_refer)

        time_per_iter_refer = statistics.mean(times)
        bandwidth = read_write_bytes / time_per_iter_refer / 1.0e9

        logging.info(
            f"Average of all iterations: "
            f"Forward, B: {B}, "
            f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
            f"Effective BW: {bandwidth: .2f} GB/s, "  # noqa: B950
            f"Time: {time_per_iter_refer * 1.0e6:.0f}us "
        )


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size-list", type=str, default="20")
@click.option("--batch-size", default=512)
@click.option("--embedding-dim-list", type=str, default="128")
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--managed", default="device")
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings-list", type=str, default="100000")
@click.option("--reuse", default=0.0)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--pooling", type=str, default="sum")
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.NONE.value)
@click.option("--pruning-ratio", type=float, default=None)
@click.option("--pruning-hash-load-factor", default=0.75)
@click.option("--use-array-for-index-remapping", is_flag=True, default=True)
@click.option("--check-median", is_flag=True, default=True)
@click.option("--iters", default=100)
@click.option("--runs-of-iters", default=5)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--report-aibench", is_flag=True)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
@click.option("--use-cpu", is_flag=True, default=False)
@click.option("--export-trace", is_flag=True, default=False)
@click.option(
    "--trace-url",
    type=str,
    default="{tbe_type}_tbe_spec_{phase}_trace_{ospid}.json",
)
@click.option(
    "--warmup-runs",
    default=2,
    help="Number of warmup runs. Ignored if --warmup-ms is set.",
)
@click.option(
    "--warmup-ms",
    type=int,
    default=None,
    help="Warmup duration in milliseconds. Disables the --run-nums option.",
)
def nbit_device_with_spec(  # noqa C901
    alpha: float,
    bag_size_list: str,
    batch_size: int,
    embedding_dim_list: str,
    weights_precision: SparseType,
    managed: str,
    mixed: bool,
    num_embeddings_list: str,
    reuse: float,
    weighted: bool,
    pooling: str,
    bounds_check_mode: int,
    pruning_ratio: Optional[float],
    pruning_hash_load_factor: float,
    use_array_for_index_remapping: bool,
    check_median: bool,
    iters: int,
    runs_of_iters: int,
    output_dtype: SparseType,
    report_aibench: bool,
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
    use_cpu: bool,
    export_trace: bool,
    trace_url: str,
    warmup_runs: int,
    warmup_ms: Optional[int],
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    reporter = BenchmarkReporter(report_aibench)
    B = batch_size
    Ds = [int(D) for D in embedding_dim_list.split(",")]
    Ls = [int(L) for L in bag_size_list.split(",")]
    Es = [int(E) for E in num_embeddings_list.split(",")]
    E = np.mean(Es)
    D = np.mean(Ds)
    L = np.mean(Ls)
    T = len(Ds)
    logging.info("TBE Spec:")
    logging.info("#, E, D, L")
    for i, (e, d, bag_size) in enumerate(zip(Es, Ds, Ls)):
        logging.info(f"{i}, {e}, {d}, {bag_size}")
    logging.info(f"Mean(Es) = {E}, Mean(Ds) = {D}, Mean(Ls) = {L}")
    index_remapping = None

    mem_for_pruning = 0
    if pruning_ratio:
        original_Es = Es
        assert pruning_ratio < 1 and pruning_ratio >= 0
        index_remapping = []
        new_Es = []
        for original_E in original_Es:
            E = math.ceil(original_E * (1.0 - pruning_ratio))
            mapping = torch.tensor([-1] * original_E, dtype=torch.int32)
            selected_indices = random.sample(range(original_E), E)
            for i, idx in enumerate(selected_indices):
                mapping[idx] = i
            index_remapping.append(mapping)
            if use_array_for_index_remapping:
                mem_for_pruning += mapping.numel() * 4
            else:
                mem_for_pruning += E / pruning_hash_load_factor * 2 * 4
            new_Es.append(E)
        Es = new_Es
        E = np.mean(Es)
        logging.info(f"After prunnig (pruning_ratio={pruning_ratio}")
        logging.info("#, E, D, L")
        for i, (e, d, bag_size) in enumerate(zip(Es, Ds, Ls)):
            logging.info(f"{i}, {e}, {d}, {bag_size}")
        logging.info(f"Mean(Es) = {E}, Mean(Ds) = {D}, Mean(Ls) = {L}")

    if managed == "device":
        managed_option = EmbeddingLocation.DEVICE
    else:
        managed_option = EmbeddingLocation.MANAGED
    # Override managed_option to HOST if using CPU
    if use_cpu:
        managed_option = EmbeddingLocation.HOST

    if pooling is None or pooling == "sum":
        pooling = "sum"
        pooling_mode = PoolingMode.SUM
        do_pooling = True
    elif pooling == "mean":
        pooling_mode = PoolingMode.MEAN
        do_pooling = True
    else:  # "none"
        pooling_mode = PoolingMode.NONE
        do_pooling = False

    emb = IntNBitTableBatchedEmbeddingBagsCodegen(
        [("", e, d, weights_precision, managed_option) for d, e in zip(Ds, Es)],
        device="cpu" if use_cpu else None,
        bounds_check_mode=BoundsCheckMode(bounds_check_mode),
        index_remapping=index_remapping,
        pruning_hash_load_factor=pruning_hash_load_factor,
        use_array_for_index_remapping=use_array_for_index_remapping,
        output_dtype=output_dtype,
        pooling_mode=pooling_mode,
        fp8_exponent_bits=fp8_exponent_bits,
        fp8_exponent_bias=fp8_exponent_bias,
    )
    if use_cpu:
        emb = emb.cpu()
    else:
        emb = emb.cuda()
    emb.fill_random_weights()
    fill_random_scale_bias(emb, T, weights_precision)

    nparams_byte = sum(w.numel() for (w, _) in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    if do_pooling:
        read_write_bytes = sum(
            [
                output_size_multiplier * B * d
                + param_size_multiplier * B * bag_size * d
                for bag_size, d in zip(Ls, Ds)
            ]
        )
    else:
        read_write_bytes = sum(
            [
                output_size_multiplier * B * bag_size * d
                + param_size_multiplier * B * bag_size * d
                for bag_size, d in zip(Ls, Ds)
            ]
        )
    logging.info(
        f"{weights_precision} Embedding tables: {sum(Es)} rows, {nparams_byte / param_size_multiplier / 1.0e9: .2f} GParam, "
        f"{nparams_byte / 1.0e9: .2f} GB"  # IntN TBE use byte for storage
    )
    logging.info(
        f"Accessed weights per batch: {B * sum(Ls)} rows, "
        f"{B * sum([bag_size * d for bag_size, d in zip(Ls, Ds)]) * param_size_multiplier / 1.0e9: .2f} GB"
    )

    times = []
    kineto_request = []
    for i in range(runs_of_iters):
        # Generate a request for each table then combine
        all_requests = {
            "indices": [[] for _ in range(iters)],
            "offsets": [[] for _ in range(iters)],
            "weights": [[] for _ in range(iters)],
        }
        # row = iter, column = tensor
        for t, (bag_size, e) in enumerate(zip(Ls, Es)):
            requests = generate_requests(
                iters,
                B,
                1,
                bag_size,
                e,
                reuse=reuse,
                # don't use zipf if e isn't large enough compared to bag_size.
                alpha=alpha if (e / bag_size) > 2.0 else 1.0,
                # need many more samples for zipf if bag_size is very small.
                zipf_oversample_ratio=3 if bag_size > 5 else 10,
                weighted=weighted,
                use_cpu=use_cpu,
            )
            for it, req in enumerate(requests):
                indices, offsets, weights = req.unpack_3()
                all_requests["indices"][it].append(indices)
                if t > 0:
                    offsets = offsets[1:]  # remove the first element
                    offsets += all_requests["offsets"][it][t - 1][-1]
                all_requests["offsets"][it].append(offsets)
                all_requests["weights"][it].append(weights)
        requests = []
        for it in range(iters):
            indices = torch.concat(all_requests["indices"][it])
            offsets = torch.concat(all_requests["offsets"][it])
            if weighted:
                weights = torch.concat(all_requests["weights"][it])
            else:
                weights = None
            requests.append(TBERequest(indices, offsets, weights))
        if use_cpu:
            requests = [
                TBERequest(
                    req.indices.cpu().int(),
                    req.offsets.cpu().int(),
                    req.per_sample_weigths.cpu() if req.per_sample_weights else None,
                )
                for req in requests
            ]
        else:
            requests = [
                TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
                for req in requests
            ]
        del all_requests
        assert len(requests) == iters

        # forward
        if use_cpu:
            time_per_iter = benchmark_cpu_requests(
                requests,
                lambda indices, offsets, per_sample_weights: emb.forward(
                    indices.int(),
                    offsets.int(),
                    per_sample_weights,
                ),
            )
        else:
            time_per_iter = benchmark_requests(
                requests,
                lambda indices, offsets, per_sample_weights: emb.forward(
                    indices.int(),
                    offsets.int(),
                    per_sample_weights,
                ),
                check_median=check_median,
                warmup_ms=warmup_ms,
            )

        # copy the request of last iteration for kineto profile benchmark
        if i == runs_of_iters - 1:
            kineto_request = requests

        # free up memory
        del requests

        logging.info(
            f"Iteration {i}: "
            f"{weights_precision} Forward, B: {B}, "
            f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
            f"Time: {time_per_iter * 1.0e6:.0f}us, "
            f"Memory Usage For Pruning: {mem_for_pruning / 1.0e9:.0f} GB"
        )

        if i >= warmup_runs:
            times.append(time_per_iter)

    time_per_iter = statistics.mean(times)
    bandwidth = read_write_bytes / time_per_iter / 1.0e9

    logging.info(
        f"Average of all iterations: "
        f"{weights_precision} Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {bandwidth: .2f} GB/s, "  # noqa: B950
        f"Time: {time_per_iter * 1.0e6:.0f}us, "
        f"Memory Usage For Pruning: {mem_for_pruning / 1.0e9:.0f} GB"
    )

    # pyre-ignore[3]
    def context_factory(on_trace_ready: Callable[[profile], None]):
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    if not use_cpu:
        # profile with kineto
        tbe_type: str = "split"
        time_dict = {"kernel_time": None}  # Shared variable to hold the kernel time
        trace_info = ("fwd", trace_url, tbe_type, "embedding_codegen_forward")

        # warm-up right before profiling
        # warmup_ms prioritized over warmup_runs
        if warmup_ms or warmup_runs:
            bench_warmup(
                kineto_request[0],
                # pyre-ignore[6]
                warmup_ms,
                warmup_runs,
                lambda indices, offsets, per_sample_weights: emb.forward(
                    indices.int(),
                    offsets.int(),
                    per_sample_weights,
                ),
            )

        with context_factory(
            # pyre-ignore[6]
            lambda p: time_dict.update(kernel_time=kineto_trace_profiler(p, trace_info))
        ):
            # forward
            time_per_iter = benchmark_requests(
                kineto_request,
                lambda indices, offsets, per_sample_weights: emb.forward(
                    indices.int(),
                    offsets.int(),
                    per_sample_weights,
                ),
                check_median=check_median,
            )

        if export_trace:
            kernel_time = time_dict["kernel_time"]
            # pyre-ignore[6]
            bandwidth = read_write_bytes / kernel_time / 1.0e3

            logging.info(
                f"kineto profiled stats: "
                f"{weights_precision} Forward, B: {B}, "
                f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
                f"BW: {bandwidth: .2f} GB/s, "  # noqa: B950
                f"Time: {kernel_time:.0f}us, "
                f"Memory Usage For Pruning: {mem_for_pruning / 1.0e9:.0f} GB"
            )

    # free up memory
    del kineto_request

    reporter.emit_metric(
        type="NET",
        metric=f"bandwidth_{weights_precision}",
        unit="scalar",
        value=str(bandwidth),
    )
    reporter.emit_metric(
        type="NET",
        metric=f"time_per_iter_{weights_precision}",
        unit="scalar",
        value=str(time_per_iter * 1.0e6),
    )


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.1)
@click.option("--uvm-num-embeddings", default=int(1e5))
@click.option("--uvm-tables", default=1)
@click.option("--uvm-bag-size", default=1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--use-cache", is_flag=True, default=False)
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-load-factor", default=0.2)
@click.option("--enforce-hbm", is_flag=True, default=False)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
@click.option("--uvm-host-mapped", is_flag=True, default=False)
def nbit_uvm(
    alpha: bool,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    iters: int,
    warmup_runs: int,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    uvm_num_embeddings: int,
    uvm_tables: int,
    uvm_bag_size: int,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
    output_dtype: SparseType,
    use_cache: bool,
    cache_algorithm: str,
    cache_load_factor: float,
    enforce_hbm: bool,
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
    uvm_host_mapped: bool,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    E_uvm = uvm_num_embeddings
    T = num_tables
    T_uvm = uvm_tables
    assert T_uvm <= T
    assert (
        T_uvm > 0
    ), f"T_uvm specified {T_uvm} <= 0. If not testing UVM, please use device benchmark."
    T_gpu = T - T_uvm
    L_uvm = uvm_bag_size
    cache_alg = CacheAlgorithm.LRU if cache_algorithm == "lru" else CacheAlgorithm.LFU
    managed_type = (
        EmbeddingLocation.MANAGED_CACHING if use_cache else EmbeddingLocation.MANAGED
    )

    logging.info(f"T: {T}, T_uvm: {T_uvm}, T_gpu: {T_gpu}")

    if mixed:
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T
    emb_uvm = IntNBitTableBatchedEmbeddingBagsCodegen(
        [
            (
                "",
                E_uvm,
                d,
                weights_precision,
                managed_type,
            )
            for d in Ds[:T_uvm]
        ],
        output_dtype=output_dtype,
        cache_load_factor=cache_load_factor,
        cache_algorithm=cache_alg,
        enforce_hbm=enforce_hbm,
        fp8_exponent_bits=fp8_exponent_bits,
        fp8_exponent_bias=fp8_exponent_bias,
        uvm_host_mapped=uvm_host_mapped,
    ).cuda()
    emb_uvm.fill_random_weights()

    if T_gpu > 0:
        emb_gpu = IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    "",
                    E,
                    d,
                    weights_precision,
                    EmbeddingLocation.DEVICE,
                )
                for d in Ds[T_uvm:]
            ],
            output_dtype=output_dtype,
            fp8_exponent_bits=fp8_exponent_bits,
            fp8_exponent_bias=fp8_exponent_bias,
            uvm_host_mapped=uvm_host_mapped,
        ).cuda()
        emb_gpu.fill_random_weights()

        emb_mixed = IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    "",
                    e,
                    d,
                    weights_precision,
                    managed_option,
                )
                for (e, d, managed_option) in zip(
                    [E_uvm] * T_uvm + [E] * T_gpu,
                    Ds,
                    [managed_type] * T_uvm + [EmbeddingLocation.DEVICE] * T_gpu,
                )
            ],
            output_dtype=output_dtype,
            cache_load_factor=cache_load_factor,
            cache_algorithm=cache_alg,
            enforce_hbm=enforce_hbm,
            fp8_exponent_bits=fp8_exponent_bits,
            fp8_exponent_bias=fp8_exponent_bias,
            uvm_host_mapped=uvm_host_mapped,
        ).cuda()
        emb_mixed.fill_random_weights()

    requests_uvm = generate_requests(
        iters,
        B,
        T_uvm,
        L_uvm,
        E_uvm,
        reuse=reuse,
        alpha=alpha,
        weighted=weighted,
    )
    requests_uvm = [
        TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
        for req in requests_uvm
    ]

    requests_gpu = None
    if T_gpu > 0:
        requests_gpu = generate_requests(
            iters,
            B,
            T_gpu,
            L,
            E,
            reuse=reuse,
            alpha=alpha,
            weighted=False,
        )
        requests_gpu = [
            TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
            for req in requests_gpu
        ]

    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes_uvm = (
        output_size_multiplier * B * sum(Ds[:T_uvm])
        + param_size_multiplier * B * sum(Ds[:T_uvm]) * L_uvm
    )

    if T_gpu > 0:
        nparams_byte = sum(w.numel() for (w, _) in emb_mixed.split_embedding_weights())
        logging.info(
            f"{weights_precision} Embedding tables: {E * T_gpu + E_uvm * T_uvm} rows, {nparams_byte / param_size_multiplier / 1.0e9: .2f} GParam, "
            f"{nparams_byte / 1.0e9: .2f} GB"  # IntN TBE use byte for storage
        )
        logging.info(
            f"Accessed weights per batch: {B * (T_gpu * L + T_uvm * L_uvm)} rows, "
            f"{B * (L * sum(Ds[T_uvm:]) + L_uvm * sum(Ds[:T_uvm])) * param_size_multiplier / 1.0e9: .2f} GB"
        )
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("uvm forward")

    time_per_iter = benchmark_requests(
        requests_uvm,
        lambda indices, offsets, per_sample_weights: emb_uvm.forward(
            indices.int(),
            offsets.int(),
            per_sample_weights,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        num_warmups=warmup_runs,
    )
    logging.info(
        f"UVM NBit Forward, {weights_precision}, B: {B}, "
        f"E_uvm: {E_uvm}, T: {T_uvm}, D: {D}, L: {L_uvm}, W: {weighted}, "
        f"BW: {read_write_bytes_uvm / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"Time: {time_per_iter * 1.0e6:.0f}us"
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    if T_gpu > 0:
        requests = []
        assert requests_gpu is not None
        for rs_uvm, rs_gpu in zip(requests_uvm, requests_gpu):
            indices = torch.cat([rs_uvm.indices, rs_gpu.indices])
            lengths = [L_uvm] * (T_uvm * B) + [L] * (T_gpu * B)
            offsets = torch.tensor(([0] + np.cumsum(lengths).tolist())).int().cuda()
            per_sample_weights = None
            if weighted:
                this_rs_uvm_weights = rs_uvm.per_sample_weights
                assert this_rs_uvm_weights is not None
                this_rs_gpu_weights = rs_gpu.per_sample_weights
                assert this_rs_gpu_weights is not None
                per_sample_weights = torch.cat(
                    [this_rs_uvm_weights, this_rs_gpu_weights]
                )
            requests.append(TBERequest(indices, offsets, per_sample_weights))

        # forward
        time_per_iter = benchmark_requests(
            requests_gpu,
            lambda indices, offsets, per_sample_weights: emb_gpu.forward(
                indices.int(),
                offsets.int(),
                per_sample_weights,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        )

        read_write_bytes_hbm = (
            output_size_multiplier * B * sum(Ds[T_uvm:])
            + param_size_multiplier * B * sum(Ds[T_uvm:]) * L
        )
        logging.info(
            f"GPU NBit Forward, {weights_precision}, B: {B}, "
            f"E: {E}, T: {T_gpu}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {read_write_bytes_hbm / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
            f"Time: {time_per_iter * 1.0e6:.0f}us"
        )

        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb_mixed.forward(
                indices.int(),
                offsets.int(),
                per_sample_weights,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            num_warmups=warmup_runs,
        )
        read_write_bytes_total = read_write_bytes_uvm + read_write_bytes_hbm
        logging.info(
            f"Mixed NBit Forward, {weights_precision}, B: {B}, "
            f"E_GPU: {E}, E_UVM: {E_uvm}, T_GPU: {T_gpu}, T_UVM: {T_uvm}, D: {D}, L_GPU: {L}, L_UVM: {L_uvm}, W: {weighted}, "
            f"BW: {read_write_bytes_total / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
            f"Time: {time_per_iter * 1.0e6:.0f}us"
        )

        # benchmark prefetch
        emb_mixed.reset_cache_states()
        for req in requests:
            indices, offsets = req.unpack_2()
            emb_mixed.forward(indices, offsets)
        # TODO: Add warmup runs
        prefetch_time, forward_time = benchmark_pipelined_requests(
            requests,
            lambda indices, offsets, indices_weights: emb_mixed.prefetch(
                indices,
                offsets,
            ),
            # pyre-fixme[6]: For 3rd argument expected `(Tensor, Tensor,
            #  Optional[Tensor]) -> None` but got `(indices: Any, offsets: Any,
            #  indices_weights: Any) -> Tensor`.
            lambda indices, offsets, indices_weights: emb_mixed.forward(
                indices,
                offsets,
                indices_weights,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        )
        e2e_time = prefetch_time + forward_time

        logging.info(
            f"Forward(LXU) {weights_precision}, reuse: {reuse}, alpha: {alpha}, B: {B}, "
            f"E: {E}, T: {T}, D: {D}, L: {L}, "
            f"Te2e: {e2e_time * 1.0e6:.0f}us, "
            f"e2e BW: {read_write_bytes_total / e2e_time / 1.0e9: .2f} GB/s, "
            f"Tprefetch: {prefetch_time * 1.0e6:.0f}us, "
            f"TfwdTime: {forward_time * 1.0e6:.0f}us, "
            f"{read_write_bytes_total / forward_time / 1.0e9: .2f} GB/s"
        )


@cli.command()
@click.option("--test-name", type=str, default="")
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--iters", default=100)
@click.option("--warmup_runs", default=10)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--use-cache", is_flag=True, default=False)
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-load-factor", default=0.2)
@click.option("--enforce-hbm", is_flag=True, default=False)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
@click.option("--record-cache", is_flag=True, default=False)
@click.option("--uvm-host-mapped", is_flag=True, default=False)
@click.option(
    "--dump-requests", type=int, default=0, help="number of reqs to dump (0=no dump)"
)
def nbit_uvm_compare_direct_mapped(
    test_name: str,
    alpha: bool,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    iters: int,
    warmup_runs: int,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
    output_dtype: SparseType,
    use_cache: bool,
    cache_algorithm: str,
    cache_load_factor: float,
    enforce_hbm: bool,
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
    record_cache: bool,
    uvm_host_mapped: bool,
    dump_requests: int,
) -> None:
    logging.info(json.dumps({k: str(v) for k, v in locals().items()}, indent=2))

    np.random.seed(42)
    torch.manual_seed(42)
    B: int = batch_size
    D: int = embedding_dim
    L: int = bag_size
    E: int = num_embeddings
    T: int = num_tables
    cache_alg: CacheAlgorithm = (
        CacheAlgorithm.LRU if cache_algorithm == "lru" else CacheAlgorithm.LFU
    )
    managed_type: EmbeddingLocation = (
        EmbeddingLocation.MANAGED_CACHING if use_cache else EmbeddingLocation.MANAGED
    )

    if mixed:
        Ds: List[int] = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        # pyre-fixme[9]: D has type `int`; used as `floating[typing.Any]`.
        D = np.average(Ds)
    else:
        Ds: List[int] = [D] * T

    _requests_uvm = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weighted=weighted,
    )
    requests_uvm: List[TBERequest] = [
        TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
        for req in _requests_uvm
    ]

    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes_uvm: float = (
        output_size_multiplier * B * sum(Ds[:T])
        + param_size_multiplier * B * sum(Ds[:T]) * L
    )

    stats: Dict[str, Any] = {
        "B": B,
        "T": T,
        "E": E,
        "L": L,
        "D": D,
        "reuse": reuse,
    }

    def bench_uvm_cls(
        name: str = "32way",
        cache_assoc: int = 32,
        record_cache: bool = False,
        hbm: bool = False,
    ) -> None:
        loc = managed_type if not hbm else EmbeddingLocation.DEVICE
        emb = IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    "",
                    E,
                    d,
                    weights_precision,
                    loc,
                )
                for d in Ds[:T]
            ],
            output_dtype=output_dtype,
            cache_load_factor=cache_load_factor,
            cache_algorithm=cache_alg,
            cache_assoc=cache_assoc,
            enforce_hbm=enforce_hbm,
            fp8_exponent_bits=fp8_exponent_bits,
            fp8_exponent_bias=fp8_exponent_bias,
            gather_uvm_cache_stats=record_cache,
            uvm_host_mapped=uvm_host_mapped,
        ).cuda()
        emb.fill_random_weights()
        fill_random_scale_bias(emb, T, weights_precision)

        nvtx_range = (
            f"UVM-RECORD-CACHE-{name.upper()}"
            if record_cache
            else f"UVM-{name.upper()}"
        )
        callback_after_warmup = emb.reset_uvm_cache_stats if record_cache else None

        torch.cuda.cudart().cudaProfilerStart()
        time_per_iter = benchmark_requests(
            requests_uvm,
            lambda indices, offsets, per_sample_weights: emb.forward(
                indices.int(),
                offsets.int(),
                per_sample_weights,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            num_warmups=warmup_runs,
            nvtx_range=nvtx_range,
            callback_after_warmup=callback_after_warmup,
        )
        torch.cuda.cudart().cudaProfilerStop()

        nonlocal stats
        if name not in stats:
            stats[name] = {}

        if not record_cache:
            # Only measure time when cache counter is off (serious overhead)
            if name not in stats:
                stats[name] = {}
            stats[name]["bytes"] = read_write_bytes_uvm
            stats[name]["time_per_iter"] = time_per_iter * 1e6

            logging.info(
                f"[{name.center(8)}] "
                f"UVM NBit Forward, {weights_precision}, B: {B}, "
                f"E_uvm: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
                f"BW: {read_write_bytes_uvm / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
                f"Time: {time_per_iter * 1.0e6:.0f}us"
            )

        if record_cache:
            ucs = emb.uvm_cache_stats.detach().cpu().numpy().tolist()
            cache_stats = {
                "num_calls": ucs[0],
                "num_requested_indices": ucs[1],
                "num_unique_indices": ucs[2],
                "num_unique_misses": ucs[3],
                "num_conflict_unique_misses": ucs[4],
                "num_conflict_misses": ucs[5],
            }
            stats[name]["cache_stats"] = cache_stats
            logging.info(f"[{name:>8s}] cache stats {cache_stats}")

    bench_uvm_cls(name="HBM", hbm=True)
    bench_uvm_cls(name="32way", cache_assoc=32)
    bench_uvm_cls(name="1way", cache_assoc=1)

    if record_cache:
        bench_uvm_cls(
            name="32way",
            cache_assoc=32,
            record_cache=True,
        )
        bench_uvm_cls(
            name="1way",
            cache_assoc=1,
            record_cache=True,
        )

    if test_name:
        folder = Path(os.getenv("HOME", ".")) / test_name

        if not folder.is_dir():
            logging.info(f"MAKING FOLDER {folder}")
            folder.mkdir(parents=True, mode=0o755)

        with (folder / "uvm_stats.txt").open("w") as f:
            logging.info(f"Dumping stats at {folder}")
            print(stats, file=f)

        if dump_requests:
            with (folder / "requests.txt").open("w") as f:
                for req in requests_uvm[:dump_requests]:
                    ind, off = req.unpack_2()
                    print(ind.cpu().numpy().tolist(), file=f)
                    print(off.cpu().numpy().tolist(), file=f)


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-load-factor", default=0.2)
@click.option("--cache-assoc", default=32)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--enforce-hbm", is_flag=True, default=False)
@click.option("--record-cache-miss-counter", is_flag=True, default=False)
@click.option("--record-tablewise-cache-miss", is_flag=True, default=False)
@click.option("--gather-uvm-cache-stats", is_flag=True, default=False)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
@click.option("--uvm-host-mapped", is_flag=True, default=False)
def nbit_cache(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    cache_algorithm: str,
    cache_load_factor: float,
    cache_assoc: int,
    embedding_dim: int,
    weights_precision: SparseType,
    iters: int,
    warmup_runs: int,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
    output_dtype: SparseType,
    enforce_hbm: bool,
    record_cache_miss_counter: bool,
    record_tablewise_cache_miss: bool,
    gather_uvm_cache_stats: bool,
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
    uvm_host_mapped: bool,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    cache_alg = CacheAlgorithm.LRU if cache_algorithm == "lru" else CacheAlgorithm.LFU
    if mixed:
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    emb_nc = IntNBitTableBatchedEmbeddingBagsCodegen(
        [
            (
                "",
                E,
                d,
                weights_precision,
                EmbeddingLocation.MANAGED,
            )
            for d in Ds
        ],
        output_dtype=output_dtype,
        enforce_hbm=enforce_hbm,
        fp8_exponent_bits=fp8_exponent_bits,
        fp8_exponent_bias=fp8_exponent_bias,
        cache_assoc=cache_assoc,
        uvm_host_mapped=uvm_host_mapped,
    ).cuda()
    emb_nc.fill_random_weights()
    fill_random_scale_bias(emb_nc, T, weights_precision)

    emb = IntNBitTableBatchedEmbeddingBagsCodegen(
        [
            (
                "",
                E,
                d,
                weights_precision,
                EmbeddingLocation.MANAGED_CACHING,
            )
            for d in Ds
        ],
        record_cache_metrics=RecordCacheMetrics(
            record_cache_miss_counter, record_tablewise_cache_miss
        ),
        gather_uvm_cache_stats=gather_uvm_cache_stats,
        cache_load_factor=cache_load_factor,
        cache_algorithm=cache_alg,
        output_dtype=output_dtype,
        enforce_hbm=enforce_hbm,
        fp8_exponent_bits=fp8_exponent_bits,
        fp8_exponent_bias=fp8_exponent_bias,
        cache_assoc=cache_assoc,
        uvm_host_mapped=uvm_host_mapped,
    ).cuda()
    emb.fill_random_weights()
    fill_random_scale_bias(emb, T, weights_precision)

    nparams_byte = sum(w.numel() for (w, _) in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes = (
        # read L rows per batch per table.
        param_size_multiplier * B * sum(Ds) * L
        # write 1 row (assuming pooling) per batch per table.
        + output_size_multiplier * B * sum(Ds)
    )
    logging.info(
        f"{weights_precision} Embedding tables: {E * T} rows, {nparams_byte / param_size_multiplier / 1.0e9: .2f} GParam, "
        f"{nparams_byte / 1.0e9: .2f} GB"  # IntN TBE use byte for storage
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        # pyre-fixme[58]: `*` is not supported for operand types `int` and
        #  `Union[np.floating[typing.Any], int]`.
        f"{B * T * L * D * param_size_multiplier / 1.0e9: .2f} GB"
    )

    requests = generate_requests(
        2 * iters, B, T, L, E, reuse=reuse, alpha=alpha, weighted=weighted
    )
    requests = [
        TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
        for req in requests
    ]
    warmup_requests, requests = requests[:iters], requests[iters:]

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb_nc(
            indices.int(), offsets.int(), per_sample_weights
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        num_warmups=warmup_runs,
    )
    logging.info(
        f"Forward (UVM) {weights_precision}, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    # warm up
    for req in warmup_requests:
        indices, offsets = req.unpack_2()
        emb.forward(indices.int(), offsets.int())

    # get cache miss rate (forward only) and exchanged cache lines (prefetch)
    cache_misses = []
    exchanged_cache_lines = []
    unique_indices = []
    input_indices = []
    NOT_FOUND = -1
    # reset the cache miss counters after warmup
    if record_cache_miss_counter or record_tablewise_cache_miss:
        emb.reset_cache_miss_counter()
    if gather_uvm_cache_stats:
        emb.reset_uvm_cache_stats()

    for req in requests:
        indices, offsets = req.unpack_2()
        # pyre-fixme[29]: `Union[(self: TensorBase, memory_format:
        #  Optional[memory_format] = ...) -> Tensor, Tensor, Module]` is not a
        #  function.
        old_lxu_cache_state = emb.lxu_cache_state.clone()
        emb.prefetch(indices, offsets)
        exchanged_cache_lines.append(
            # pyre-fixme[16]: Item `bool` of `bool | Tensor` has no attribute `sum`.
            (emb.lxu_cache_state != old_lxu_cache_state)
            .sum()
            .item()
        )
        cache_misses.append(
            (emb.lxu_cache_locations_list.top() == NOT_FOUND).sum().item()
        )
        emb.forward(indices, offsets)
        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            emb.cache_hash_size_cumsum,
            indices,
            offsets,
        )
        unique_indices.append(len(torch.unique(linear_cache_indices, sorted=False)))
        input_indices.append(len(indices))

    logging.info(
        f"Exchanged cache lines -- mean: {sum(exchanged_cache_lines) / len(requests): .2f}, "
        f"max: {max(exchanged_cache_lines)}, min: {min(exchanged_cache_lines)}"
    )
    logging.info(
        f"Cache miss -- mean: {sum(cache_misses) / len(requests)}, "
        f"max: {max(cache_misses)}, min: {min(cache_misses)}"
    )
    logging.info(
        f"input_indices -- mean: {sum(input_indices) / len(requests)}, "
        f"max: {max(input_indices)}, min: {min(input_indices)}"
    )
    logging.info(
        f"unique_indices -- mean: {sum(unique_indices) / len(requests)}, "
        f"max: {max(unique_indices)}, min: {min(unique_indices)}"
    )
    unique_miss_rate = [a / b for (a, b) in zip(exchanged_cache_lines, unique_indices)]
    logging.info(
        f"unique_miss_rate -- mean: {sum(unique_miss_rate) / len(requests)}, "
        f"max: {max(unique_miss_rate)}, min: {min(unique_miss_rate)}"
    )
    if record_cache_miss_counter or record_tablewise_cache_miss:
        emb.print_cache_miss_counter()
    if gather_uvm_cache_stats:
        emb.print_uvm_cache_stats()

    # benchmark prefetch
    if record_cache_miss_counter or record_tablewise_cache_miss:
        emb.reset_cache_states()
    if gather_uvm_cache_stats:
        emb.reset_uvm_cache_stats()

    for req in warmup_requests:
        indices, offsets = req.unpack_2()
        emb.forward(indices, offsets)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("pipeline")
    # TODO: Add warmup_runs
    prefetch_time, forward_time = benchmark_pipelined_requests(
        requests,
        lambda indices, offsets, indices_weights: emb.prefetch(
            indices,
            offsets,
        ),
        # pyre-fixme[6]: For 3rd argument expected `(Tensor, Tensor,
        #  Optional[Tensor]) -> None` but got `(indices: Any, offsets: Any,
        #  indices_weights: Any) -> Tensor`.
        lambda indices, offsets, indices_weights: emb.forward(
            indices,
            offsets,
            indices_weights,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    e2e_time = prefetch_time + forward_time
    torch.cuda.nvtx.range_pop()

    logging.info(
        f"Forward(LXU) {weights_precision}, reuse: {reuse}, alpha: {alpha}, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, "
        f"Te2e: {e2e_time * 1.0e6:.0f}us, "
        f"e2e BW: {read_write_bytes / e2e_time / 1.0e9: .2f} GB/s, "
        f"Tprefetch: {prefetch_time * 1.0e6:.0f}us, "
        # 2x for reading exchanged_cache_lines from CPU memory through UVM and writing them to GPU HBM.
        f"{2 * sum(exchanged_cache_lines) * param_size_multiplier * D / prefetch_time / len(requests) / 1.0e9: .2f} GB/s, "
        f"TfwdTime: {forward_time * 1.0e6:.0f}us, "
        f"{read_write_bytes / forward_time / 1.0e9: .2f} GB/s"
    )
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    cli()
