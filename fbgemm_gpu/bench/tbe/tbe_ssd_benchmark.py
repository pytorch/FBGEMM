#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import tempfile
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import benchmark_requests
from fbgemm_gpu.tbe.ssd import (
    SSDIntNBitTableBatchedEmbeddingBags,
    SSDTableBatchedEmbeddingBags,
)
from fbgemm_gpu.tbe.utils import generate_requests, get_device, round_up, TBERequest
from fbgemm_gpu.utils.loader import load_torch_module
from torch.autograd.profiler import record_function
from torch.profiler import profile

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

load_torch_module(
    "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings",
)


@click.group()
def cli() -> None:
    pass


def benchmark_ssd_function(
    iters: int,
    warmup_iters: int,
    # pyre-fixme[2]: Parameter must be annotated.
    s,
    buf: torch.Tensor,
    indices: torch.Tensor,
    indices_per_itr: int,
) -> Tuple[float, float]:
    actions_count_cpu = torch.tensor([indices_per_itr]).long().cpu()
    # warmup
    for i in range(warmup_iters):
        start = i * indices_per_itr
        end = start + indices_per_itr
        indices_this_itr = indices[start:end]
        # Benchmark code
        s.get(indices_this_itr, buf, actions_count_cpu)
        s.set(indices_this_itr, buf, actions_count_cpu)
    logging.info("Finished warmup")
    total_time_read_ns = 0
    total_time_write_ns = 0

    for i in range(iters):
        start = (i + warmup_iters) * indices_per_itr
        end = start + indices_per_itr
        indices_this_itr = indices[start:end]
        # Benchmark code
        start = time.time_ns()
        s.get(indices_this_itr, buf, actions_count_cpu)
        read_end = time.time_ns()
        s.set(indices_this_itr, buf, actions_count_cpu)
        end = time.time_ns()
        total_time_read_ns += read_end - start
        total_time_write_ns += end - read_end
        if i % 100 == 0:
            logging.info(
                f"{i}, {(read_end - start) / 10**3} us, {(end - read_end) / 10**3} us"
            )
    return (total_time_read_ns / iters, total_time_write_ns / iters)


def benchmark_read_write(
    ssd_prefix: str,
    batch_size: int,
    bag_size: int,
    num_embeddings: int,
    embedding_dim: int,
    iters: int,
    warmup_iters: int,
    num_shards: int,
    num_threads: int,
    block_cache_size_mb: int,
) -> None:
    idx_dtype = torch.int64
    data_dtype = torch.float32
    np.random.seed(42)
    torch.random.manual_seed(43)
    elem_size = 4

    with tempfile.TemporaryDirectory(prefix=ssd_prefix) as ssd_directory:
        # pyre-fixme[16]: Module `classes` has no attribute `fbgemm`.
        ssd_db = torch.classes.fbgemm.EmbeddingRocksDBWrapper(
            ssd_directory,
            num_shards,
            num_threads,
            0,  # ssd_memtable_flush_period,
            0,  # ssd_memtable_flush_offset,
            4,  # ssd_l0_files_per_compact,
            embedding_dim,
            0,  # ssd_rate_limit_mbps,
            1,  # ssd_size_ratio,
            8,  # ssd_compaction_trigger,
            536870912,  # 512MB ssd_write_buffer_size,
            8,  # ssd_max_write_buffer_num,
            -0.01,  # ssd_uniform_init_lower
            0.01,  # ssd_uniform_init_upper
            32,  # row_storage_bitwidth
            block_cache_size_mb * (2**20),  # block cache size
        )

        total_indices = (warmup_iters + iters) * batch_size * bag_size
        indices_per_itr = batch_size * bag_size
        indices = torch.randint(
            low=0, high=num_embeddings, size=(total_indices,), dtype=idx_dtype
        )
        buf = torch.empty((batch_size * bag_size, embedding_dim), dtype=data_dtype)

        read_lat_ns, write_lat_ns = benchmark_ssd_function(
            iters, warmup_iters, ssd_db, buf, indices, indices_per_itr
        )
        total_bytes = batch_size * embedding_dim * bag_size * elem_size
        byte_seconds_per_ns = total_bytes * 1e9
        gibps_rd = byte_seconds_per_ns / (read_lat_ns * 2**30)
        gibps_wr = byte_seconds_per_ns / (write_lat_ns * 2**30)
        gibps_tot = 2 * byte_seconds_per_ns / ((read_lat_ns + write_lat_ns) * 2**30)
        logging.info(
            f"Total bytes: {total_bytes / 1e9:0.2f} GB, "
            f"Read_us: {read_lat_ns / 1000:8.0f}, "
            f"Write_us: {write_lat_ns / 1000:8.0f}, "
            f"Total_us: {(read_lat_ns + write_lat_ns) / 1000:8.0f}, "
            f"TMaxQPS: {1e9 * batch_size / (read_lat_ns + write_lat_ns):8.0f}, "
            f"GiBps Rd: {gibps_rd:3.2f}, "
            f"GiBps Wr: {gibps_wr:3.2f}, "
            f"GiBps R+W: {gibps_tot:3.2f}, "
        )
        del ssd_db


@cli.command()
# @click.option("--num-tables", default=64)
@click.option("--num-embeddings", default=int(1.5e9))
@click.option("--embedding-dim", default=128)
@click.option("--batch-size", default=4096)
@click.option("--bag-size", default=10)
@click.option("--iters", default=400)
@click.option("--warmup-iters", default=100)
@click.option(
    "--ssd-prefix", default="/tmp/ssd_benchmark_embedding"
)  # Check P556577690 and https://fburl.com/t9lf4d7v
@click.option("--num-shards", default=8)
@click.option("--num-threads", default=8)
@click.option("--block-cache-size-mb", default=0)
def ssd_read_write(
    ssd_prefix: str,
    num_embeddings: int,
    embedding_dim: int,
    bag_size: int,
    batch_size: int,
    iters: int,
    warmup_iters: int,
    num_shards: int,
    num_threads: int,
    block_cache_size_mb: int,
) -> None:
    benchmark_read_write(
        ssd_prefix,
        batch_size,
        bag_size,
        num_embeddings,
        embedding_dim,
        iters,
        warmup_iters,
        num_shards,
        num_threads,
        block_cache_size_mb,
    )


@cli.command()
# recommended value: alpha=1.15 for training and alpha=1.09 for inference
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=500)
@click.option("--warmup-runs", default=0)
@click.option("--managed", default="device")
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--pooling", type=str, default="sum")
@click.option("--weighted-num-requires-grad", type=int, default=None)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP32)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
@click.option("--ssd-prefix", type=str, default="/tmp/ssd_benchmark")
@click.option("--block-cache-size-mb", default=0)
@click.option("--export-trace", is_flag=True, default=False)
@click.option(
    "--trace-url",
    type=str,
    default="manifold://gpu_traces/tree/fbgemm_gpu/ssd_tbe/trace_{ospid}.json",
)
def ssd_training(  # noqa C901
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
    pooling: str,
    weighted_num_requires_grad: Optional[int],
    flush_gpu_cache_size_mb: int,
    output_dtype: SparseType,
    requests_data_file: Optional[str],
    tables: Optional[str],
    ssd_prefix: Optional[str],
    block_cache_size_mb: int,
    export_trace: bool,
    trace_url: str,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B: int = batch_size
    D: int = embedding_dim
    L: int = bag_size
    E: int = num_embeddings
    T: int = num_tables

    if weighted_num_requires_grad:
        assert weighted_num_requires_grad <= T
        weighted_requires_grad_tables = np.random.choice(
            T, replace=False, size=(weighted_num_requires_grad,)
        ).tolist()
        feature_requires_grad = (
            torch.tensor(
                [1 if t in weighted_requires_grad_tables else 0 for t in range(T)]
            )
            .to(get_device())
            .int()
        )
    else:
        feature_requires_grad = None
    if mixed:
        Ds: List[int] = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        # pyre-fixme[9]: D has type `int`; used as `floating[typing.Any]`.
        D = np.average(Ds)
    else:
        Ds: List[int] = [D] * T

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

    feature_table_map = list(range(T))
    common_args: Dict[str, Any] = {
        "feature_table_map": feature_table_map,
        "learning_rate": 0.1,
        "eps": 0.1,
        "pooling_mode": pooling_mode,
    }
    common_split_tbe_args: Dict[str, Any] = {
        # SSD only supports rowwise-adagrad
        "optimizer": OptimType.EXACT_ROWWISE_ADAGRAD,
        "weights_precision": weights_precision,
        "output_dtype": output_dtype,
    }
    common_split_tbe_args.update(common_args)
    split_tbe_compute_device: ComputeDevice = (
        ComputeDevice.CUDA if torch.cuda.is_available() else ComputeDevice.CPU
    )

    def gen_split_tbe_generator(
        location: EmbeddingLocation,
    ) -> Callable[[], SplitTableBatchedEmbeddingBagsCodegen]:
        return lambda: SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    d,
                    location if torch.cuda.is_available() else EmbeddingLocation.HOST,
                    split_tbe_compute_device,
                )
                for d in Ds
            ],
            **common_split_tbe_args,
        )

    # TODO: Adjust cache sets
    cache_set = max(T * B * L, 1)
    tempdir = tempfile.mkdtemp(prefix=ssd_prefix)
    logging.info(f"Using SSD dir: {tempdir}")
    tbe_generators = {
        "HBM": gen_split_tbe_generator(EmbeddingLocation.DEVICE),
        "UVM": gen_split_tbe_generator(EmbeddingLocation.MANAGED),
        "UVM_CACHING": gen_split_tbe_generator(EmbeddingLocation.MANAGED_CACHING),
        "SSD": lambda: SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, d) for d in Ds],
            cache_sets=cache_set,
            ssd_storage_directory=tempdir,
            ssd_cache_location=EmbeddingLocation.DEVICE,
            ssd_rocksdb_shards=8,
            ssd_block_cache_size_per_tbe=block_cache_size_mb * (2**20),
            **common_args,
        ),
    }

    # Generate input data
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
        use_cpu=not torch.cuda.is_available(),
    )

    # Generate gradients for backward
    if do_pooling:
        grad_output = torch.randn(B, sum(Ds)).to(get_device())
    else:
        grad_output = torch.randn(B * T * L, D).to(get_device())

    # Compute read/write bytes
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    if do_pooling:
        read_write_bytes = (
            output_size_multiplier * B * sum(Ds)
            + param_size_multiplier * B * sum(Ds) * L
        )
    else:
        read_write_bytes = (
            output_size_multiplier * B * sum(Ds) * L
            + param_size_multiplier * B * sum(Ds) * L
        )

    logging.info(f"Batch read write bytes: {read_write_bytes / 1.0e9: .2f} GB")

    # Compute width of test name and bandwidth widths to improve report
    # readability
    name_width = 0
    for k in tbe_generators.keys():
        name_width = max(name_width, len(k))
    name_width += len("Backward") + 2
    bw_width = 8

    def gen_forward_func(
        emb: Union[SplitTableBatchedEmbeddingBagsCodegen, SSDTableBatchedEmbeddingBags],
        feature_requires_grad: Optional[torch.Tensor],
    ) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        return lambda indices, offsets, per_sample_weights: emb.forward(
            indices.long(),
            offsets.long(),
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        )

    def _kineto_trace_handler(p: profile) -> None:
        p.export_chrome_trace(trace_url.format(ospid=os.getpid()))

    prof_ctx = (
        profile(on_trace_ready=_kineto_trace_handler) if export_trace else nullcontext()
    )

    with prof_ctx:
        # Execute tests
        report = []
        nparams = 0
        for prefix, generator in tbe_generators.items():
            # Instantiate TBE
            emb = generator().to(get_device())

            # Forward
            test_name = f"{prefix} Forward"
            logging.info(f"Running benchmark: {test_name}")
            with record_function(f"## {test_name} ##"):
                time_per_iter = benchmark_requests(
                    requests,
                    gen_forward_func(emb, feature_requires_grad),
                    flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
                    num_warmups=warmup_runs,
                    periodic_logs=True,
                )

                bw = f"{read_write_bytes / time_per_iter / 1.0e9: .2f}"
                report.append(
                    f"{test_name: <{name_width}} B: {B}, "
                    f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
                    f"BW: {bw: <{bw_width}} GB/s, "  # noqa: B950
                    f"T: {time_per_iter * 1.0e6:.0f}us"
                )

            # Backward
            test_name = f"{prefix} Backward"
            logging.info(f"Running benchmark: {test_name}")
            with record_function(f"## {test_name} ##"):
                time_per_iter = benchmark_requests(
                    requests,
                    gen_forward_func(emb, feature_requires_grad),
                    flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
                    bwd_only=True,
                    grad=grad_output,
                    num_warmups=warmup_runs,
                    periodic_logs=True,
                )

                bw = f"{2 * read_write_bytes / time_per_iter / 1.0e9: .2f}"
                report.append(
                    f"{test_name: <{name_width}} B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
                    f"BW: {bw: <{bw_width}} GB/s, "
                    f"T: {time_per_iter * 1.0e6:.0f}us"
                )

            # Compute nparams once
            if prefix == "HBM":
                nparams = sum(w.numel() for w in emb.split_embedding_weights())

            # Delete module to make room for other modules
            del emb

    # Print report
    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * sum(Ds) * L * param_size_multiplier / 1.0e9: .2f} GB"
    )
    for r in report:
        logging.info(r)


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--iters", default=100)
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
@click.option("--ssd-cache-loc", default="device")
def nbit_ssd(
    alpha: bool,
    bag_size: int,  # L
    batch_size: int,  # B
    embedding_dim: int,  # D
    weights_precision: SparseType,
    iters: int,
    mixed: bool,
    num_embeddings: int,  # E
    num_tables: int,  # T
    reuse: float,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
    output_dtype: SparseType,
    use_cache: bool,
    cache_algorithm: str,
    cache_load_factor: float,
    enforce_hbm: bool,
    ssd_cache_loc: str,
) -> None:

    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables

    cache_alg = CacheAlgorithm.LRU
    managed_type = (
        EmbeddingLocation.MANAGED_CACHING if use_cache else EmbeddingLocation.MANAGED
    )

    ssd_cache_location = (
        EmbeddingLocation.MANAGED
        if ssd_cache_loc == "managed"
        else EmbeddingLocation.DEVICE
    )

    logging.info(f"T: {T}")

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
                E,
                d,
                weights_precision,
                managed_type,
            )
            for d in Ds
        ],
        output_dtype=output_dtype,
        cache_load_factor=cache_load_factor,
        cache_algorithm=cache_alg,
        enforce_hbm=enforce_hbm,
    ).cuda()
    emb_uvm.fill_random_weights()

    feature_table_map = list(range(T))
    C = max(T * B * L, 1)
    emb_ssd = SSDIntNBitTableBatchedEmbeddingBags(
        embedding_specs=[("", E, d, weights_precision) for d in Ds],
        feature_table_map=feature_table_map,
        ssd_storage_directory=tempfile.mkdtemp(),
        cache_sets=C,
        ssd_uniform_init_lower=-0.1,
        ssd_uniform_init_upper=0.1,
        ssd_shards=2,
        pooling_mode=PoolingMode.SUM,
        ssd_cache_location=ssd_cache_location,  # adjust the cache locations
    ).cuda()

    emb_cpu = IntNBitTableBatchedEmbeddingBagsCodegen(
        [
            (
                "",
                E,
                d,
                weights_precision,
                EmbeddingLocation.HOST,
            )
            for d in Ds
        ],
        output_dtype=output_dtype,
        device="cpu",
    )
    emb_cpu.fill_random_weights()

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weighted=weighted,
    )
    requests_gpu = [
        TBERequest(req.indices.int(), req.offsets.int(), req.per_sample_weights)
        for req in requests
    ]

    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes = (
        output_size_multiplier * B * sum(Ds) + param_size_multiplier * B * sum(Ds) * L
    )

    nparams_byte = sum(w.numel() for (w, _) in emb_cpu.split_embedding_weights())
    logging.info(
        f"{weights_precision} Embedding tables: {E * T} rows, {nparams_byte / param_size_multiplier / 1.0e9: .2f} GParam, "
        f"{nparams_byte / 1.0e9: .2f} GB"  # IntN TBE use byte for storage
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        f"{B * (L * sum(Ds)) * param_size_multiplier / 1.0e9: .2f} GB"
    )

    # UVM
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("uvm forward")
    time_per_iter = benchmark_requests(
        requests_gpu,
        lambda indices, offsets, per_sample_weights: emb_uvm.forward(
            indices.int(),
            offsets.int(),
            per_sample_weights,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"UVM NBit Forward, {weights_precision}, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"Time: {time_per_iter * 1.0e6:.0f}us"
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    # SSD
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("ssd forward")
    time_per_iter = benchmark_requests(
        requests_gpu,
        lambda indices, offsets, per_sample_weights: emb_ssd.forward(
            indices.int(),
            offsets.int(),
            per_sample_weights,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"SSD NBit Forward, {weights_precision}, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"Time: {time_per_iter * 1.0e6:.0f}us"
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    # CPU
    requests_cpu = [
        TBERequest(
            req.indices.int().cpu(), req.offsets.int().cpu(), req.per_sample_weights
        )
        for req in requests
    ]
    time_per_iter = benchmark_requests(
        requests_cpu,
        lambda indices, offsets, per_sample_weights: emb_cpu.forward(
            indices.int().cpu(),
            offsets.int().cpu(),
            per_sample_weights,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )

    logging.info(
        f"CPU NBit Forward, {weights_precision}, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"Time: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--iters", default=100, help="Number of iterations to benchmark")
@click.option("--num-inserts", default=1024, help="Number of rows to insert")
@click.option("--dim", default=128, help="Width of each row to insert")
@click.option(
    "--use-pipeline",
    is_flag=True,
    default=False,
    help="Use a fraction of SMs (using a grid size < the number of SMs)",
)
@click.option(
    "--use-malloc-managed",
    is_flag=True,
    default=False,
    help="Use cudaMallocManaged for the host buffer instead of "
    "malloc+cudaHostRegister",
)
@click.option(
    "--preferred-sms",
    default=-1,
    help="The preferred number of SMs for the kernel to use when using "
    "--use-pipeline. The value is ignored when not using "
    "--use-pipeline.",
)
def masked_index_benchmark(
    iters: int,
    num_inserts: int,
    dim: int,
    use_pipeline: bool,
    use_malloc_managed: bool,
    preferred_sms: int,
) -> None:
    """
    A benchmark for measuring host-to-device copy performance using
    `torch.ops.fbgemm.masked_index_put`. The host buffer is a UVM
    buffer (by default it is malloc+cudaHostRegister).

    Args:

        iters (int): Number of iterations to benchmark

        num_inserts (int):  Number of rows to insert

        dim (int): Width of each row to insert

        use_pipeline (bool): Use a fraction of SMs (using a grid size
        < the number of SMs)

        use_malloc_managed (bool): Use cudaMallocManaged for the host
        buffer instead of malloc+cudaHostRegister

        preferred_sms (int): The preferred number of SMs for the
        kernel to use when use_pipeline=True. The value is ignored
        when use_pipeline=False

    Returns:

        None
    """

    # Common configs
    dtype = torch.half
    device = "cuda"

    # Generate requests
    values_all = torch.ops.fbgemm.new_unified_tensor(
        torch.zeros(1, device=device, dtype=dtype),
        [num_inserts * iters, dim],
        is_host_mapped=not use_malloc_managed,
    )
    output = torch.empty(num_inserts, dim, dtype=dtype, device=device)
    indices = torch.arange(num_inserts, dtype=torch.long, device=device)
    count = torch.as_tensor([indices.numel()], dtype=torch.int, device=device)

    requests = []
    for it in range(iters):
        values = values_all[it * num_inserts : (it + 1) * num_inserts]
        requests.append(TBERequest(output, indices, values))

    # Run benchmark
    time_per_iter = benchmark_requests(
        requests,
        lambda output, indices, values: torch.ops.fbgemm.masked_index_put(
            output,
            indices,
            values,
            count=count,
            use_pipeline=use_pipeline,
            preferred_sms=preferred_sms,
        ),
        num_warmups=10,
    )

    # Report performance
    buffer_bytes = num_inserts * dim * values_all.element_size()
    logging.info(
        f"masked_index_benchmark: use_pipeline {use_pipeline}, "
        f"Read/write bytes {buffer_bytes} bytes, "
        f"BW: {buffer_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"Time {time_per_iter * 1.0e6:.0f} us"
    )


if __name__ == "__main__":
    cli()
