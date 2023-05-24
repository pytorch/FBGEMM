#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Tuple

import click
import numpy as np
import torch
from fbgemm_gpu.bench.bench_utils import benchmark_requests
from fbgemm_gpu.split_embedding_utils import generate_requests, round_up
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.ssd_split_table_batched_embeddings_ops import (
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    SparseType,
    SSDIntNBitTableBatchedEmbeddingBags,
)

logging.basicConfig(level=logging.DEBUG)

torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings"
)


logging.basicConfig(level=logging.DEBUG)


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
        if i % 10 == 0:
            logging.info(
                f"{i}, {(read_end - start) / 10**6}, {(end - read_end) / 10**6}"
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
) -> None:
    import tempfile

    idx_dtype = torch.int64
    data_dtype = torch.float32
    np.random.seed(42)
    torch.random.manual_seed(43)
    elem_size = 4

    with tempfile.TemporaryDirectory(prefix=ssd_prefix) as ssd_directory:
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
            f"Batch Size: {batch_size}, "
            f"Bag_size: {bag_size:3d}, "
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
@click.option("--batch-size", default=1024)
@click.option("--bag-size", default=1)
@click.option("--iters", default=1000)
@click.option("--warmup-iters", default=100)
@click.option(
    "--ssd-prefix", default="/tmp/ssd_benchmark_embedding"
)  # Check P556577690 and https://fburl.com/t9lf4d7v
@click.option("--num-shards", default=8)
@click.option("--num-threads", default=8)
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
    )


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
    import tempfile

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
    requests_gpu = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]

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
        # pyre-ignore
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
        # pyre-ignore
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
        (a.int().cpu(), b.int().cpu(), c if c else None) for (a, b, c) in requests
    ]
    time_per_iter = benchmark_requests(
        # pyre-ignore
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


if __name__ == "__main__":
    cli()
