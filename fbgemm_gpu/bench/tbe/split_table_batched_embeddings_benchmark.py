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
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Tuple

import click
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    get_available_compute_device,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import (
    benchmark_pipelined_requests,
    benchmark_requests,
    benchmark_requests_with_spec,
    benchmark_vbe,
    EmbeddingOpsCommonConfigLoader,
    generate_merged_output_and_offsets,
    TbeBenchClickInterface,
    TBEBenchmarkingConfigLoader,
)
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import generate_requests, get_device, round_up, TBERequest
from torch import Tensor
from torch.profiler import profile

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

try:
    import mtia.host_runtime.torch_mtia.dynamic_library  # pyright: ignore  # noqa: F401  # pyre-ignore[21]

    torch.mtia.init()
except Exception:
    pass


@click.group()
def cli() -> None:
    pass


def get_compute_device(d: torch.device) -> Tuple[ComputeDevice, EmbeddingLocation]:
    if d.type == "cuda":
        return (ComputeDevice.CUDA, EmbeddingLocation.DEVICE)
    elif d.type == "mtia":
        return (ComputeDevice.MTIA, EmbeddingLocation.HOST)
    else:
        return (ComputeDevice.CPU, EmbeddingLocation.HOST)


def get_pooling(pooling: Optional[str]) -> Tuple[PoolingMode, bool]:
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
    return pooling_mode, do_pooling


def compute_read_write_bytes(
    Es: list[int],
    Bs: list[int],
    Ds: list[int],
    L: int | float,
    do_pooling: bool,
    output_dtype: SparseType,
    weights_precision: SparseType,
) -> float:
    # Calculate read/write bytes for bandwidth calculation
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0

    nparams = sum(d * e for d, e in zip(Ds, Es))

    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    numel_per_batch = sum([b * d for b, d in zip(Bs, Ds)])
    _L = 1 if do_pooling else L
    total_batch_accesses = param_size_multiplier * numel_per_batch * L

    read_write_bytes = (
        output_size_multiplier * numel_per_batch * _L + total_batch_accesses
    )
    logging.info(f"Accessed weights per batch: {total_batch_accesses / 1.0e9: .2f} GB")
    return read_write_bytes


@cli.command()
@TbeBenchClickInterface.common_options
@TbeBenchClickInterface.device_options
@TbeBenchClickInterface.table_options
@click.option(
    "--weighted-num-requires-grad",
    type=int,
    default=None,
    help="Number of tables requiring gradient computation for weighted embeddings. Default is None.",
)
@click.option(
    "--dense",
    is_flag=True,
    default=False,
    help="Use dense embedding tables. Default is False.",
)
@click.option(
    "--output-dtype",
    type=SparseType,
    default=SparseType.FP32,
    help="Data type of the output embeddings. Default is FP32.",
)
@click.option(
    "--indices-dtype",
    type=click.Choice(["32", "64"]),
    default="64",
    help="Data type for indices, either 32-bit or 64-bit. Default is 64.",
)
@click.option(
    "--requests_data_file",
    type=str,
    default=None,
    help="File path for requests data. Default is None.",
)
@click.option(
    "--indices-file",
    type=str,
    default=None,
    help="Path to the indices file. Default is None.",
)
@click.option(
    "--offsets-file",
    type=str,
    default=None,
    help="Path to the offsets file. Default is None.",
)
@click.option(
    "--export-trace",
    is_flag=True,
    default=False,
    help="Enable export of trace for profiling. Default is False.",
)
@click.option(
    "--trace-url",
    type=str,
    default="{tbe_type}_tbe_{phase}_trace_{ospid}.json",
)
@click.option(
    "--uvm-host-mapped",
    is_flag=True,
    default=False,
    help="Use host mapped UVM buffers in SSD-TBE (malloc+cudaHostRegister)",
)
@click.option("--ssd", is_flag=True, default=False)
@click.option(
    "--ssd-prefix", type=str, default="/tmp/ssd_benchmark", help="SSD directory prefix"
)
@click.option("--cache-load-factor", default=0.2)
@click.option(
    "--num-requests",
    default=-1,
    help="Number of input batches to generate. If the value is smaller than "
    "iters, the benchmark will reuse the input batches",
)
def device(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    cache_precision: Optional[SparseType],
    stoc: bool,
    iters: int,
    warmup_runs: int,
    managed: click.Choice,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    row_wise: bool,
    weighted: bool,
    pooling: str,
    weighted_num_requires_grad: Optional[int],
    bounds_check_mode: int,
    flush_gpu_cache_size_mb: int,
    dense: bool,
    output_dtype: SparseType,
    indices_dtype: str,
    requests_data_file: Optional[str],
    tables: Optional[str],
    export_trace: bool,
    trace_url: str,
    uvm_host_mapped: bool,
    ssd: bool,
    ssd_prefix: str,
    cache_load_factor: float,
    num_requests: int,
    indices_file: Optional[str],
    offsets_file: Optional[str],
) -> None:
    assert not ssd or not dense, "--ssd cannot be used together with --dense"
    num_requests = iters if num_requests == -1 else num_requests
    indices_dtype_torch: torch.dtype = (
        torch.int32 if int(indices_dtype) == 32 else torch.int64
    )
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
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
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD if row_wise else OptimType.EXACT_ADAGRAD

    if managed == "device":
        managed_option = (
            EmbeddingLocation.DEVICE
            if torch.cuda.is_available()
            else EmbeddingLocation.HOST
        )
    elif managed == "managed":
        managed_option = EmbeddingLocation.MANAGED
    elif managed == "managed_caching":
        managed_option = EmbeddingLocation.MANAGED_CACHING
    else:
        raise ValueError(f"Unknown --managed-option {managed}")

    pooling_mode, do_pooling = get_pooling(pooling)

    common_split_args: dict[str, Any] = {
        "weights_precision": weights_precision,
        "stochastic_rounding": stoc,
        "output_dtype": output_dtype,
        "pooling_mode": pooling_mode,
        "bounds_check_mode": BoundsCheckMode(bounds_check_mode),
        "uvm_host_mapped": uvm_host_mapped,
        "optimizer": optimizer,
        "learning_rate": 0.1,
        "eps": 0.1,
        "feature_table_map": list(range(T)),
    }

    if dense:
        tbe_type: str = "dense"
        emb = DenseTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                )
                for d in Ds
            ],
            pooling_mode=pooling_mode,
            use_cpu=get_available_compute_device() == ComputeDevice.CPU,
        )
    elif ssd:
        assert (
            torch.cuda.is_available()
        ), "SSDTableBatchedEmbeddingBags only supports GPU execution"
        cache_set = max(T * B, 1)
        tempdir = tempfile.mkdtemp(prefix=ssd_prefix)
        emb = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, d) for d in Ds],
            cache_sets=cache_set,
            ssd_storage_directory=tempdir,
            ssd_cache_location=EmbeddingLocation.DEVICE,
            ssd_rocksdb_shards=8,
            **common_split_args,
        )
    else:
        tbe_type: str = "split"
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                    managed_option,
                    get_available_compute_device(),
                )
                for d in Ds
            ],
            cache_precision=cache_precision,
            cache_algorithm=CacheAlgorithm.LRU,
            cache_load_factor=cache_load_factor,
            **common_split_args,
        )
    emb = emb.to(get_device())

    if weights_precision in [SparseType.INT8, SparseType.NFP8]:
        # pyre-fixme[29]: `Union[(self: DenseTableBatchedEmbeddingBagsCodegen,
        #  min_val: float, max_val: float) -> None, (self:
        #  SplitTableBatchedEmbeddingBagsCodegen, min_val: float, max_val: float) ->
        #  None, Tensor, Module]` is not a function.
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(d * E for d in Ds)
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

    logging.info(f"Managed option: {managed}")
    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * sum(Ds) * L * param_size_multiplier / 1.0e9: .2f} GB"
    )
    requests = generate_requests(
        num_requests,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weighted=weighted,
        requests_data_file=requests_data_file,
        indices_file=indices_file,
        offsets_file=offsets_file,
        tables=tables,
        use_cpu=get_available_compute_device() == ComputeDevice.CPU,
        index_dtype=torch.long,
        offset_dtype=torch.long,
    )

    def _kineto_trace_handler(p: profile, phase: str) -> None:
        p.export_chrome_trace(
            trace_url.format(tbe_type=tbe_type, phase=phase, ospid=os.getpid())
        )

    # pyre-ignore[3]
    def context_factory(on_trace_ready: Callable[[profile], None]):
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    with context_factory(lambda p: _kineto_trace_handler(p, "fwd")):
        # forward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb.forward(
                indices.to(dtype=indices_dtype_torch),
                offsets.to(dtype=indices_dtype_torch),
                per_sample_weights,
                feature_requires_grad=feature_requires_grad,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            num_warmups=warmup_runs,
            iters=iters,
        )

    logging.info(
        f"Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, a: {alpha}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    if output_dtype == SparseType.INT8:
        # backward bench not representative
        return

    if do_pooling:
        grad_output = torch.randn(B, sum(Ds)).to(get_device())
    else:
        # pyre-fixme[6]: For 2nd argument expected `Union[int, SymInt]` but got
        #  `Union[floating[typing.Any], int]`.
        grad_output = torch.randn(B * T * L, D).to(get_device())

    with context_factory(lambda p: _kineto_trace_handler(p, "fwd_bwd")):
        # backward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb(
                indices.to(dtype=indices_dtype_torch),
                offsets.to(dtype=indices_dtype_torch),
                per_sample_weights,
                feature_requires_grad=feature_requires_grad,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            bwd_only=True,
            grad=grad_output,
            num_warmups=warmup_runs,
            iters=iters,
        )

    logging.info(
        f"Backward, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, a: {alpha}, "
        f"BW: {2 * read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@TbeBenchClickInterface.common_options
@TbeBenchClickInterface.table_options
@click.option("--uvm-tables", default=1)
@click.option("--uvm-bag-size", default=1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP32)
@click.option("--use-cache", is_flag=True, default=False)
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-load-factor", default=0.2)
@click.option("--enforce-hbm", is_flag=True, default=False)
@click.option("--no-conflict-misses", is_flag=True, default=False)
@click.option("--all-conflict-misses", is_flag=True, default=False)
@click.option(
    "--uvm-host-mapped",
    is_flag=True,
    default=False,
    help="Use host mapped UVM buffers in SSD-TBE (malloc+cudaHostRegister)",
)
@click.option(
    "--export-trace",
    is_flag=True,
    default=False,
    help="Enable export of trace for profiling. Default is False.",
)
@click.option(
    "--trace-url",
    type=str,
    default="{tbe_type}_tbe_{phase}_trace_{ospid}.json",
)
def uvm(  # noqa: C901
    alpha: bool,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
    warmup_runs: int,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    uvm_tables: int,
    uvm_bag_size: int,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
    requests_data_file: Optional[str],
    tables: Optional[str],
    output_dtype: SparseType,
    use_cache: bool,
    cache_algorithm: str,
    cache_load_factor: float,
    enforce_hbm: bool,
    # Simulate a UVM cache with a cache conflict miss rate of 0%
    no_conflict_misses: bool,
    # Simulate a UVM cache with a cache conflict miss rate of 100%
    all_conflict_misses: bool,
    uvm_host_mapped: bool,
    export_trace: bool,
    trace_url: str,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    T_uvm = uvm_tables
    assert T_uvm <= T
    assert (
        T_uvm > 0
    ), f"T_uvm specified {T_uvm} <= 0. If not testing UVM, please use device benchmark."
    assert torch.cuda.is_available(), "UVM benchmark requires CUDA device"

    T_gpu = T - T_uvm
    L_uvm = uvm_bag_size
    eval_conflict_misses: bool = no_conflict_misses or all_conflict_misses

    cache_alg = CacheAlgorithm.LRU if cache_algorithm == "lru" else CacheAlgorithm.LFU
    managed_type = (
        EmbeddingLocation.MANAGED_CACHING if use_cache else EmbeddingLocation.MANAGED
    )

    if mixed:
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    emb_uvm = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                managed_type,
                ComputeDevice.CUDA,
            )
            for d in Ds[:T_uvm]
        ],
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
        output_dtype=output_dtype,
        cache_load_factor=cache_load_factor,
        cache_algorithm=cache_alg,
        enforce_hbm=enforce_hbm,
        uvm_host_mapped=uvm_host_mapped,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb_uvm.init_embedding_weights_uniform(-0.0003, 0.0003)

    if T_gpu > 0:
        emb_gpu = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for d in Ds[T_uvm:]
            ],
            weights_precision=weights_precision,
            stochastic_rounding=stoc,
        ).cuda()

        if weights_precision == SparseType.INT8:
            emb_gpu.init_embedding_weights_uniform(-0.0003, 0.0003)

        emb_mixed = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                    managed_option,
                    ComputeDevice.CUDA,
                )
                for (d, managed_option) in zip(
                    Ds,
                    [managed_type] * T_uvm + [EmbeddingLocation.DEVICE] * T_gpu,
                )
            ],
            weights_precision=weights_precision,
            stochastic_rounding=stoc,
            output_dtype=output_dtype,
            cache_load_factor=cache_load_factor,
            cache_algorithm=cache_alg,
            enforce_hbm=enforce_hbm,
            uvm_host_mapped=uvm_host_mapped,
        ).cuda()

        if weights_precision == SparseType.INT8:
            emb_mixed.init_embedding_weights_uniform(-0.0003, 0.0003)

    requests_uvm = generate_requests(
        iters,
        B,
        T_uvm,
        L_uvm,
        E,
        reuse=reuse,
        alpha=alpha,
        weighted=weighted,
        requests_data_file=requests_data_file,
        tables=tables,
        index_dtype=torch.long,
        offset_dtype=torch.long,
    )

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
            requests_data_file=requests_data_file,
            tables=tables,
            index_dtype=torch.long,
            offset_dtype=torch.long,
        )

    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes_uvm = (
        output_size_multiplier * B * sum(Ds[:T_uvm])
        + param_size_multiplier * B * sum(Ds[:T_uvm]) * L_uvm
    )

    if eval_conflict_misses:
        assert (
            use_cache
        ), "--use-cache is required for --no-conflict-misses or all-conflict-misses"
        assert (no_conflict_misses and not all_conflict_misses) or (
            not no_conflict_misses and all_conflict_misses
        ), "Cannot use both --no-conflict-misses and --all-conflict-misses at the same time!"
        logging.info(
            "Evaluate {}: Cache shape {}".format(
                "no_conflict_misses" if no_conflict_misses else "all_conflict_misses",
                emb_uvm.lxu_cache_weights.shape,
            )
        )
        num_cache_slots = emb_uvm.lxu_cache_weights.shape[0]
        for it, req in enumerate(requests_uvm):
            indices, offsets = req.unpack_2()
            num_uniq = 0
            all_inverse = []
            for t in range(T_uvm):
                uniq, inverse = indices[offsets[t * B] : offsets[(t + 1) * B]].unique(
                    return_inverse=True
                )
                all_inverse.append(inverse + num_uniq)
                num_uniq += uniq.numel()
            assert (
                num_cache_slots >= num_uniq
            ), "num_cache_slots < num_uniq: Please increase --cache-load-factor"

            # Intercept prefetch
            if no_conflict_misses:
                locations = np.random.choice(
                    np.arange(num_cache_slots), size=num_uniq, replace=False
                )
                locations = (
                    torch.from_numpy(locations).to(torch.int32).to(indices.device)
                )
                locations = locations.index_select(
                    dim=0, index=torch.concat(all_inverse)
                )
                assert (
                    locations.numel() == indices.numel()
                ), "The number of elements in locations and indices tensors are not the same!"
            else:
                locations = torch.full_like(
                    indices, -1, dtype=torch.int32, device=indices.device
                )
            emb_uvm.lxu_cache_locations_list.append(locations)
            emb_uvm.timesteps_prefetched.append(it)

    # pyre-ignore[53]
    def run_bench(indices: Tensor, offsets: Tensor, per_sample_weights: Tensor) -> None:
        if eval_conflict_misses:
            # Set uvm_cache_stats
            assert (
                emb_uvm.local_uvm_cache_stats.numel() == emb_uvm.uvm_cache_stats_size
            ), "The number of elements in the local_uvm_cache_stats tensor is not equal to its declared size!"
            # Use uvm_cache_stats_index::num_conflict_unique_misses
            emb_uvm.local_uvm_cache_stats[4] = 0 if no_conflict_misses else 1

        emb_uvm.forward(
            indices,
            offsets,
            per_sample_weights,
        )

    def _kineto_trace_handler(p: profile, phase: str) -> None:
        p.export_chrome_trace(
            trace_url.format(tbe_type="uvm", phase=phase, ospid=os.getpid())
        )

    # pyre-ignore[3]
    def context_factory(on_trace_ready: Callable[[profile], None]):
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    with context_factory(lambda p: _kineto_trace_handler(p, "fwd")):
        time_per_iter = benchmark_requests(
            requests_uvm,
            # pyre-fixme[6]: For 2nd argument expected `(Tensor, Tensor,
            #  Optional[Tensor]) -> Tensor` but got `(indices: Tensor, offsets: Tensor,
            #  per_sample_weights: Tensor) -> None`.
            run_bench,
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            num_warmups=warmup_runs,
        )
    logging.info(
        f"UVM Forward, B: {B}, "
        f"E: {E}, T: {T_uvm}, D: {D}, L: {L_uvm}, W: {weighted}, "
        f"BW: {read_write_bytes_uvm / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )
    print(
        f"|{uvm_tables}|{embedding_dim}|{read_write_bytes_uvm / time_per_iter / 1.0e9: .2f}|"
    )

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
        with context_factory(lambda p: _kineto_trace_handler(p, "gpu_fwd")):
            time_per_iter = benchmark_requests(
                requests_gpu,
                lambda indices, offsets, per_sample_weights: emb_gpu.forward(
                    indices,
                    offsets,
                    per_sample_weights,
                ),
                flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
                num_warmups=warmup_runs,
            )
        read_write_bytes_hbm = (
            output_size_multiplier * B * sum(Ds[T_uvm:])
            + param_size_multiplier * B * sum(Ds[T_uvm:]) * L
        )
        logging.info(
            f"GPU Forward, B: {B}, "
            f"E: {E}, T: {T_gpu}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {read_write_bytes_hbm / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
            f"T: {time_per_iter * 1.0e6:.0f}us"
        )

        with context_factory(lambda p: _kineto_trace_handler(p, "mixed_fwd")):
            time_per_iter = benchmark_requests(
                requests,
                lambda indices, offsets, per_sample_weights: emb_mixed.forward(
                    indices,
                    offsets,
                    per_sample_weights,
                ),
                flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
                num_warmups=warmup_runs,
            )
        read_write_bytes_total = read_write_bytes_uvm + read_write_bytes_hbm
        logging.info(
            f"Mixed Forward, B: {B}, "
            f"E: {E}, T_GPU: {T_gpu}, T_UVM: {T_uvm}, D: {D}, L_GPU: {L}, L_UVM: {L_uvm}, W: {weighted}, "
            f"BW: {read_write_bytes_total / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
            f"T: {time_per_iter * 1.0e6:.0f}us"
        )


@cli.command()
@TbeBenchClickInterface.common_options
@TbeBenchClickInterface.table_options
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-load-factor", default=0.2)
@click.option("--long-index", is_flag=True, default=False)
@click.option(
    "--reuse",
    default=0.1,  # Overriding the default value to 0.1, @TbeBenchClickInterface.common_options has default value 0.0
    help="The inter-batch indices reuse rate for the benchmark, default is 0.1.",
)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--cache-precision", type=SparseType, default=None)
@click.option(
    "--uvm-host-mapped",
    is_flag=True,
    default=False,
    help="Use host mapped UVM buffers in SSD-TBE (malloc+cudaHostRegister)",
)
@click.option(
    "--export-trace",
    is_flag=True,
    default=False,
    help="Enable export of trace for profiling. Default is False.",
)
@click.option(
    "--trace-url",
    type=str,
    default="{tbe_type}_tbe_{phase}_trace_{ospid}.json",
)
def cache(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    cache_algorithm: str,
    cache_load_factor: float,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
    warmup_runs: int,
    long_index: bool,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
    requests_data_file: Optional[str],
    tables: Optional[str],
    uvm_host_mapped: bool,
    cache_precision: SparseType,
    export_trace: bool,
    trace_url: str,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    cache_alg = CacheAlgorithm.LRU if cache_algorithm == "lru" else CacheAlgorithm.LFU

    assert torch.cuda.is_available(), "Cache benchmark requires CUDA device"
    if mixed:
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    emb_nc = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                EmbeddingLocation.MANAGED,
                ComputeDevice.CUDA,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        weights_precision=weights_precision,
        cache_precision=cache_precision,
        stochastic_rounding=stoc,
        uvm_host_mapped=uvm_host_mapped,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb_nc.init_embedding_weights_uniform(-0.0003, 0.0003)

    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                EmbeddingLocation.MANAGED_CACHING,
                ComputeDevice.CUDA,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
        cache_load_factor=cache_load_factor,
        cache_algorithm=cache_alg,
        uvm_host_mapped=uvm_host_mapped,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    logging.info(
        f"Embedding tables: {E * T} rows, {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        # pyre-fixme[58]: `*` is not supported for operand types `int` and
        #  `Union[np.floating[typing.Any], int]`.
        f"{B * T * L * D * param_size_multiplier / 1.0e9: .2f} GB"
    )

    requests = generate_requests(
        2 * iters,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weighted=weighted,
        requests_data_file=requests_data_file,
        tables=tables,
        index_dtype=torch.long,
        offset_dtype=torch.long,
    )
    warmup_requests, requests = requests[:iters], requests[iters:]
    grad_output = torch.randn(B, sum(Ds)).cuda()

    def _kineto_trace_handler(p: profile, phase: str) -> None:
        p.export_chrome_trace(
            trace_url.format(tbe_type="cache", phase=phase, ospid=os.getpid())
        )

    # pyre-ignore[3]
    def context_factory(on_trace_ready: Callable[[profile], None]):
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    with context_factory(lambda p: _kineto_trace_handler(p, "fwd_bwd")):
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb_nc(
                indices, offsets, per_sample_weights
            ).backward(grad_output),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            num_warmups=warmup_runs,
        )
    logging.info(
        f"ForwardBackward (UVM), B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    # warm up
    for req in warmup_requests:
        indices, offsets = req.unpack_2()
        emb.forward(indices, offsets)
    # get cache miss rate (forward and backward) and exchanged cache lines (prefetch)
    cache_misses = []
    exchanged_cache_lines = []
    NOT_FOUND = -1
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
        cache_misses.append((emb.lxu_cache_locations_list[0] == NOT_FOUND).sum().item())
        emb.forward(indices, offsets)
    logging.info(
        f"Exchanged cache lines -- mean: {sum(exchanged_cache_lines) / len(requests): .2f}, "
        f"max: {max(exchanged_cache_lines)}, min: {min(exchanged_cache_lines)}"
    )
    logging.info(
        f"Cache miss -- mean: {sum(cache_misses) / len(requests)}, "
        f"max: {max(cache_misses)}, min: {min(cache_misses)}"
    )

    # benchmark prefetch
    emb.reset_cache_states()
    for req in warmup_requests:
        indices, offsets = req.unpack_2()
        emb.forward(indices, offsets)
    # TODO: Add warmup_runs
    with context_factory(lambda p: _kineto_trace_handler(p, "prefetch")):
        prefetch_time, forward_backward_time = benchmark_pipelined_requests(
            requests,
            lambda indices, offsets, indices_weights: emb.prefetch(indices, offsets),
            lambda indices, offsets, indices_weights: emb.forward(
                indices, offsets, indices_weights
            ).backward(grad_output),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        )
    e2e_time = prefetch_time + forward_backward_time

    logging.info(
        f"ForwardBackward (LXU), reuse: {reuse}, alpha: {alpha}, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / e2e_time / 1.0e9: .2f} GB/s, "
        f"Tprefetch: {prefetch_time * 1.0e6:.0f}us, "
        f"{2 * sum(exchanged_cache_lines) * param_size_multiplier * D / prefetch_time / len(requests) / 1.0e9: .2f} GB/s, "
        f"Tfwdbwd: {forward_backward_time * 1.0e6:.0f}us, "
        f"{3 * param_size_multiplier * B * sum(Ds) * L / forward_backward_time / 1.0e9: .2f} GB/s, "
        f"Te2e: {e2e_time * 1.0e6:.0f}us, "
    )


@cli.command()
@click.option(
    "--embedding-dim-list",
    type=str,
    default="128",
    help="A comma-separated list of embedding dimensions for each table. Default is '128'. The number of embedding dimensions will determine the number of tables.",
)
@click.option(
    "--num-embeddings-list",
    type=str,
    default="100000",
    help="A comma-separated list of number of embeddings for each table, default is '100000'.",
)
@click.option(
    "--output-dtype",
    type=SparseType,
    default=SparseType.FP32,
    help="The output data type, default is FP32.",
)
@click.option(
    "--export-trace",
    is_flag=True,
    default=False,
    help="Enable export of trace for profiling. Default is False.",
)
@click.option(
    "--trace-url",
    type=str,
    default="{tbe_type}_tbe_{phase}_trace_{ospid}.json",
)
@TbeBenchClickInterface.common_options
@TbeBenchClickInterface.device_options
@TbeBenchClickInterface.vbe_options
def device_with_spec(  # noqa C901
    alpha: float,
    bag_size_list: str,
    bag_size_sigma_list: str,
    batch_size: int,
    embedding_dim_list: str,
    weights_precision: SparseType,
    cache_precision: Optional[SparseType],
    stoc: bool,
    iters: int,
    warmup_runs: int,
    managed: str,
    num_embeddings_list: str,
    reuse: float,
    row_wise: bool,
    weighted: bool,
    pooling: str,
    bounds_check_mode: int,
    flush_gpu_cache_size_mb: int,
    output_dtype: SparseType,
    export_trace: bool,
    trace_url: str,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    Ds = [int(D) for D in embedding_dim_list.split(",")]
    Es = [int(E) for E in num_embeddings_list.split(",")]
    T = len(Ds)

    use_variable_bag_sizes = bag_size_sigma_list != "None"

    if use_variable_bag_sizes:
        Ls = [int(mu) for mu in bag_size_list.split(",")]
        sigma_Ls = [int(sigma) for sigma in bag_size_sigma_list.split(",")]
        assert T == len(Ls) and T == len(sigma_Ls), (
            f"bag-size-list (length: {len(Ls)}) and bag-size-sigma-list "
            f"(length: {len(sigma_Ls)}) must have the same length as "
            f"embedding-dim-list (length: {T})"
        )
    else:
        Ls = [int(L) for L in bag_size_list.split(",")]
        assert T == len(Ls), (
            f"bag-size-list (length: {len(Ls)}) must have the same length as "
            f"embedding-dim-list (length: {T})"
        )

    assert T == len(Es), (
        f"num-embeddings-list (length: {len(Es)}) must have the same length as "
        f"embedding-dim-list (length: {T})"
    )

    assert T >= 1, "There must be at least one table"

    feature_requires_grad = None
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD if row_wise else OptimType.EXACT_ADAGRAD

    if managed == "device":
        # Currently, we set EmbeddingLocation.HOST for MTIA.
        managed_option = (
            EmbeddingLocation.DEVICE
            if get_available_compute_device() == ComputeDevice.CUDA
            else EmbeddingLocation.HOST
        )
    else:
        managed_option = EmbeddingLocation.MANAGED

    pooling_mode, do_pooling = get_pooling(pooling)

    if not do_pooling:
        ref_D = Ds[0]
        for D in Ds:
            assert (
                D == ref_D
            ), "All embedding dimensions must be the same for sequence TBE"

    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                e,
                d,
                managed_option,
                get_available_compute_device(),
            )
            for d, e in zip(Ds, Es)
        ],
        optimizer=optimizer,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=weights_precision,
        cache_precision=cache_precision,
        stochastic_rounding=stoc,
        output_dtype=output_dtype,
        pooling_mode=pooling_mode,
        bounds_check_mode=BoundsCheckMode(bounds_check_mode),
    )
    emb = emb.to(get_device())

    if weights_precision == SparseType.INT8:
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0

    # Generate a request for each table then combine
    all_requests = {
        "indices": [[] for _ in range(iters)],
        "offsets": [[] for _ in range(iters)],
        "weights": [[] for _ in range(iters)],
    }
    # row = iter, column = tensor
    for t, e in enumerate(Es):
        # (indices, offsets, weights)
        requests = generate_requests(
            iters,
            B,
            1,
            Ls[t],
            e,
            reuse=reuse,
            alpha=alpha,
            weighted=weighted,
            # pyre-fixme[61]: `sigma_Ls` is undefined, or not always defined.
            sigma_L=sigma_Ls[t] if use_variable_bag_sizes else None,
            zipf_oversample_ratio=3 if Ls[t] > 5 else 5,
            use_cpu=get_available_compute_device() == ComputeDevice.CPU,
            index_dtype=torch.long,
            offset_dtype=torch.long,
        )
        for i, req in enumerate(requests):
            indices, offsets, weights = req.unpack_3()
            all_requests["indices"][i].append(indices)
            if t > 0:
                offsets = offsets[1:]  # remove the first element
                offsets += all_requests["offsets"][i][t - 1][-1]
            all_requests["offsets"][i].append(offsets)
            all_requests["weights"][i].append(weights)

    prev_indices_len = -1
    requests = []
    for i in range(iters):
        indices = torch.concat(all_requests["indices"][i])
        if prev_indices_len == -1:
            prev_indices_len = indices.numel()
        assert (
            prev_indices_len == indices.numel()
        ), "Number of indices for every iteration must be the same"
        offsets = torch.concat(all_requests["offsets"][i])
        if weighted:
            weights = torch.concat(all_requests["weights"][i])
        else:
            weights = None
        requests.append(TBERequest(indices, offsets, weights))

    del all_requests

    assert len(requests) == iters

    sum_DLs = sum([d * l for d, l in zip(Ds, Ls)])
    if do_pooling:
        read_write_bytes = (
            output_size_multiplier * B * sum(Ds) + param_size_multiplier * B * sum_DLs
        )
    else:
        read_write_bytes = (
            output_size_multiplier * B * sum_DLs + param_size_multiplier * B * sum_DLs
        )

    if use_variable_bag_sizes:
        # pyre-ignore [61]
        Ls_str = f"mu {Ls} sigma {sigma_Ls}"
    else:
        Ls_str = f"{Ls}"

    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * sum_DLs * param_size_multiplier / 1.0e9: .2f} GB"
    )

    def _kineto_trace_handler(p: profile, phase: str) -> None:
        p.export_chrome_trace(
            trace_url.format(tbe_type="split", phase=phase, ospid=os.getpid())
        )

    # pyre-ignore[3]
    def context_factory(on_trace_ready: Callable[[profile], None]):
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    with context_factory(lambda p: _kineto_trace_handler(p, "fwd")):
        # forward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb.forward(
                indices,
                offsets,
                per_sample_weights,
                feature_requires_grad=feature_requires_grad,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            num_warmups=warmup_runs,
        )
    logging.info(
        f"Forward, B: {B}, "
        f"Es: {Es}, T: {T}, Ds: {Ds}, Ls: {Ls_str}, W: {weighted}, a: {alpha}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    if output_dtype == SparseType.INT8:
        # backward bench not representative
        return

    if do_pooling:
        grad_output = torch.randn(B, sum(Ds)).to(get_device())
    else:
        # Obtain B * L from indices len
        # pyre-ignore[19]
        # pyre-fixme[61]: `D` is undefined, or not always defined.
        grad_output = torch.randn(requests[0].indices.numel(), D).to(get_device())

    with context_factory(lambda p: _kineto_trace_handler(p, "fwd_bwd")):
        # backward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb(
                indices,
                offsets,
                per_sample_weights,
                feature_requires_grad=feature_requires_grad,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            bwd_only=True,
            grad=grad_output,
            num_warmups=warmup_runs,
        )
    logging.info(
        f"Backward, B: {B}, Es: {Es}, T: {T}, Ds: {Ds}, Ls: {Ls_str}, a: {alpha}, "
        f"BW: {2 * read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option(
    "--batch-size-list",
    type=str,
    required=True,
    help="A comma separated list of batch sizes (B) for each table.",
)
@click.option(
    "--embedding-dim-list",
    type=str,
    required=True,
    help="A comma separated list of embedding dimensions (D) for each table.",
)
@click.option(
    "--bag-size-list",
    type=str,
    required=True,
    help="A comma separated list of bag sizes (L) for each table.",
)
@click.option(
    "--bag-size-sigma-list",
    type=str,
    default="None",
    help="A comma separated list of standard deviations for generating bag sizes per table. "
    "If 'None' is set, bag sizes are fixed per table.",
)
@click.option(
    "--num-embeddings-list",
    type=str,
    required=True,
    help="A comma separated list of number of embeddings (E) for each table.",
)
@click.option(
    "--alpha-list",
    type=str,
    default="None",
    help="A comma separated list of ZipF-alpha values for index distribution for each table. "
    "If 'None' is set, uniform distribution is used.",
)
@click.option(
    "--num-tables",
    type=int,
    required=True,
    help="The number of tables.",
)
@click.option(
    "--weighted",
    is_flag=True,
    default=False,
    help="Whether the table is weighted or not",
)
@click.option(
    "--print-kernel-summary",
    is_flag=True,
    default=False,
    help="Whether the table is weighted or not",
)
@click.option("--ssd", is_flag=True, default=False)
@click.option(
    "--ssd-prefix", type=str, default="/tmp/ssd_benchmark", help="SSD directory prefix"
)
@click.option(
    "--merge-output",
    is_flag=True,
    default=False,
    help="Write VBE outputs to one tensor.",
)
@click.option(
    "--merge-output-num-ranks",
    type=int,
    default=8,
    help="Number of ranks for merged output allocation.",
)
@click.option(
    "--merge-output-num-tbe-ops",
    type=int,
    default=2,
    help="Number of TBE ops for merged output allocation.",
)
@TBEBenchmarkingConfigLoader.options
@EmbeddingOpsCommonConfigLoader.options
@click.pass_context
def vbe(  # noqa: C901
    context: click.Context,
    batch_size_list: str,
    embedding_dim_list: str,
    bag_size_list: str,
    bag_size_sigma_list: str,
    num_embeddings_list: str,
    alpha_list: str,
    num_tables: int,
    weighted: bool,
    print_kernel_summary: bool,
    ssd: bool,
    ssd_prefix: str,
    merge_output: bool,
    merge_output_num_ranks: int,
    merge_output_num_tbe_ops: int,
    # pyre-ignore[2]
    **kwargs,
) -> None:
    """
    A benchmark function to evaluate variable batch-size table-batched
    embedding (VBE) kernels for both forward and backward. Unlike TBE,
    batch sizes can be specified per table for VBE.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # Load general TBE benchmarking configuration from cli arguments
    benchconfig = TBEBenchmarkingConfigLoader.load(context)
    if benchconfig.num_requests != benchconfig.iterations:
        raise ValueError("--bench-num-requests is not supported.")

    if benchconfig.flush_gpu_cache_size_mb != 0:
        raise ValueError("--bench-flush-gpu-cache-size is not supported.")

    # Load common embedding op configuration from cli arguments
    embconfig = EmbeddingOpsCommonConfigLoader.load(context)
    if embconfig.uvm_host_mapped:
        raise ValueError("--emb-uvm-host-mapped is not supported.")

    T = num_tables
    alphas: list[float] = []
    if alpha_list in ["None", ""]:
        alphas = [0.0] * T
    elif "," not in alpha_list:
        alphas = [float(alpha_list)] * T
    else:
        alphas = [float(alpha) for alpha in alpha_list.split(",")]
    Bs = [int(v) for v in batch_size_list.split(",")]
    Ds = [int(v) for v in embedding_dim_list.split(",")]
    Ls = [int(v) for v in bag_size_list.split(",")]
    sigma_Ls = (
        [int(sigma) for sigma in bag_size_sigma_list.split(",")]
        if bag_size_sigma_list != "None"
        else [0] * T
    )
    Es = [int(v) for v in num_embeddings_list.split(",")]

    # All these variables must have the same length.
    assert T == len(alphas)
    assert T == len(Bs)
    assert T == len(Ds)
    assert T == len(Ls)
    assert T == len(sigma_Ls)
    assert T == len(Es)

    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
    managed_option = (
        EmbeddingLocation.DEVICE
        if get_available_compute_device() == ComputeDevice.CUDA
        else EmbeddingLocation.HOST
    )

    common_split_args: dict[str, Any] = {
        "weights_precision": embconfig.weights_dtype,
        "stochastic_rounding": embconfig.stochastic_rounding,
        "output_dtype": embconfig.output_dtype,
        "pooling_mode": embconfig.pooling_mode,
        "bounds_check_mode": embconfig.bounds_check_mode,
        "optimizer": optimizer,
        "learning_rate": 0.1,
        "eps": 0.1,
        "feature_table_map": list(range(T)),
    }

    if ssd:
        cache_set = max(T * max(Bs), 1)
        tempdir = tempfile.mkdtemp(prefix=ssd_prefix)
        emb = SSDTableBatchedEmbeddingBags(
            [(E, D) for E, D in zip(Es, Ds)],
            cache_sets=cache_set,
            ssd_storage_directory=tempdir,
            ssd_cache_location=EmbeddingLocation.DEVICE,
            ssd_rocksdb_shards=8,
            **common_split_args,
        )
    else:
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    D,
                    managed_option,
                    get_available_compute_device(),
                )
                for E, D in zip(Es, Ds)
            ],
            cache_precision=embconfig.cache_dtype,
            **common_split_args,
        )
    emb = emb.to(get_device())
    all_requests = {
        "indices": [[] for _ in range(benchconfig.iterations)],
        "offsets": [[] for _ in range(benchconfig.iterations)],
        "weights": [[] for _ in range(benchconfig.iterations)],
    }
    for t, (E, B, L, sigma_L, alpha) in enumerate(zip(Es, Bs, Ls, sigma_Ls, alphas)):
        # Generate a request for a single table.
        local_requests = generate_requests(
            benchconfig.iterations,
            B,
            1,
            L,
            E,
            alpha=alpha,
            weighted=weighted,
            sigma_L=sigma_L,
            zipf_oversample_ratio=3 if L > 5 else 5,
            use_cpu=get_available_compute_device() == ComputeDevice.CPU,
            index_dtype=torch.long,
            offset_dtype=torch.long,
        )

        # Store requests for each table in all_requests.
        for i, req in enumerate(local_requests):
            indices, offsets, weights = req.unpack_3()
            all_requests["indices"][i].append(indices)
            if t > 0:
                offsets = offsets[1:]  # remove the first element
                offsets += all_requests["offsets"][i][t - 1][-1]
            all_requests["offsets"][i].append(offsets)
            all_requests["weights"][i].append(weights)

    # pyre-ignore[53]
    def _kineto_trace_handler(
        p: profile, emb_op_type: str = "vbe", print_summary: bool = False
    ) -> None:
        p.export_chrome_trace(
            benchconfig.trace_url.format(emb_op_type=emb_op_type, ospid=os.getpid())
        )
        if print_summary:
            print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    emb_op_type = "vbe"

    # pyre-ignore[3, 53]
    def context_factory(on_trace_ready: Callable[[profile], None]):
        return (
            profile(on_trace_ready=on_trace_ready)
            if benchconfig.export_trace
            else nullcontext()
        )

    # Combine the requests for all tables by
    requests = [
        (
            torch.concat(all_requests["indices"][i]),
            torch.concat(all_requests["offsets"][i]),
            torch.concat(all_requests["weights"][i]) if weighted else None,
        )
        for i in range(benchconfig.iterations)
    ]

    del all_requests

    batch_size_per_feature_per_rank = [[B] for B in Bs]
    out = None
    out_offsets = None
    if merge_output:
        batch_size_per_feature_per_rank, out, out_offsets = (
            generate_merged_output_and_offsets(
                Ds,
                Bs,
                embconfig.output_dtype.as_dtype(),
                get_device(),
                num_ranks=merge_output_num_ranks,
                num_tbe_ops=merge_output_num_tbe_ops,
            )
        )

    with context_factory(
        lambda p: _kineto_trace_handler(p, emb_op_type, print_kernel_summary)
    ):
        fwd_time_sec, bwd_time_sec = benchmark_vbe(
            requests,
            func=lambda indices, offsets, per_sample_weights: emb.forward(
                indices,
                offsets,
                per_sample_weights,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                vbe_output=out,
                vbe_output_offsets=out_offsets,
            ),
            num_warmups=benchconfig.warmup_iterations,
        )
    logging.info(
        f"T: {T}, Bs: {Bs}, Ds: {Ds}, Ls: {Ls}, Es: {Es}, alphas: {alphas}\n"
        f"fwd: {fwd_time_sec * 1.0e6:.0f}us, bwd: {bwd_time_sec * 1.0e6:.0f}us"
    )


@cli.command()
@TbeBenchClickInterface.common_options
@TbeBenchClickInterface.device_options
@click.option(
    "--weighted-num-requires-grad",
    type=int,
    default=None,
    help="Number of tables requiring gradient computation for weighted embeddings. Default is None.",
)
@click.option(
    "--output-dtype",
    type=SparseType,
    default=SparseType.FP32,
    help="Data type of the output embeddings. Default is FP32.",
)
@click.option(
    "--export-trace",
    is_flag=True,
    default=False,
    help="Enable export of trace for profiling. Default is False.",
)
@click.option(
    "--trace-url",
    type=str,
    default="bench_tbe_{device}_{phase}_trace_{emb_loc}_{ospid}.json",
    help="URL for trace file. Default is bench_tbe_{device}_{phase}_trace_{embbeding_location}_{ospid}.json.",
)
@click.option(
    "--optimizer",
    type=OptimType,
    default=OptimType.EXACT_ROWWISE_ADAGRAD,
    help="Optimizer type for embedding updates. Default is EXACT_ROWWISE_ADAGRAD.",
)
@click.option(
    "--indices-file",
    type=str,
    default=None,
    help="Path to the indices file. Default is None.",
)
@click.option(
    "--offsets-file",
    type=str,
    default=None,
    help="Path to the offsets file. Default is None.",
)
@click.option(
    "--specs-file",
    type=str,
    default=None,
    help="Path to the specs.pt file containing embedding specs. "
    "Expected format: dict with 'embedding_specs' key containing list of "
    "(num_embeddings, embedding_dim, location, compute_device) tuples. "
    "When provided, overrides --num-embeddings, --embedding-dim, --num-tables.",
)
@click.option(
    "--feature-table-map-file",
    type=str,
    default=None,
    help="Comma-separated feature table map. Default is None (uses range(T)).",
)
@click.option(
    "--batch-size-per-feature-per-rank-file",
    type=str,
    default=None,
    help="a file or Comma-separated list of batch sizes per table for VBE mode. "
    "Each value represents the batch size for a table. "
    "If provided, enables Variable Batch Embedding (VBE) mode. "
    "Example: '128,256,512' for 3 tables with different batch sizes.",
)
@click.option(
    "--compare",
    is_flag=True,
    type=bool,
    default=False,
    help="Run CPU benchmark",
)
def device_from_files(  # noqa C901
    alpha: float,  # unused
    batch_size: int,  # unused
    reuse: float,  # unused
    cache_precision: SparseType,  # unused
    managed: str,  # unused
    row_wise: bool,  # unused
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
    warmup_runs: int,
    weighted: bool,
    pooling: str,
    weighted_num_requires_grad: Optional[int],
    bounds_check_mode: int,
    flush_gpu_cache_size_mb: int,
    output_dtype: SparseType,
    optimizer: OptimType,
    export_trace: bool,
    trace_url: str,
    compare: bool,
    indices_file: str,
    offsets_file: str,
    specs_file: str,
    feature_table_map_file: str,
    batch_size_per_feature_per_rank_file: Optional[str],
) -> None:
    """
    Benchmark that compares TBE performance between current accelerator (GPU/CUDA/MTIA)
    and CPU. This helps understand the performance difference between accelerated
    and CPU-based embedding operations.
    """
    assert specs_file is not None, "Required embedding specs"
    assert indices_file is not None, "Required indices file"
    assert offsets_file is not None, "Required offsets file"
    assert os.path.exists(
        specs_file
    ), f"Expect a specs_file file but file not found: {specs_file}"
    assert os.path.exists(
        indices_file
    ), f"Expect a indices_file file but file not found: {indices_file}"
    assert os.path.exists(
        offsets_file
    ), f"Expect a offsets_file file but file not found: {offsets_file}"

    # Determine current accelerator
    accelerator_device = torch.accelerator.current_accelerator(check_available=True)

    if accelerator_device is None and not compare:
        raise AssertionError(
            "No accelerator available. Please specify --compare to run on CPU"
        )
    map_location = torch.device("cpu") if compare else accelerator_device

    # Load specs from file
    embedding_specs: list[Tuple[int, int, EmbeddingLocation, ComputeDevice]] = (
        torch.load(specs_file, weights_only=False)
    )
    logging.info(f"Loaded embedding_specs: {embedding_specs}")
    _Es: list[int]
    _Ds: list[int]
    emb_loc: list[EmbeddingLocation]
    compute_device: list[ComputeDevice]
    _Es, _Ds, emb_loc, compute_device = zip(*embedding_specs)

    # Determine location suffix for trace URL based on embedding locations
    location_suffix: str = "device"
    if EmbeddingLocation.MANAGED_CACHING in emb_loc:
        location_suffix = "cache"
    elif EmbeddingLocation.MANAGED in emb_loc:
        location_suffix = "uvm"

    Es: list[int] = _Es
    Ds: list[int] = _Ds
    assert len(Es) == len(
        Ds
    ), f"Number of embeddings and dimensions mismatched {len(Es)} != {len(Ds)}"
    feature_table_map = list(range(len(Es)))
    if feature_table_map_file is not None:
        assert os.path.exists(
            feature_table_map_file
        ), f"Expect a feature_table_map file but file not found: {feature_table_map_file}"
        feature_table_map = torch.load(feature_table_map_file, weights_only=False)
        Es = [_Es[t] for t in feature_table_map]
        Ds = [_Ds[t] for t in feature_table_map]
        logging.info(f"Loaded feature_table_map: {feature_table_map}")
    else:
        logging.info(
            f"feature_table_map file is not provided, set to f{feature_table_map}"
            + "Please note that this assumption may be wrong. If you see boundscheck error, "
            + "provide feature_table_map."
        )

    T: int = len(Es)
    num_features: int = len(feature_table_map)
    assert (
        T == num_features
    ), f"Number of features mismatched, found {num_features} from feature_table_map but T = {T}"
    logging.info(f"_Es: {_Es} _Ds: {_Ds} num_features: {T}")

    indices_tensor: Tensor = torch.load(
        indices_file, map_location=map_location, weights_only=True
    )
    offsets_tensor: Tensor = torch.load(
        offsets_file, map_location=map_location, weights_only=True
    )
    per_sample_weights_tensor: Optional[Tensor] = (
        None if not weighted else torch.randn(indices_tensor.size())
    )
    logging.info(f"Loaded indices: {indices_tensor.shape} on {indices_tensor.device}")
    logging.info(f"Loaded offsets: {offsets_tensor.shape} on {offsets_tensor.device}")

    total_B: int = offsets_tensor.numel() - 1
    assert total_B > 0, f"Invalid offsets tensor: total_B={total_B} must be positive"
    avg_E: float = float(np.mean(Es))
    avg_D: float = float(np.mean(Ds))
    avg_L: float = float(indices_tensor.numel() / total_B)

    batch_size_per_feature_per_rank: Optional[list[list[int]]] = None
    if batch_size_per_feature_per_rank_file is not None:
        batch_size_per_feature_per_rank = torch.load(
            batch_size_per_feature_per_rank_file, weights_only=True
        ).tolist()
        assert (
            len(batch_size_per_feature_per_rank) == T
        ), f"Number of features mismatched, found {len(batch_size_per_feature_per_rank)} from batch_size_per_feature_per_rank but T = {T}"
    else:
        assert (
            total_B % T == 0
        ), f"Expect constant batch size but total_B = {total_B} and T = {T}"

    Bs = (
        [sum(b_r) for b_r in batch_size_per_feature_per_rank]
        if batch_size_per_feature_per_rank is not None
        else [total_B // T] * T
    )

    pooling_mode, do_pooling = get_pooling(pooling)

    # Calculate read/write bytes for bandwidth calculation
    read_write_bytes: float = compute_read_write_bytes(
        Es, Bs, Ds, avg_L, do_pooling, output_dtype, weights_precision
    )

    common_args: Dict[str, Any] = {
        "weights_precision": weights_precision,
        "stochastic_rounding": stoc,
        "output_dtype": output_dtype,
        "pooling_mode": pooling_mode,
        "bounds_check_mode": BoundsCheckMode(bounds_check_mode),
        "optimizer": optimizer,
        "learning_rate": 0.1,
        "eps": 0.1,
        "feature_table_map": feature_table_map,
    }

    # TODO: Refactor to only use one request over multiple iterations
    def gen_requests(
        indices_tensor: torch.Tensor,
        offsets_tensor: torch.Tensor,
        device: torch.device,
        per_sample_weights_tensor: Optional[torch.Tensor],
        batch_size_per_feature_per_rank: Optional[list[list[int]]],
    ) -> list[TBERequest]:
        rs = []
        indices = indices_tensor.to(device)
        offsets = offsets_tensor.to(device)
        per_sample_weights = (
            per_sample_weights_tensor.to(device)
            if per_sample_weights_tensor is not None
            else None
        )
        for _ in range(iters):
            rs.append(
                TBERequest(
                    indices,
                    offsets,
                    per_sample_weights,
                    batch_size_per_feature_per_rank,
                )
            )
        return rs

    results: Dict[str, Dict[str, float]] = {}

    def _kineto_trace_handler(p: profile, device_name: str, phase: str) -> None:
        trace_filename = trace_url.format(
            device=device_name, phase=phase, emb_loc=location_suffix, ospid=os.getpid()
        )
        p.export_chrome_trace(trace_filename)

    def context_factory(
        on_trace_ready: Callable[[profile], None],
    ) -> profile | nullcontext[None]:
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    # Helper function to run benchmark on a specific device
    def run_benchmark_on_device(device: torch.device) -> Dict[str, float]:
        device_name = device.type.upper()
        use_cpu = device_name == "CPU"

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Running benchmark on {device_name}")
        logging.info(f"{'=' * 60}")

        # Create TBE for this device
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    e,
                    d,
                    loc if not use_cpu else EmbeddingLocation.HOST,
                    dev if not use_cpu else ComputeDevice.CPU,
                )
                for (e, d, loc, dev) in zip(_Es, _Ds, emb_loc, compute_device)
            ],
            **common_args,
        )
        emb = emb.to(device)
        if weights_precision in [SparseType.INT8, SparseType.NFP8]:
            emb.init_embedding_weights_uniform(-0.0003, 0.0003)

        requests = gen_requests(
            indices_tensor,
            offsets_tensor,
            device,
            per_sample_weights_tensor,
            batch_size_per_feature_per_rank,
        )

        # Prepare feature_requires_grad if needed
        feature_requires_grad: Optional[Tensor] = None
        if weighted_num_requires_grad:
            weighted_requires_grad_tables = np.random.choice(
                T, replace=False, size=(weighted_num_requires_grad,)
            ).tolist()
            feature_requires_grad = (
                torch.tensor(
                    [1 if t in weighted_requires_grad_tables else 0 for t in range(T)]
                )
                .to(device)
                .int()
            )

        with context_factory(lambda p: _kineto_trace_handler(p, device_name, "fwd")):
            fwd_time_per_iter = benchmark_requests_with_spec(
                requests,
                lambda indices, offsets, psw, batch_size_per_feature_per_rank: emb.forward(
                    indices,
                    offsets,
                    psw,
                    feature_requires_grad=feature_requires_grad,
                    batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                ),
                flush_gpu_cache_size_mb=(flush_gpu_cache_size_mb if not use_cpu else 0),
                num_warmups=warmup_runs,
            )

        fwd_time_us = fwd_time_per_iter * 1.0e6
        fwd_bw_gbs = read_write_bytes / fwd_time_per_iter / 1.0e9
        logging.info(
            f"[{device_name}] Forward, total B: {total_B}, "
            f"E: {avg_E:.0f}, T: {T}, D: {avg_D:.0f}, L: {avg_L:.1f}, W: {weighted}, "
            f"BW: {fwd_bw_gbs: .2f} GB/s, "
            f"T: {fwd_time_us:.0f}us"
        )

        result: Dict[str, float] = {
            "fwd_time_us": fwd_time_us,
            "fwd_bw_gbs": fwd_bw_gbs,
        }

        if output_dtype == SparseType.INT8:
            # backward bench not representative for INT8 output
            return result

        with context_factory(
            lambda p: _kineto_trace_handler(p, device_name, "fwd_bwd")
        ):
            bwd_time_per_iter = benchmark_requests_with_spec(
                requests,
                lambda indices, offsets, psw, batch_size_per_feature_per_rank: emb.forward(
                    indices,
                    offsets,
                    psw,
                    feature_requires_grad=feature_requires_grad,
                    batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                ),
                flush_gpu_cache_size_mb=(flush_gpu_cache_size_mb if not use_cpu else 0),
                num_warmups=warmup_runs,
                bwd_only=True,
            )
        bwd_time_us = bwd_time_per_iter * 1.0e6
        bwd_bw_gbs = 2 * read_write_bytes / bwd_time_per_iter / 1.0e9
        logging.info(
            f"[{device_name}] Backward, total B: {total_B}, "
            f"E: {avg_E:.0f}, T: {T}, D: {avg_D:.0f}, L: {avg_L:.1f}, W: {weighted}, "
            f"BW: {bwd_bw_gbs: .2f} GB/s, "
            f"T: {bwd_time_us:.0f}us"
        )
        result["bwd_time_us"] = bwd_time_us
        result["bwd_bw_gbs"] = bwd_bw_gbs
        return result

    # Run CPU benchmark
    if compare:
        cpu_device = torch.device("cpu")
        results["CPU"] = run_benchmark_on_device(cpu_device)

    # Run accelerator benchmark if available
    if accelerator_device is not None:
        accel_name = accelerator_device.type.upper()
        if emb_loc is None:
            dev, loc = get_compute_device(accelerator_device)
            emb_loc = [loc] * len(_Es)
            compute_device = [dev] * len(_Es)
        results[accel_name] = run_benchmark_on_device(accelerator_device)

    # Print comparison summary
    logging.info(f"\n{'=' * 60}")
    logging.info("COMPARISON SUMMARY")
    logging.info(f"{'=' * 60}")
    logging.info(
        f"Configuration: B={total_B}, E={avg_E:.0f}, T={T}, "
        f"D={avg_D:.0f}, L={avg_L:.1f}, W={weighted}"
    )
    logging.info(
        f"Weights precision: {weights_precision}, Output dtype: {output_dtype}"
    )
    logging.info("")

    # Print table header
    header = f"{'Device':<15} {'Fwd Time (us)':<15} {'Fwd BW (GB/s)':<15}"
    if output_dtype != SparseType.INT8:
        header += f" {'Bwd Time (us)':<15} {'Bwd BW (GB/s)':<15}"
    logging.info(header)
    logging.info("-" * len(header))

    for device_name, metrics in results.items():
        row = f"{device_name:<15} {metrics['fwd_time_us']:<15.2f} {metrics['fwd_bw_gbs']:<15.2f}"
        if output_dtype != SparseType.INT8:
            row += f" {metrics.get('bwd_time_us', 0):<15.2f} {metrics.get('bwd_bw_gbs', 0):<15.2f}"
        logging.info(row)

    # Print speedup if accelerator is available
    if compare and accelerator_device is not None:
        accel_name = accelerator_device.type.upper()
        fwd_speedup = results["CPU"]["fwd_time_us"] / results[accel_name]["fwd_time_us"]
        logging.info("")
        logging.info(f"Forward Speedup ({accel_name} vs CPU): {fwd_speedup:.2f}x")

        if output_dtype != SparseType.INT8:
            bwd_speedup = (
                results["CPU"]["bwd_time_us"] / results[accel_name]["bwd_time_us"]
            )
            logging.info(f"Backward Speedup ({accel_name} vs CPU): {bwd_speedup:.2f}x")


if __name__ == "__main__":
    cli()
