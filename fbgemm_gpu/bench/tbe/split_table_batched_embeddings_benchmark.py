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
from typing import Any, Callable, Dict, List, Optional

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
    benchmark_vbe,
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


@cli.command()
# recommended value: alpha=1.15 for training and alpha=1.09 for inference
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--cache-precision", type=SparseType, default=None)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option(
    "--managed",
    default="device",
    type=click.Choice(["device", "managed", "managed_caching"], case_sensitive=False),
)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--pooling", type=str, default="sum")
@click.option("--weighted-num-requires-grad", type=int, default=None)
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.NONE.value)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--dense", is_flag=True, default=False)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP32)
@click.option("--indices-dtype", type=click.Choice(["32", "64"]), default="64")
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
@click.option("--export-trace", is_flag=True, default=False)
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

    common_split_args: Dict[str, Any] = {
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

    if weights_precision == SparseType.INT8:
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
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
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
        f"Backward, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {2 * read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.1)
@click.option("--uvm-tables", default=1)
@click.option("--uvm-bag-size", default=1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
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
def uvm(
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
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-load-factor", default=0.2)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--long-index", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
@click.option(
    "--uvm-host-mapped",
    is_flag=True,
    default=False,
    help="Use host mapped UVM buffers in SSD-TBE (malloc+cudaHostRegister)",
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
@click.option("--alpha", default=1.0)
@click.option(
    "--bag-size-list",
    type=str,
    default="20",
)
@click.option(
    "--bag-size-sigma-list",
    type=str,
    default="None",
    help="A list of bag size standard deviations for generating bag sizes "
    "(one std per table). If set, the benchmark will treat --bag-size-list as a "
    "list of bag size means.",
)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim-list", type=str, default="128")
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--cache-precision", type=SparseType, default=None)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option("--managed", default="device")
@click.option("--num-embeddings-list", type=str, default="100000")
@click.option("--reuse", default=0.0)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--pooling", type=str, default="sum")
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.NONE.value)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP32)
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
        f"Es: {Es}, T: {T}, Ds: {Ds}, Ls: {Ls_str}, W: {weighted}, "
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
        f"Backward, B: {B}, Es: {Es}, T: {T}, Ds: {Ds}, Ls: {Ls_str}, "
        f"BW: {2 * read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--batch-sizes", default="128000,1280")
@click.option("--embedding-dims", default="1024,16")
@click.option("--bag-sizes", default="5,2")
@click.option("--nums-embeddings", default="10000,1000000")
@click.option("--num-tables", default=2)
@click.option("--iters", default=100)
def vbe(
    batch_sizes: str,
    embedding_dims: str,
    bag_sizes: str,
    nums_embeddings: str,
    num_tables: int,
    iters: int,
) -> None:
    """
    A benchmark function to evaluate variable batch-size table-batched
    embedding (VBE) kernels for both forward and backward. Unlike TBE,
    batch sizes can be specified per table for VBE.

    Args:
        batch_sizes (str):
            A comma separated list of batch sizes for each table.

        embedding_dims (str):
            A comma separated list of embedding dimensions for each table.

        bag_sizes (str):
            A comma separated list of bag sizes for each table.

        num_embeddings (str):
            A comma separated list of number of embeddings for each table.

        num_tables (int):
            The number of tables.

        iters (int):
            The number of iterations to run the benchmark for.
    """

    torch.manual_seed(42)
    Bs = [int(v) for v in batch_sizes.split(",")]
    Ds = [int(v) for v in embedding_dims.split(",")]
    Ls = [int(v) for v in bag_sizes.split(",")]
    Es = [int(v) for v in nums_embeddings.split(",")]
    T = num_tables

    # All these variables must have the same length.
    assert T == len(Bs)
    assert T == len(Ds)
    assert T == len(Ls)
    assert T == len(Es)

    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
    managed_option = (
        EmbeddingLocation.DEVICE
        if get_available_compute_device() != ComputeDevice.CPU
        else EmbeddingLocation.HOST
    )
    pooling_mode = PoolingMode.SUM

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
        optimizer=optimizer,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=SparseType.FP32,
        stochastic_rounding=False,
        output_dtype=SparseType.FP32,
        pooling_mode=pooling_mode,
        bounds_check_mode=BoundsCheckMode(BoundsCheckMode.NONE.value),
    ).to(get_device())

    lengths_list: List[torch.Tensor] = []
    num_values_per_table: List[int] = []
    for t, B in enumerate(Bs):
        L = Ls[t]
        # Assume a uniformly distributed random number in [0, 2L)
        # On average it should be L.
        lengths_list.append(
            torch.randint(
                low=0, high=2 * L, size=(B,), dtype=torch.int64, device=get_device()
            )
        )

        # num_values is used later.
        # Note: sum().tolist() returns a scalar value.
        # pyre-ignore
        num_values: int = torch.sum(lengths_list[-1]).tolist()
        num_values_per_table.append(num_values)

    lengths = torch.cat(lengths_list, 0)

    # Convert lengths into offsets.
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths).long()

    # Set up values.
    values_list: List[torch.Tensor] = []
    for t, E in enumerate(Es):
        # Assuming that an index distribution is uniform [0, E)
        values_list.append(
            torch.randint(
                low=0,
                high=E,
                size=(num_values_per_table[t],),
                dtype=torch.int32,
                device=get_device(),
            )
        )
    values = torch.cat(values_list, 0).long()

    requests = [
        (
            values,
            offsets,
        )
        for _ in range(iters)
    ]

    fwd_time_sec, bwd_time_sec = benchmark_vbe(
        requests,
        func=lambda indices, offsets: emb.forward(
            indices,
            offsets,
            batch_size_per_feature_per_rank=[[B] for B in Bs],
        ),
    )
    logging.info(
        f"T: {T}, Bs: {Bs}, Ds: {Ds}, Ls: {Ls}, Es: {Es}\n"
        f"fwd: {fwd_time_sec * 1.0e6:.0f}us, bwd: {bwd_time_sec * 1.0e6:.0f}us"
    )


if __name__ == "__main__":
    cli()
