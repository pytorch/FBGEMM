#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
import fbgemm_gpu
import numpy as np

import torch

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_embedding_utils import generate_requests, get_device, round_up

from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import Tensor

haveAIBench = False
try:
    from aibench_observer.utils.observer import emitMetric

    haveAIBench = True
except Exception:
    haveAIBench = False


# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import (
        benchmark_pipelined_requests,
        benchmark_requests,
        benchmark_requests_refer,
        benchmark_torch_function,
        benchmark_vbe,
        fill_random_scale_bias,
    )
else:
    from fbgemm_gpu.bench.bench_utils import (
        benchmark_pipelined_requests,
        benchmark_requests,
        benchmark_requests_refer,
        benchmark_torch_function,
        benchmark_vbe,
        fill_random_scale_bias,
    )


logging.basicConfig(level=logging.DEBUG)


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
@click.option("--pooling", type=str, default="sum")
@click.option("--weighted-num-requires-grad", type=int, default=None)
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.NONE.value)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--dense", is_flag=True, default=False)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP32)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
def device(  # noqa C901
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
    bounds_check_mode: int,
    flush_gpu_cache_size_mb: int,
    dense: bool,
    output_dtype: SparseType,
    requests_data_file: Optional[str],
    tables: Optional[str],
) -> None:
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

    if dense:
        emb = DenseTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                )
                for d in Ds
            ],
            pooling_mode=pooling_mode,
            use_cpu=not torch.cuda.is_available(),
        )
    else:
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                    managed_option,
                    ComputeDevice.CUDA
                    if torch.cuda.is_available()
                    else ComputeDevice.CPU,
                )
                for d in Ds
            ],
            optimizer=optimizer,
            learning_rate=0.1,
            eps=0.1,
            weights_precision=weights_precision,
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

    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * sum(Ds) * L * param_size_multiplier / 1.0e9: .2f} GB"
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

    # forward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb.forward(
            indices.long(),
            offsets.long(),
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        num_warmups=warmup_runs,
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
        grad_output = torch.randn(B * T * L, D).to(get_device())
    # backward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb(
            indices.long(),
            offsets.long(),
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        bwd_only=True,
        grad=grad_output,
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
def uvm(
    alpha: bool,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
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
    T_gpu = T - T_uvm
    L_uvm = uvm_bag_size

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
        )

    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes_uvm = (
        output_size_multiplier * B * sum(Ds[:T_uvm])
        + param_size_multiplier * B * sum(Ds[:T_uvm]) * L_uvm
    )

    time_per_iter = benchmark_requests(
        requests_uvm,
        lambda indices, offsets, per_sample_weights: emb_uvm.forward(
            indices.long(),
            offsets.long(),
            per_sample_weights,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"UVM Forward, B: {B}, "
        f"E: {E}, T: {T_uvm}, D: {D}, L: {L_uvm}, W: {weighted}, "
        f"BW: {read_write_bytes_uvm / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    if T_gpu > 0:
        requests = []
        assert requests_gpu is not None
        for rs_uvm, rs_gpu in zip(requests_uvm, requests_gpu):
            indices = torch.cat([rs_uvm[0], rs_gpu[0]])
            lengths = [L_uvm] * (T_uvm * B) + [L] * (T_gpu * B)
            offsets = torch.tensor(([0] + np.cumsum(lengths).tolist())).int().cuda()
            per_sample_weights = None
            if weighted:
                this_rs_uvm_weights = rs_uvm[2]
                assert this_rs_uvm_weights is not None
                this_rs_gpu_weights = rs_gpu[2]
                assert this_rs_gpu_weights is not None
                per_sample_weights = torch.cat(
                    [this_rs_uvm_weights, this_rs_gpu_weights]
                )
            requests.append((indices, offsets, per_sample_weights))

        # forward
        time_per_iter = benchmark_requests(
            requests_gpu,
            lambda indices, offsets, per_sample_weights: emb_gpu.forward(
                indices.long(),
                offsets.long(),
                per_sample_weights,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
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
                indices.long(),
                offsets.long(),
                per_sample_weights,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
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
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
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
    long_index: bool,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
    requests_data_file: Optional[str],
    tables: Optional[str],
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
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    logging.info(
        f"Embedding tables: {E * T} rows, {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier  / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
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
    )
    warmup_requests, requests = requests[:iters], requests[iters:]
    grad_output = torch.randn(B, sum(Ds)).cuda()

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb_nc(
            indices.long(), offsets.long(), per_sample_weights
        ).backward(grad_output),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"ForwardBackward (UVM), B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    # warm up
    for indices, offsets, _ in warmup_requests:
        emb.forward(indices.long(), offsets.long())
    # get cache miss rate (forward and backward) and exchanged cache lines (prefetch)
    cache_misses = []
    exchanged_cache_lines = []
    NOT_FOUND = -1
    for indices, offsets, _ in requests:
        old_lxu_cache_state = emb.lxu_cache_state.clone()
        emb.prefetch(indices.long(), offsets.long())
        exchanged_cache_lines.append(
            (emb.lxu_cache_state != old_lxu_cache_state).sum().item()
        )
        cache_misses.append((emb.lxu_cache_locations_list[0] == NOT_FOUND).sum().item())
        emb.forward(indices.long(), offsets.long())
    logging.info(
        f"Exchanged cache lines -- mean: {sum(exchanged_cache_lines)/len(requests): .2f}, "
        f"max: {max(exchanged_cache_lines)}, min: {min(exchanged_cache_lines)}"
    )
    logging.info(
        f"Cache miss -- mean: {sum(cache_misses)/len(requests)}, "
        f"max: {max(cache_misses)}, min: {min(cache_misses)}"
    )

    # benchmark prefetch
    emb.reset_cache_states()
    for indices, offsets, _ in warmup_requests:
        emb.forward(indices, offsets)
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


def benchmark_cpu_requests(
    requests: List[Tuple[torch.IntTensor, torch.IntTensor, Optional[torch.Tensor]]],
    func: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
) -> float:
    import time

    start_time = time.perf_counter()
    for indices, offsets, weights in requests:
        func(indices, offsets, weights)
    end_time = time.perf_counter()
    return (end_time - start_time) / len(requests)


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
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
def nbit_cpu(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
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

    emb = IntNBitTableBatchedEmbeddingBagsCodegen(
        [("", E, d, weights_precision, EmbeddingLocation.HOST) for d in Ds],
        device="cpu",
        index_remapping=[torch.arange(E) for _ in Ds] if index_remapping else None,
        output_dtype=output_dtype,
        fp8_exponent_bits=fp8_exponent_bits,
        fp8_exponent_bias=fp8_exponent_bias,
    ).cpu()
    emb.fill_random_weights()
    fill_random_scale_bias(emb, T, weights_precision)

    nparams_byte = sum(w.numel() for (w, _) in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes = (
        output_size_multiplier * B * T * D + param_size_multiplier * B * T * L * D
    )
    logging.info(
        f"{weights_precision} Embedding tables: {E * T} rows, {nparams_byte / param_size_multiplier / 1.0e9: .2f} GParam, "
        f"{nparams_byte / 1.0e9: .2f} GB"  # IntN TBE use byte for storage
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
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
        (a.cpu().int(), b.cpu().int(), c.cpu() if c else None) for (a, b, c) in requests
    ]

    time_per_iter = benchmark_cpu_requests(
        # pyre-fixme[6]: For 1st param expected `List[Tuple[IntTensor, IntTensor,
        #  Optional[Tensor]]]` but got `List[Tuple[Tensor, Tensor, Optional[Tensor]]]`.
        requests,
        lambda indices, offsets, per_sample_weights: emb.forward(
            indices,
            offsets,
            per_sample_weights,
        ),
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
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--report-aibench", is_flag=True)
@click.option("--run-reference", is_flag=True, default=False)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
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
    warmup_runs: int,
    output_dtype: SparseType,
    report_aibench: bool,
    run_reference: bool,
    requests_data_file: Optional[str],
    tables: Optional[str],
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
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
        requests = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]

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

    if report_aibench and haveAIBench:
        print(
            emitMetric(
                type="NET",
                metric=f"bandwidth_{weights_precision}",
                unit="scalar",
                value=str(bandwidth),
            )
        )
        print(
            emitMetric(
                type="NET",
                metric=f"time_per_iter_{weights_precision}",
                unit="scalar",
                value=str(time_per_iter * 1.0e6),
            )
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
            requests = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]

            # forward
            time_per_iter_refer = benchmark_requests_refer(
                requests,
                T,
                B,
                L,
                E,
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
@click.option("--warmup-runs", default=2)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--report-aibench", is_flag=True)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
@click.option("--use-cpu", is_flag=True, default=False)
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
    warmup_runs: int,
    output_dtype: SparseType,
    report_aibench: bool,
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
    use_cpu: bool,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
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
            for it, (indices, offsets, weights) in enumerate(requests):
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
            requests.append((indices, offsets, weights))
        if use_cpu:
            requests = [
                (a.cpu().int(), b.cpu().int(), c.cpu() if c else None)
                for (a, b, c) in requests
            ]
        else:
            requests = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]
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
            )

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

    if report_aibench and haveAIBench:
        print(
            emitMetric(
                type="NET",
                metric=f"bandwidth_{weights_precision}",
                unit="scalar",
                value=str(bandwidth),
            )
        )
        print(
            emitMetric(
                type="NET",
                metric=f"time_per_iter_{weights_precision}",
                unit="scalar",
                value=str(time_per_iter * 1.0e6),
            )
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
def nbit_uvm(
    alpha: bool,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    iters: int,
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
    requests_uvm = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests_uvm]

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
            (a.int(), b.int(), c if c else None) for (a, b, c) in requests_gpu
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
            indices = torch.cat([rs_uvm[0], rs_gpu[0]])
            lengths = [L_uvm] * (T_uvm * B) + [L] * (T_gpu * B)
            offsets = torch.tensor(([0] + np.cumsum(lengths).tolist())).int().cuda()
            per_sample_weights = None
            if weighted:
                this_rs_uvm_weights = rs_uvm[2]
                assert this_rs_uvm_weights is not None
                this_rs_gpu_weights = rs_gpu[2]
                assert this_rs_gpu_weights is not None
                per_sample_weights = torch.cat(
                    [this_rs_uvm_weights, this_rs_gpu_weights]
                )
            requests.append((indices, offsets, per_sample_weights))

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
        for indices, offsets, _ in requests:
            emb_mixed.forward(indices, offsets)
        prefetch_time, forward_time = benchmark_pipelined_requests(
            requests,
            lambda indices, offsets, indices_weights: emb_mixed.prefetch(
                indices,
                offsets,
            ),
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
@click.option("--warmup", default=10)
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
    warmup: int,
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
    # pyre-fixme[9]: requests_uvm has type `List[Tuple[IntTensor, IntTensor,
    #  Optional[Tensor]]]`; used as `List[Tuple[Tensor, Tensor, Optional[Tensor]]]`.
    requests_uvm: List[Tuple[torch.IntTensor, torch.IntTensor, Optional[Tensor]]] = [
        (a.int(), b.int(), c if c else None) for (a, b, c) in _requests_uvm
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
            num_warmups=warmup,
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
                    ind, off, _ = req
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
    ).cuda()
    emb.fill_random_weights()
    fill_random_scale_bias(emb, T, weights_precision)

    nparams_byte = sum(w.numel() for (w, _) in emb.split_embedding_weights())
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes = (
        param_size_multiplier * B * sum(Ds) * L
        + output_size_multiplier * B * sum(Ds)
        + param_size_multiplier * B * sum(Ds) * L
    )
    logging.info(
        f"{weights_precision} Embedding tables: {E * T} rows, {nparams_byte / param_size_multiplier / 1.0e9: .2f} GParam, "
        f"{nparams_byte / 1.0e9: .2f} GB"  # IntN TBE use byte for storage
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        f"{B * T * L * D * param_size_multiplier / 1.0e9: .2f} GB"
    )

    requests = generate_requests(
        2 * iters, B, T, L, E, reuse=reuse, alpha=alpha, weighted=weighted
    )
    requests = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]
    warmup_requests, requests = requests[:iters], requests[iters:]

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb_nc(
            indices.int(), offsets.int(), per_sample_weights
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"Forward (UVM) {weights_precision}, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    # warm up
    for indices, offsets, _ in warmup_requests:
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

    for indices, offsets, _ in requests:
        old_lxu_cache_state = emb.lxu_cache_state.clone()
        emb.prefetch(indices, offsets)
        exchanged_cache_lines.append(
            (emb.lxu_cache_state != old_lxu_cache_state).sum().item()
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
        f"Exchanged cache lines -- mean: {sum(exchanged_cache_lines)/len(requests): .2f}, "
        f"max: {max(exchanged_cache_lines)}, min: {min(exchanged_cache_lines)}"
    )
    logging.info(
        f"Cache miss -- mean: {sum(cache_misses)/len(requests)}, "
        f"max: {max(cache_misses)}, min: {min(cache_misses)}"
    )
    logging.info(
        f"input_indices -- mean: {sum(input_indices)/len(requests)}, "
        f"max: {max(input_indices)}, min: {min(input_indices)}"
    )
    logging.info(
        f"unique_indices -- mean: {sum(unique_indices)/len(requests)}, "
        f"max: {max(unique_indices)}, min: {min(unique_indices)}"
    )
    unique_miss_rate = [a / b for (a, b) in zip(exchanged_cache_lines, unique_indices)]
    logging.info(
        f"unique_miss_rate -- mean: {sum(unique_miss_rate)/len(requests)}, "
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

    for indices, offsets, _ in warmup_requests:
        emb.forward(indices, offsets)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("pipeline")
    prefetch_time, forward_time = benchmark_pipelined_requests(
        requests,
        lambda indices, offsets, indices_weights: emb.prefetch(
            indices,
            offsets,
        ),
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
        f"{2 * sum(exchanged_cache_lines) * param_size_multiplier * D / prefetch_time / len(requests) / 1.0e9: .2f} GB/s, "
        f"TfwdTime: {forward_time * 1.0e6:.0f}us, "
        f"{read_write_bytes / forward_time / 1.0e9: .2f} GB/s"
    )
    torch.cuda.cudart().cudaProfilerStop()


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=2048)
@click.option("--iters", default=10)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=100)
@click.option("--pruning-hash-load-factor", default=0.75)
@click.option("--hit-rate", default=0.9)
@click.option("--use-cpu", is_flag=True, default=False)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
def hashtable(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    num_embeddings: int,
    num_tables: int,
    pruning_hash_load_factor: float,
    hit_rate: float,
    use_cpu: bool,
    requests_data_file: Optional[str],
    tables: Optional[str],
) -> None:
    B = batch_size
    T = num_tables
    L = bag_size
    E = num_embeddings
    np.random.seed(42)
    torch.manual_seed(42)
    if hit_rate == 1.0:
        chosen_indices = torch.cat([torch.arange(E) for _ in range(T)], dim=0).int()
    else:
        chosen_indices = (
            torch.randint(low=0, high=int(E * 1.0 / hit_rate), size=(E * T,))
            .view(-1)
            .int()
        )
    dense_indices = torch.cat([torch.arange(E) for _ in range(T)], dim=0).int()
    offsets = torch.tensor([E * t for t in range(T + 1)]).int()
    assert offsets[-1] == chosen_indices.numel()
    assert offsets.numel() == T + 1
    assert (offsets.numel() - 1) // T == 1

    capacities = [round_up(int(E / pruning_hash_load_factor), 32) for _ in range(T)]

    hash_table = torch.zeros(
        (sum(capacities), 2),
        dtype=torch.int32,
    )
    hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).long()

    assert hash_table.numel() * 4 < 2**32
    # initialize
    hash_table[:, :] = -1
    torch.ops.fbgemm.pruned_hashmap_insert(
        chosen_indices, dense_indices, offsets, hash_table, hash_table_offsets
    )

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        requests_data_file=requests_data_file,
        tables=tables,
    )

    if not use_cpu:
        hash_table = hash_table.cuda()
        hash_table_offsets = hash_table_offsets.cuda()
        requests = [(a.cuda().int(), b.cuda().int(), c) for (a, b, c) in requests]
    else:
        requests = [(a.int().cpu(), b.int().cpu(), c) for (a, b, c) in requests]

    empirical_hit_rate = np.mean(
        [
            torch.ops.fbgemm.pruned_hashmap_lookup(
                indices, offsets, hash_table, hash_table_offsets
            )
            .ne(-1)
            .sum()
            .item()
            / indices.numel()
            for indices, offsets, _ in requests
        ]
    )

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: torch.ops.fbgemm.pruned_hashmap_lookup(
            indices, offsets, hash_table, hash_table_offsets
        ),
    )

    logging.info(
        f"LinearTable: B: {B}, T: {T}, L: {L}, E: {E}, QPS: {B * T * L / time_per_iter / 1.0e9:.2f}B QPS/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us, pruning load factor: {E * T / hash_table.shape[0] * 100:.1f}%, hit rate: {empirical_hit_rate * 100:.2f}%, Table size: {hash_table.numel() * 4 / 1.0e9:.0f} GB"
    )

    if use_cpu:
        ht = torch.classes.fbgemm.PrunedMapCPU()
        ht.insert(chosen_indices, dense_indices, offsets, T)

        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, _: ht.lookup(indices, offsets),
        )

        logging.info(
            f"HashTable: B: {B}, T: {T}, L: {L}, E: {E}, QPS: {B * T * L / time_per_iter / 1.0e9:.2f}B QPS/s, "
            f"T: {time_per_iter * 1.0e6:.0f}us, pruning load factor: {E * T / hash_table.shape[0] * 100:.1f}%, hit rate: {empirical_hit_rate * 100:.2f}%, Table size: {hash_table.numel() * 4 / 1.0e9:.0f} GB"
        )


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=2048)
@click.option("--iters", default=100)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=100)
@click.option("--pruning-ratio", default=0.9)
@click.option("--device", default="cuda")
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
def pruned_array(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    num_embeddings: int,
    num_tables: int,
    pruning_ratio: float,
    device: str,
    requests_data_file: Optional[str],
    tables: Optional[str],
) -> None:
    B = batch_size
    T = num_tables
    L = bag_size
    E = num_embeddings
    np.random.seed(42)
    torch.manual_seed(42)
    assert pruning_ratio > 0 and pruning_ratio <= 1
    original_E = int(E / (1.0 - pruning_ratio))
    index_remappings = torch.tensor(
        [-1] * original_E * T, dtype=torch.int32, device=device
    )
    index_remappings_offsets = torch.empty(T + 1, dtype=torch.int64, device=device)
    index_remappings_offsets[0] = 0
    dense_indices = torch.tensor(range(E), dtype=torch.int32, device=device)
    for t in range(T):
        selected_indices = torch.add(
            torch.randperm(original_E, device=device), t * original_E
        )[:E]
        index_remappings[selected_indices] = dense_indices
        index_remappings_offsets[t + 1] = index_remappings_offsets[t] + original_E

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        requests_data_file=requests_data_file,
        tables=tables,
        use_cpu=True if device == "cpu" else False,
    )
    requests = [(a.int().to(device), b.int().to(device), c) for (a, b, c) in requests]

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: torch.ops.fbgemm.pruned_array_lookup(
            indices,
            offsets,
            index_remappings,
            index_remappings_offsets,
        ),
    )

    logging.info(
        f"LinearTable: B: {B}, T: {T}, L: {L}, E: {E}, QPS: {B * T * L / time_per_iter / 1.0e9:.2f}B QPS/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us, Pruning Ratio: {pruning_ratio * 100:.2f}%, Table size: {original_E * T * 4 / 1.0e9:.0f} GB"
    )


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--iters", default=100)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.WARNING.value)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
def bounds_check_indices(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    num_embeddings: int,
    num_tables: int,
    bounds_check_mode: int,
    requests_data_file: Optional[str],
    tables: Optional[str],
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    L = bag_size
    E = num_embeddings
    T = num_tables

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        requests_data_file=requests_data_file,
        tables=tables,
    )
    # requests = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]

    warning = torch.tensor([0]).long().to(get_device())
    rows_per_table = torch.tensor([E for _ in range(T)]).long().to(get_device())
    # forward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: torch.ops.fbgemm.bounds_check_indices(
            rows_per_table,
            indices,
            offsets,
            BoundsCheckMode(bounds_check_mode),
            warning,
        ),
    )

    logging.info(
        f"Bounds Check Indices:  B: {B}, "
        f"E: {E}, T: {T}, L: {L}, "
        f"BW: {(8 * B * T * L + 8 * (B * T + 1)) / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--num-tables", type=int, default=32)
@click.option("--embedding-dim", type=int, default=248)
@click.option("--num-embeddings", type=int, default=int(1e5))
@click.option("--update-row-num", type=int, default=1e4)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
@click.option("--iters", type=int, default=100)
@click.option("--fp8-exponent-bits", type=int, default=None)
@click.option("--fp8-exponent-bias", type=int, default=None)
def emb_inplace_update(  # noqa C901
    num_tables: int,
    embedding_dim: int,
    num_embeddings: int,
    update_row_num: int,
    weights_precision: SparseType,
    output_dtype: SparseType,
    iters: int,
    fp8_exponent_bits: Optional[int],
    fp8_exponent_bias: Optional[int],
) -> None:
    if open_source:
        logging.warning(
            "emb_inplace_update op benchmark doesn't support open source now!"
        )
        return

    np.random.seed(42)
    torch.manual_seed(42)

    T = num_tables
    D = embedding_dim
    E = num_embeddings
    N = update_row_num

    D_alignment = max(weights_precision.align_size() for t in range(T))
    D_alignment = max(D_alignment, output_dtype.align_size())
    D = round_up(D, D_alignment)
    Ds = [
        round_up(
            np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
            D_alignment,
        )
        for _ in range(T)
    ]
    Es = [E] * T
    row_alignment = 16  # use_cpu = False -> only test CUDA function now

    weights_ty_list = [weights_precision] * T
    managed = [EmbeddingLocation.DEVICE] * T
    embedding_specs = [
        (
            "",
            E,
            D,
            W_TY,
            EmbeddingLocation(M),
        )
        for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
    ]
    op = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=embedding_specs,
        output_dtype=output_dtype,
        device=torch.cuda.current_device(),
        fp8_exponent_bits=fp8_exponent_bits,
        fp8_exponent_bias=fp8_exponent_bias,
    )
    # Initilize the random weights for int nbit table split embedding bag
    op.fill_random_weights()

    update_table_idx = [np.random.randint(low=0, high=T) for _ in range(N)]
    # Generate non-dup indices
    table_map = {}
    update_row_idx = []
    for t in update_table_idx:
        while True:
            row_idx = np.random.randint(low=0, high=Es[t])
            if t not in table_map or row_idx not in table_map[t]:
                break
        if t in table_map:
            table_map[t].append(row_idx)
        else:
            table_map[t] = []
        table_map[t].append(row_idx)
        update_row_idx.append(row_idx)
    update_weight_size = sum(
        [
            rounded_row_size_in_bytes(
                Ds[t],
                weights_ty_list[t],
                row_alignment,
            )
            for t in update_table_idx
        ]
    )

    update_weights = torch.randint(
        low=0,
        high=255,
        size=(update_weight_size,),
        dtype=torch.uint8,
        device=torch.cuda.current_device(),
    )

    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    read_write_bytes = output_size_multiplier * N * D + param_size_multiplier * N * D

    # Update op weights with the customized ops
    op.embedding_inplace_update_internal(
        update_table_idx,
        update_row_idx,
        update_weights,
    )

    time_per_iter, _ = benchmark_torch_function(
        op.embedding_inplace_update_internal,
        (update_table_idx, update_row_idx, update_weights),
        iters=iters,
    )

    logging.info(
        f"Emb inplace update (including H2D for metadata): "
        f"T: {T}, D: {D}, E: {E}, N: {N}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9:.2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    update_offsets = []
    update_offset = 0
    for table_idx in update_table_idx:
        D_bytes = rounded_row_size_in_bytes(
            Ds[table_idx],
            weights_ty_list[table_idx],
            row_alignment,
        )
        update_offsets.append(update_offset)
        update_offset += D_bytes
    update_offsets.append(update_offset)

    update_table_idx = torch.tensor(
        update_table_idx,
        device=torch.cuda.current_device(),
        dtype=torch.int32,
    )
    update_row_idx = torch.tensor(
        update_row_idx,
        device=torch.cuda.current_device(),
        dtype=torch.int32,
    )
    update_offsets = torch.tensor(
        update_offsets,
        device=torch.cuda.current_device(),
        dtype=torch.int64,
    )

    time_per_iter, _ = benchmark_torch_function(
        torch.ops.fbgemm.emb_inplace_update,
        (
            op.weights_dev,
            op.weights_uvm,
            op.weights_placements,
            op.weights_offsets,
            op.weights_tys,
            op.D_offsets,
            update_weights,
            update_table_idx,
            update_row_idx,
            update_offsets,
            16,  # row_alignment
        ),
        iters=iters,
    )

    logging.info(
        f"Emb inplace update (pure device update op): "
        f"T: {T}, D: {D}, E: {E}, N: {N}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9:.2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
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
        managed_option = (
            EmbeddingLocation.DEVICE
            if torch.cuda.is_available()
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
                ComputeDevice.CUDA if torch.cuda.is_available() else ComputeDevice.CPU,
            )
            for d, e in zip(Ds, Es)
        ],
        optimizer=optimizer,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=weights_precision,
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
            sigma_L=sigma_Ls[t] if use_variable_bag_sizes else None,
            zipf_oversample_ratio=3 if Ls[t] > 5 else 5,
        )
        for i, (indices, offsets, weights) in enumerate(requests):
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
        requests.append((indices, offsets, weights))

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
            indices.long(),
            offsets.long(),
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
        grad_output = torch.randn(requests[0][0].numel(), D).to(get_device())
    # backward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb(
            indices.long(),
            offsets.long(),
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        bwd_only=True,
        grad=grad_output,
    )
    logging.info(
        f"Backward, B: {B}, Es: {Es}, T: {T}, Ds: {Ds}, Ls: {Ls_str}, "
        f"BW: {2 * read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


def _to_offsets(lengths: torch.Tensor) -> torch.Tensor:
    return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)


@cli.command()
@click.option("--batch-size", default=128000)
@click.option("--compressed-batch-size", default=12800)
@click.option("--embedding-dim", default=128)
@click.option("--bag-size", default=5)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=20)
@click.option("--compressed-tables", default=10)
@click.option("--iters", default=100)
def vbe(
    batch_size: int,
    compressed_batch_size: int,
    embedding_dim: int,
    bag_size: int,
    num_embeddings: int,
    num_tables: int,
    compressed_tables: int,
    iters: int,
) -> None:
    torch.manual_seed(42)
    B = batch_size
    cB = compressed_batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    cT = compressed_tables
    Ds = [D] * T
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
    managed_option = (
        EmbeddingLocation.DEVICE
        if torch.cuda.is_available()
        else EmbeddingLocation.HOST
    )
    pooling_mode = PoolingMode.SUM

    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                managed_option,
                ComputeDevice.CUDA,
            )
            for d in Ds
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

    compressed_batch_sizes = ([cB] * cT) + ([B] * (T - cT))
    compressed_lengths = [L] * sum(compressed_batch_sizes)
    compressed_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(compressed_lengths, device=get_device())
    )
    compressed_values = torch.randint(
        low=0,
        high=E,
        size=(sum(compressed_lengths),),
        device=get_device(),
        dtype=torch.int32,
    )

    batch_sizes = [B] * T
    lengths = [L] * sum(batch_sizes)
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(lengths, device=get_device())
    )
    reindex = []

    for t in range(cT):
        start = t * cB
        end = cB * (t + 1)
        reindex.extend(range(start, end))
        for _ in range(B - cB):
            i = random.randint(t * cB, cB * (t + 1))
            reindex.append(i)
    reindex.extend(range(cB * cT, (cB * cT) + (B * cT)))

    reindex = torch.tensor(reindex, device=get_device())
    values = torch.index_select(compressed_values.reshape(-1, L), 0, reindex).flatten()

    requests = [
        (
            values,
            offsets,
        )
        for _ in range(iters)
    ]
    compressed_requests = [
        (
            compressed_values,
            compressed_offsets,
        )
        for _ in range(iters)
    ]

    out = benchmark_vbe(
        requests,
        compressed_requests,
        baseline_func=lambda indices, offsets: emb.forward(
            indices.long(),
            offsets.long(),
        ),
        compressed_func=lambda indices, offsets: emb.forward(
            indices.long(),
            offsets.long(),
            batch_size_per_feature_per_rank=[[bs] for bs in compressed_batch_sizes],
        ),
        reindex=reindex,
        embedding_dim=D,
    )
    logging.info(
        f"Uncompressed, B: {B}, T: {T}, D: {D}, L: {L}, "
        f"T: {out.avg * 1.0e6:.0f}us, fwd: {out.fwd * 1.0e6:.0f}us, bwd: {out.bwd * 1.0e6:.0f}us\n"
        f"Compressed, B: {B}, cB: {cB}, T: {T - cT}, cT: {cT}, D: {D}, L: {L}, "
        f"T: {out.compressed_avg * 1.0e6:.0f}us, fwd: {out.compressed_fwd * 1.0e6:.0f}us, reindex: {out.reindex * 1.0e6:.0f}us, bwd: {out.compressed_bwd * 1.0e6:.0f}us"
    )


if __name__ == "__main__":
    cli()
