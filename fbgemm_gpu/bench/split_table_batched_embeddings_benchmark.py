#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import click
import numpy as np
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    CacheAlgorithm,
    ComputeDevice,
    EmbeddingLocation,
    OptimType,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)

PRECISION_SIZE_MULTIPLIER: Dict[SparseType, int] = {
    SparseType.FP32: 4,
    SparseType.FP16: 2,
    SparseType.INT8: 1,
}


def div_round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def get_device() -> torch.device:
    return (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1))
def get_table_batched_offsets_from_dense(
    merged_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.long().contiguous().view(-1).to(get_device()),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long().to(get_device()),
    )


def generate_requests(
    iters: int,
    B: int,
    T: int,
    L: int,
    E: int,
    # inter-batch indices reuse rate
    reuse: float = 0.0,
    # alpha <= 1.0: use uniform distribution
    # alpha > 1.0: use zjpf distribution
    alpha: float = 1.0,
    weights_precision: SparseType = SparseType.FP32,
    weighted: bool = False,
) -> List[Tuple[Tensor, Tensor, Optional[Tensor]]]:
    if alpha <= 1.0:
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(iters, T, B * L),
            device=get_device(),
            dtype=torch.int32,
        )
    else:
        all_indices = (
            torch.as_tensor(np.random.zipf(a=alpha, size=(iters, T, B * L)))
            .to(get_device())
            .int()
            % E
        )
    for it in range(iters - 1):
        for t in range(T):
            reused_indices = torch.randperm(B * L, device=get_device())[
                : int(B * L * reuse)
            ]
            all_indices[it + 1, t, reused_indices] = all_indices[it, t, reused_indices]

    rs = []
    for it in range(iters):
        weights_tensor = (
            None if not weighted else torch.randn(T * B * L, device=get_device())
        )
        rs.append(
            get_table_batched_offsets_from_dense(all_indices[it].view(T, B, L))
            + (weights_tensor,)
        )
    return rs


def benchmark_requests(
    requests: List[Tuple[Tensor, Tensor, Optional[Tensor]]],
    func: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
    flush_gpu_cache_size_mb: int = 0,
) -> float:
    total_time = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    else:
        start_time = time.time()
    for (indices, offsets, weights) in requests:
        if torch.cuda.is_available():
            if flush_gpu_cache_size_mb:
                _ = torch.rand(flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float)
                torch.cuda.synchronize()
            start_event.record()
        else:
            start_time = time.time()
        func(indices, offsets, weights)
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) * 1.0e-3
        else:
            total_time += time.time() - start_time
    return total_time / len(requests)


def benchmark_pipelined_requests(
    requests: List[Tuple[Tensor, Tensor, Optional[Tensor]]],
    func1: Callable[[Tensor, Tensor, Optional[Tensor]], None],
    func2: Callable[[Tensor, Tensor, Optional[Tensor]], None],
    flush_gpu_cache_size_mb: int = 0,
) -> Tuple[float, float]:
    torch.cuda.synchronize()
    start_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in requests
    ]
    end_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in requests
    ]
    for ((indices, offsets, indices_weights), start_event, end_event) in zip(
        requests, start_events, end_events
    ):
        if flush_gpu_cache_size_mb:
            _ = torch.rand(flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float)
            torch.cuda.synchronize()
        start_event[0].record()
        func1(indices, offsets, indices_weights)
        end_event[0].record()
        start_event[1].record()
        func2(indices, offsets, indices_weights)
        end_event[1].record()
    torch.cuda.synchronize()
    return (
        sum(
            start_event[0].elapsed_time(end_event[0]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        )
        / len(requests),
        sum(
            start_event[1].elapsed_time(end_event[1]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        )
        / len(requests),
    )


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--managed", default="device")
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--weighted-num-requires-grad", type=int, default=None)
@click.option("--flush-gpu-cache-size-mb", default=0)
def device(  # noqa C901
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
    weighted_num_requires_grad: Optional[int],
    flush_gpu_cache_size_mb: int,
) -> None:
    np.random.seed(42)
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
            div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
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

    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                managed_option,
                ComputeDevice.CUDA if torch.cuda.is_available() else ComputeDevice.CPU,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
    ).to(get_device())
    if weights_precision == SparseType.INT8:
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in emb.split_embedding_weights())

    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]

    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f}GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * sum(Ds) * L * param_size_multiplier / 1.0e6: .2f}MB"
    )

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weights_precision=weights_precision,
        weighted=weighted,
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
    )
    logging.info(
        f"Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    grad_output = torch.randn(B, sum(Ds)).to(get_device())
    # backward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb(
            indices.long(),
            offsets.long(),
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        ).backward(grad_output),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"ForwardBackward, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f}GB/s, "
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
@click.option("--reuse", default=0.0)
@click.option("--uvm-tables", default=1)
@click.option("--uvm-bag-size", default=1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
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
) -> None:

    np.random.seed(42)
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

    if mixed:
        Ds = [
            div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
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
                EmbeddingLocation.MANAGED,
                ComputeDevice.CUDA,
            )
            for d in Ds[:T_uvm]
        ],
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
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
                    [EmbeddingLocation.MANAGED] * T_uvm
                    + [EmbeddingLocation.DEVICE] * T_gpu,
                )
            ],
            weights_precision=weights_precision,
            stochastic_rounding=stoc,
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
        weights_precision=weights_precision,
        weighted=weighted,
    )

    if T_gpu > 0:
        requests_gpu = generate_requests(
            iters,
            B,
            T_gpu,
            L,
            E,
            reuse=reuse,
            alpha=alpha,
            weights_precision=weights_precision,
            weighted=False,
        )

    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]

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
        f"BW: {param_size_multiplier * B * sum(Ds[:T_uvm]) * L_uvm / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    if T_gpu > 0:
        requests = []
        for rs_uvm, rs_gpu in zip(requests_uvm, requests_gpu):
            indices = torch.cat([rs_uvm[0], rs_gpu[0]])
            lengths = [L_uvm] * (T_uvm * B) + [L] * (T_gpu * B)
            offsets = torch.tensor(([0] + np.cumsum(lengths).tolist())).int().cuda()
            per_sample_weights = None
            if weighted:
                assert (this_rs_uvm_weights := rs_uvm[2]) is not None
                assert (this_rs_gpu_weights := rs_gpu[2]) is not None
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

        logging.info(
            f"GPU Forward, B: {B}, "
            f"E: {E}, T: {T_gpu}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {param_size_multiplier * B * sum(Ds[T_uvm:]) * L / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
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
        logging.info(
            f"Mixed Forward, B: {B}, "
            f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
            f"T: {time_per_iter * 1.0e6:.0f}us"
        )


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-sets", default=1024)
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
def cache(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    cache_algorithm: str,
    cache_sets: int,
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
) -> None:
    np.random.seed(42)

    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    cache_alg = CacheAlgorithm.LRU if cache_algorithm == "lru" else CacheAlgorithm.LFU
    if mixed:
        Ds = [
            div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
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
        cache_sets=cache_sets,
        cache_algorithm=cache_alg,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in emb.split_embedding_weights())
    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]
    logging.info(
        f"Embedding tables: {E * T} rows, {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier  / 1.0e6: .2f}MB"
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        f"{B * T * L * D * param_size_multiplier / 1.0e6: .2f}MB"
    )

    requests = generate_requests(
        2 * iters, B, T, L, E, reuse=reuse, alpha=alpha, weighted=weighted
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
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f}GB/s, "
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
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.clone)[[Named(self,
        #  Variable[torch._TTensor (bound to Tensor)])], Variable[torch._TTensor (bound
        #  to Tensor)]], Tensor], Tensor, torch.nn.Module]` is not a function.
        old_lxu_cache_state = emb.lxu_cache_state.clone()
        emb.prefetch(indices.long(), offsets.long())
        exchanged_cache_lines.append(
            # pyre-fixme[16]: `bool` has no attribute `sum`.
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
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / e2e_time / 1.0e9: .2f}GB/s, "
        f"Tprefetch: {prefetch_time * 1.0e6:.0f}us, "
        f"{2 * sum(exchanged_cache_lines) * param_size_multiplier * D / prefetch_time / len(requests) / 1.0e9: .2f} GB/s, "
        f"Tfwdbwd: {forward_backward_time * 1.0e6:.0f}us, "
        f"{3 * param_size_multiplier * B * sum(Ds) * L / forward_backward_time / 1.0e9: .2f} GB/s, "
        f"Te2e: {e2e_time * 1.0e6:.0f}us, "
    )


if __name__ == "__main__":
    cli()
