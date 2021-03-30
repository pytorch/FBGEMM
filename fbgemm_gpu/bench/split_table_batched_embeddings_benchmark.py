#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, List, Optional, Tuple

import click
import fbgemm_gpu.split_table_batched_embeddings_ops as split_table_batched_embeddings_ops
import numpy as np
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import OptimType, SparseType
from typing import Dict

logging.basicConfig(level=logging.DEBUG)

PRECISION_SIZE_MULTIPLIER: Dict[SparseType, int] = {
    SparseType.FP32: 4,
    SparseType.FP16: 2,
    SparseType.INT8: 1,
}


def div_round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1))
def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.long().contiguous().view(-1).cuda(),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long().cuda(),
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
) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if alpha <= 1.0:
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(iters, T, B * L),
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
    else:
        all_indices = (
            torch.as_tensor(np.random.zipf(a=alpha, size=(iters, T, B * L)))
            .to(torch.cuda.current_device())
            .int()
            % E
        )
    for it in range(iters - 1):
        for t in range(T):
            reused_indices = torch.randperm(B * L, device=torch.cuda.current_device())[
                : int(B * L * reuse)
            ]
            all_indices[it + 1, t, reused_indices] = all_indices[it, t, reused_indices]

    rs = []
    for it in range(iters):
        weights_tensor = None if not weighted else torch.randn(
            T * B * L,
            device=torch.cuda.current_device(),
            dtype=torch.float16
            if weights_precision == SparseType.FP16
            else torch.float32,
        )
        rs.append(
            get_table_batched_offsets_from_dense(all_indices[it].view(T, B, L))
            + (weights_tensor,)
        )
    return rs


def benchmark_requests(
    requests: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    f: Callable,
) -> float:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for (indices, offsets, weights) in requests:
        f(indices, offsets, weights)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / len(requests)


def benchmark_pipelined_requests(
    requests: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    f: Callable,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    g: Callable,
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
        start_event[0].record()
        f(indices, offsets, indices_weights)
        end_event[0].record()
        start_event[1].record()
        g(indices, offsets, indices_weights)
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
            .cuda()
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
        managed_option = split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
    else:
        managed_option = split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED

    emb = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                managed_option,
                split_table_batched_embeddings_ops.ComputeDevice.CUDA,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
    ).cuda()
    if weights_precision == SparseType.INT8:
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in emb.split_embedding_weights())

    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]

    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f}GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L * D * param_size_multiplier / 1.0e6: .2f}MB"
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
    )
    logging.info(
        f"Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {param_size_multiplier * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    grad_output = torch.randn(B, sum(Ds)).cuda()
    # backward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb(
            indices.long(),
            offsets.long(),
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        ).backward(grad_output),
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
) -> None:

    np.random.seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    T_uvm = uvm_tables
    assert T_uvm <= T
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
    emb_uvm = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                split_table_batched_embeddings_ops.ComputeDevice.CUDA,
            )
            for d in Ds[:T_uvm]
        ],
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb_uvm.init_embedding_weights_uniform(-0.0003, 0.0003)

    emb_gpu = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                split_table_batched_embeddings_ops.ComputeDevice.CUDA,
            )
            for d in Ds[T_uvm:]
        ],
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb_gpu.init_embedding_weights_uniform(-0.0003, 0.0003)

    emb_mixed = (
        split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                    managed_option,
                    split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                )
                for (d, managed_option) in zip(
                    Ds,
                    [split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED]
                    * T_uvm
                    + [split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE]
                    * T_gpu,
                )
            ],
            weights_precision=weights_precision,
            stochastic_rounding=stoc,
        ).cuda()
    )

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
    requests = []
    for rs_uvm, rs_gpu in zip(requests_uvm, requests_gpu):
        indices = torch.cat([rs_uvm[0], rs_gpu[0]])
        lengths = [L_uvm] * (T_uvm * B) + [L] * (T_gpu * B)
        offsets = torch.tensor(([0] + np.cumsum(lengths).tolist())).int().cuda()
        per_sample_weights = None
        if weighted:
            assert rs_uvm[2] is not None
            assert rs_gpu[2] is not None
            per_sample_weights = torch.cat([rs_uvm[2], rs_gpu[2]])
        requests.append((indices, offsets, per_sample_weights))

    # forward
    time_per_iter = benchmark_requests(
        requests_gpu,
        lambda indices, offsets, per_sample_weights: emb_gpu.forward(
            indices.long(),
            offsets.long(),
            per_sample_weights,
        ),
    )
    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]

    logging.info(
        f"GPU Forward, B: {B}, "
        f"E: {E}, T: {T_gpu}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {param_size_multiplier * B * T_gpu * L * D / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    time_per_iter = benchmark_requests(
        requests_uvm,
        lambda indices, offsets, per_sample_weights: emb_uvm.forward(
            indices.long(),
            offsets.long(),
            per_sample_weights,
        ),
    )
    logging.info(
        f"UVM Forward, B: {B}, "
        f"E: {E}, T: {T_gpu}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {param_size_multiplier * B * T_gpu * L * D / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb_mixed.forward(
            indices.long(),
            offsets.long(),
            per_sample_weights,
        ),
    )
    logging.info(
        f"Mixed Forward, B: {B}, "
        f"E: {E}, T: {T_gpu}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {param_size_multiplier * B * T_gpu * L * D / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
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
) -> None:
    np.random.seed(42)

    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    cache_alg = (
        split_table_batched_embeddings_ops.CacheAlgorithm.LRU
        if cache_algorithm == "lru"
        else split_table_batched_embeddings_ops.CacheAlgorithm.LFU
    )
    if mixed:
        Ds = [
            div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    emb_nc = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                split_table_batched_embeddings_ops.ComputeDevice.CUDA,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb_nc.init_embedding_weights_uniform(-0.0003, 0.0003)

    emb = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING,
                split_table_batched_embeddings_ops.ComputeDevice.CUDA,
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
        # pyre-fixme[16]: `SplitTableBatchedEmbeddingBagsCodegen` has no attribute
        #  `lxu_cache_state`.
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
