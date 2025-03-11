#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
import os
import random
from contextlib import nullcontext
from itertools import accumulate
from typing import Callable, Optional

import click
import fbgemm_gpu
import numpy as np

import torch

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
)
from fbgemm_gpu.tbe.bench import benchmark_eval_compression, benchmark_requests
from fbgemm_gpu.tbe.utils import generate_requests, get_device, round_up, TBERequest
from torch.profiler import profile

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

logging.basicConfig(level=logging.DEBUG)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=2048)
@click.option("--iters", default=10)
@click.option("--warmup-runs", default=0)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=100)
@click.option("--pruning-hash-load-factor", default=0.75)
@click.option("--hit-rate", default=0.9)
@click.option("--use-cpu", is_flag=True, default=False)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
def pruned_hashmap_insert(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    warmup_runs: int,
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
        requests = [
            TBERequest(
                req.indices.cuda().int(),
                req.offsets.cuda().int(),
                req.per_sample_weights,
            )
            for req in requests
        ]
    else:
        requests = [
            TBERequest(
                req.indices.int().cpu(), req.offsets.int().cpu(), req.per_sample_weights
            )
            for req in requests
        ]

    empirical_hit_rate = np.mean(
        [
            torch.ops.fbgemm.pruned_hashmap_lookup(
                req.indices, req.offsets, hash_table, hash_table_offsets
            )
            .ne(-1)
            .sum()
            .item()
            / req.indices.numel()
            for req in requests
        ]
    )

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: torch.ops.fbgemm.pruned_hashmap_lookup(
            indices, offsets, hash_table, hash_table_offsets
        ),
        num_warmups=warmup_runs,
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
            num_warmups=warmup_runs,
        )

        logging.info(
            f"HashTable: B: {B}, T: {T}, L: {L}, E: {E}, QPS: {B * T * L / time_per_iter / 1.0e9:.2f}B QPS/s, "
            f"T: {time_per_iter * 1.0e6:.0f}us, pruning load factor: {E * T / hash_table.shape[0] * 100:.1f}%, hit rate: {empirical_hit_rate * 100:.2f}%, Table size: {hash_table.numel() * 4 / 1.0e9:.0f} GB"
        )


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=2048)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=100)
@click.option("--pruning-ratio", default=0.9)
@click.option("--device", default="cuda")
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
def pruned_array_lookup(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    warmup_runs: int,
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
    requests = [
        TBERequest(
            req.indices.int().to(device),
            req.offsets.int().to(device),
            req.per_sample_weights,
        )
        for req in requests
    ]

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: torch.ops.fbgemm.pruned_array_lookup(
            indices,
            offsets,
            index_remappings,
            index_remappings_offsets,
        ),
        num_warmups=warmup_runs,
    )

    logging.info(
        f"LinearTable: B: {B}, T: {T}, L: {L}, E: {E}, QPS: {B * T * L / time_per_iter / 1.0e9:.2f}B QPS/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us, Pruning Ratio: {pruning_ratio * 100:.2f}%, Table size: {original_E * T * 4 / 1.0e9:.0f} GB"
    )


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=0)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option(
    "--bounds-check-mode",
    type=int,
    default=BoundsCheckMode.WARNING.value,
    help=f"Available modes: FATAL={BoundsCheckMode.FATAL.value}, "
    f"WARNING={BoundsCheckMode.WARNING.value}, "
    f"IGNORE={BoundsCheckMode.IGNORE.value}, "
    f"NONE={BoundsCheckMode.NONE.value}",
)
@click.option("--requests_data_file", type=str, default=None)
@click.option("--tables", type=str, default=None)
@click.option(
    "--batch-sizes",
    type=str,
    default="",
    help="A list of batch sizes for the variable batch size case (VBE). "
    "The list is comma separated, i.e., 512,128,4",
)
@click.option("--export-trace", is_flag=True, default=False)
@click.option(
    "--trace-url",
    type=str,
    default="bounds_check_indices_trace_{ospid}.json",
)
def bounds_check_indices(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    warmup_runs: int,
    num_embeddings: int,
    num_tables: int,
    bounds_check_mode: int,
    requests_data_file: Optional[str],
    tables: Optional[str],
    batch_sizes: str,
    export_trace: bool,
    trace_url: str,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    L = bag_size
    E = num_embeddings
    T = num_tables

    is_vbe = len(batch_sizes) > 0
    if is_vbe:
        Bs = [int(B) for B in batch_sizes.split(",")]
        assert (
            len(Bs) == T
        ), "The number of batch sizes must be the same as the number of tables"
        B_offsets = torch.tensor([0] + list(accumulate(Bs)))
        max_B = max(Bs)
        total_B = int(B_offsets[-1].item())
        requests = generate_requests(
            iters,
            total_B,
            1,
            L,
            E,
            requests_data_file=requests_data_file,
            tables=tables,
            index_dtype=torch.long,
            offset_dtype=torch.long,
        )
        B_offsets = B_offsets.to(get_device()).to(torch.int)
    else:
        B = batch_size
        Bs = [B] * T
        B_offsets = None
        max_B = -1
        total_B = B * T
        requests = generate_requests(
            iters,
            B,
            T,
            L,
            E,
            requests_data_file=requests_data_file,
            tables=tables,
            index_dtype=torch.long,
            offset_dtype=torch.long,
        )

    warning = torch.tensor([0]).long().to(get_device())
    rows_per_table = torch.tensor([E for _ in range(T)]).long().to(get_device())

    def _kineto_trace_handler(p: profile) -> None:
        p.export_chrome_trace(trace_url.format(ospid=os.getpid()))

    # pyre-ignore[3]
    def context_factory(on_trace_ready: Callable[[profile], None]):
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    if is_vbe:
        offsets = requests[0].offsets
        vbe_metadata = generate_vbe_metadata(
            offsets,
            [[b] for b in Bs],
            pooling_mode=PoolingMode.SUM,
            feature_dims_cpu=torch.tensor(
                [-1] * T, device="cpu", dtype=torch.int64
            ),  # unused
            device=get_device(),
        )
        assert vbe_metadata.B_offsets is not None
        info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(
            vbe_metadata.B_offsets,  # unused tensor
            vbe_metadata.max_B,
            vbe_metadata.B_offsets.numel() - 1,  # T
        )
        row_output_offsets, b_t_map = torch.ops.fbgemm.generate_vbe_metadata(
            vbe_metadata.B_offsets,
            vbe_metadata.B_offsets_rank_per_feature,
            vbe_metadata.output_offsets_feature_rank,
            torch.tensor(
                [-1] * (T + 1), device=get_device(), dtype=torch.int
            ),  # unused D_offsets
            -1,  # unused max_D
            False,  # nobag
            vbe_metadata.max_B_feature_rank,
            info_B_num_bits,
            offsets.numel() - 1,  # total_B
        )
    else:
        b_t_map = None
        info_B_num_bits = -1
        info_B_mask = -1

    with context_factory(lambda p: _kineto_trace_handler(p)):
        # forward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, _: torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                BoundsCheckMode(bounds_check_mode),
                warning,
                B_offsets=B_offsets,
                max_B=max_B,
                b_t_map=b_t_map,
                info_B_num_bits=info_B_num_bits,
                info_B_mask=info_B_mask,
            ),
            num_warmups=warmup_runs,
        )

    logging.info(
        f"Bounds Check Indices:  Bs: {Bs}, "
        f"E: {E}, T: {T}, L: {L}, "
        f"BW: {(8 * total_B * L + 8 * (total_B + 1)) / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
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
@click.option("--warmup-runs", default=0)
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
    warmup_runs: int,
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
        num_warmups=warmup_runs,
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
        num_warmups=warmup_runs,
    )

    logging.info(
        f"Emb inplace update (pure device update op): "
        f"T: {T}, D: {D}, E: {E}, N: {N}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9:.2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--batch-size", default=128000)
@click.option("--compressed-batch-size", default=12800)
@click.option("--embedding-dim", default=128)
@click.option("--bag-size", default=5)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=20)
@click.option("--compressed-tables", default=10)
@click.option("--iters", default=100)
def tbe_input_compression(
    batch_size: int,
    compressed_batch_size: int,
    embedding_dim: int,
    bag_size: int,
    num_embeddings: int,
    num_tables: int,
    compressed_tables: int,
    iters: int,
) -> None:
    # TODO: Add warmup_runs
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
    ).long()
    compressed_values = torch.randint(
        low=0,
        high=E,
        size=(sum(compressed_lengths),),
        device=get_device(),
        dtype=torch.long,
    )

    batch_sizes = [B] * T
    lengths = [L] * sum(batch_sizes)
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(lengths, device=get_device())
    ).long()
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
    values = (
        torch.index_select(compressed_values.reshape(-1, L), 0, reindex)
        .flatten()
        .long()
    )

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

    out = benchmark_eval_compression(
        requests,
        compressed_requests,
        baseline_func=lambda indices, offsets: emb.forward(
            indices,
            offsets,
        ),
        compressed_func=lambda indices, offsets: emb.forward(
            indices,
            offsets,
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
