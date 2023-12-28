# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from typing import List, Tuple

import click
import numpy as np
import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)

from torch import nn, Tensor

logging.basicConfig(level=logging.DEBUG)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:cumem_utils_hip")
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:split_table_batched_embeddings_hip"
        )
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:cumem_utils")
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:split_table_batched_embeddings"
        )


# pyre-ignore
def benchmark_same_input(iters: int, f, *args) -> float:
    """
    Returns average execution time in milliseconds across "iters".
    """
    # Warm-up
    f(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


# pyre-ignore
def benchmark_different_inputs(f, args) -> float:
    """
    Returns average execution time in milliseconds across "iters".
    """
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for arg in args:
        f(arg)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / len(args)


def get_num_cached_tables(num_tables: int, cached_tables_ratio: float) -> int:
    """
    Controls how # of cached tables are determined based on parameters.
    """
    return round(num_tables * cached_tables_ratio)


def create_table_offsets(
    num_tables: int, cached_tables_ratio: float, num_embeddings: int
) -> Tensor:
    """
    Returns "table size cumsum", which is information of UVM caching for tables.
    """
    num_cached_tables = get_num_cached_tables(num_tables, cached_tables_ratio)
    np_list = np.arange(0, num_embeddings * num_cached_tables, num_embeddings)
    num_uncached_tables = num_tables - num_cached_tables
    while num_uncached_tables > 0:
        added = random.randint(1, num_uncached_tables)
        pos = random.randint(0, len(np_list) - 1)
        np_list = np.insert(np_list, pos, [np_list[pos]] * added)
        num_uncached_tables -= added
    cache_hash_size_cumsum: Tensor = torch.tensor(np_list).cuda()
    return cache_hash_size_cumsum


def create_embedding_specs(
    num_tables: int,
    cached_tables_ratio: float,
    num_embeddings: int,
    embedding_dims: int,
) -> List[Tuple[str, int, int, SparseType, EmbeddingLocation]]:
    """
    Returns embedding specs to be used with IntNBitTableBatchedEmbeddingBagsCodegen.
    """
    num_cached_tables = get_num_cached_tables(num_tables, cached_tables_ratio)
    num_uncached_tables = num_tables - num_cached_tables
    embedding_specs = []
    for _ in range(min(num_cached_tables, num_uncached_tables)):
        embedding_specs.append(
            (
                "",
                num_embeddings,
                embedding_dims,
                SparseType.INT8,
                EmbeddingLocation.DEVICE,
            )
        )
        embedding_specs.append(
            (
                "",
                num_embeddings,
                embedding_dims,
                SparseType.INT8,
                EmbeddingLocation.MANAGED_CACHING,
            )
        )
    if num_cached_tables > num_uncached_tables:
        for _ in range(num_cached_tables - num_uncached_tables):
            embedding_specs.append(
                (
                    "",
                    num_embeddings,
                    embedding_dims,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
            )
    else:
        for _ in range(num_uncached_tables - num_cached_tables):
            embedding_specs.append(
                (
                    "",
                    num_embeddings,
                    embedding_dims,
                    SparseType.INT8,
                    EmbeddingLocation.DEVICE,
                )
            )
    return embedding_specs


def create_request(
    num_tables: int, num_embeddings: int, batch: int, avg_pooling_factor: int
) -> Tuple[Tensor, Tensor]:
    """
    Returns [indices, offsets], which are inputs of embedding bags.
    """
    indices: Tensor = torch.randint(
        0, num_embeddings, (num_tables * batch * avg_pooling_factor,), dtype=torch.int32
    ).cuda()

    # Pooling factors are intentionally diversified between [1, pf / 2, pf, pf* 2, pf * 4, pf * 8].
    # where pf == avg_pooling_factor.
    pooling_factors = []
    for _ in range(num_tables - 1):
        half_avg_pooling_factor = avg_pooling_factor // 2
        if half_avg_pooling_factor > 0:
            pooling_factors.append(
                random.choices(
                    [
                        1,
                        half_avg_pooling_factor,
                        avg_pooling_factor,
                        2 * avg_pooling_factor,
                        4 * avg_pooling_factor,
                        8 * avg_pooling_factor,
                    ],
                    weights=[5, 10, 15, 1, 1, 3],
                )[0]
            )
        else:
            pooling_factors.append(
                random.choices(
                    [1, avg_pooling_factor, 2 * avg_pooling_factor], weights=[2, 20, 1]
                )[0]
            )

    # Last one is whatever is the remainder.
    curr_total_pooling_factors = sum(pooling_factors)
    pooling_factors.append(num_tables * avg_pooling_factor - curr_total_pooling_factors)

    offsets_list = [0]
    for pooling_factor in pooling_factors:
        if pooling_factor == 1:
            for _ in range(batch):
                offsets_list.append(pooling_factor)
        else:
            finish_offset = offsets_list[-1] + pooling_factor * batch
            for _ in range(batch - 1):
                selected = max(
                    int(random.gauss(pooling_factor, 0.1 * pooling_factor)), 1
                )
                last_offset = offsets_list[-1]
                offsets_list.append(last_offset + selected)
            offsets_list.append(finish_offset)
    offsets: Tensor = torch.tensor(offsets_list, dtype=torch.int32).cuda()
    return (indices, offsets)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--iters", default=100)
@click.option("--num-tables", default=50)
@click.option("--cached-tables-ratio", default=1.0)
@click.option("--batch", default=100)
@click.option("--avg-pooling-factor", default=100)
def linearize_cache_indices(
    iters: int,
    num_tables: int,
    cached_tables_ratio: float,
    batch: int,
    avg_pooling_factor: int,
) -> None:
    num_embeddings: int = 1000000
    cache_hash_size_cumsum = create_table_offsets(
        num_tables, cached_tables_ratio, num_embeddings
    )
    indices, offsets = create_request(
        num_tables, num_embeddings, batch, avg_pooling_factor
    )

    t_ms = benchmark_same_input(
        iters,
        lambda indices, offsets: torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum, indices, offsets
        ),
        indices,
        offsets,
    )
    logging.info(
        f"Across {iters} runs, T: {num_tables}, Cached T: {get_num_cached_tables(num_tables, cached_tables_ratio)}, BS: {batch}, {t_ms * 1.0e3:.0f}us"
    )


@cli.command()
@click.option("--iters", default=100)
@click.option("--num-tables", default=50)
@click.option("--cached-tables-ratio", default=1.0)
@click.option("--batch", default=100)
@click.option("--avg-pooling-factor", default=100)
@click.option("--cache-load-factor", default=0.2)
def lxu_cache_lookup(
    iters: int,
    num_tables: int,
    cached_tables_ratio: float,
    batch: int,
    avg_pooling_factor: int,
    cache_load_factor: float,
) -> None:
    num_embeddings: int = 1000000
    embedding_dims: int = 128

    embedding_specs = create_embedding_specs(
        num_tables, cached_tables_ratio, num_embeddings, embedding_dims
    )

    tbe: nn.Module = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs, cache_load_factor=cache_load_factor
    )
    tbe.fill_random_weights()

    # Imitate execution flow by performing prefetching once.
    indices, offsets = create_request(
        num_tables, num_embeddings, batch, avg_pooling_factor
    )
    tbe.prefetch(indices, offsets)

    linearized_indices = torch.ops.fbgemm.linearize_cache_indices(
        tbe.cache_hash_size_cumsum, indices, offsets
    )

    t_ms = benchmark_same_input(
        iters,
        lambda linearized_indices, lxu_cache_state: torch.ops.fbgemm.lxu_cache_lookup(
            linearized_indices, lxu_cache_state, tbe.total_cache_hash_size
        ),
        linearized_indices,
        tbe.lxu_cache_state,
    )

    # Run once again to obtain cache miss ratio.
    locations = torch.ops.fbgemm.lxu_cache_lookup(
        linearized_indices, tbe.lxu_cache_state, tbe.total_cache_hash_size
    )
    num_invalid_accesses = torch.sum(linearized_indices == tbe.total_cache_hash_size)
    num_valid_accesses = linearized_indices.numel() - num_invalid_accesses
    num_misses = torch.sum(locations == -1) - num_invalid_accesses
    logging.info(
        f"Across {iters} runs, T: {num_tables}, Cached T: {get_num_cached_tables(num_tables, cached_tables_ratio)}, "
        f"BS: {batch}, cache_load_factor: {cache_load_factor}, {t_ms * 1.0e3:.0f}us, "
        f"cache miss: {num_misses.item() / num_valid_accesses * 100}%"
    )


@cli.command()
@click.option("--iters", default=100)
@click.option("--num-tables", default=50)
@click.option("--cached-tables-ratio", default=1.0)
@click.option("--batch", default=100)
@click.option("--avg-pooling-factor", default=100)
@click.option("--cache-load-factor", default=0.2)
def lru_cache_populate_byte(
    iters: int,
    num_tables: int,
    cached_tables_ratio: float,
    batch: int,
    avg_pooling_factor: int,
    cache_load_factor: float,
) -> None:
    num_warm_ups: int = 5
    num_embeddings: int = 1000000
    embedding_dims: int = 128

    embedding_specs = create_embedding_specs(
        num_tables, cached_tables_ratio, num_embeddings, embedding_dims
    )

    cc: nn.Module = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs, cache_load_factor=cache_load_factor
    )
    cc.fill_random_weights()

    warm_up_requests = []
    for _ in range(num_warm_ups):
        indices, offsets = create_request(
            num_tables, num_embeddings, batch, avg_pooling_factor
        )
        warm_up_requests.append(
            torch.ops.fbgemm.linearize_cache_indices(
                cc.cache_hash_size_cumsum, indices, offsets
            )
        )

    requests = []
    for _ in range(iters):
        indices, offsets = create_request(
            num_tables, num_embeddings, batch, avg_pooling_factor
        )
        requests.append(
            torch.ops.fbgemm.linearize_cache_indices(
                cc.cache_hash_size_cumsum, indices, offsets
            )
        )

    timestep: int = 1

    def populate(linear_indices: Tensor) -> None:
        nonlocal timestep
        torch.ops.fbgemm.lru_cache_populate_byte(
            cc.weights_uvm,
            cc.cache_hash_size_cumsum,
            cc.total_cache_hash_size,
            cc.cache_index_table_map,
            cc.weights_offsets,
            cc.weights_tys,
            cc.D_offsets,
            linear_indices,
            cc.lxu_cache_state,
            cc.lxu_cache_weights,
            timestep,
            cc.lxu_state,
        )
        timestep += 1

    for warm_up_request in warm_up_requests:
        populate(warm_up_request)

    t_ms = benchmark_different_inputs(
        populate,
        requests,
    )

    # Replay to figure out UVM access BW, which would be PCIe bound.
    replay_cc: nn.Module = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs, cache_load_factor=cache_load_factor
    )
    replay_cc.fill_random_weights()

    replay_timestep: int = 1

    def replay_populate(linear_indices: Tensor) -> None:
        nonlocal replay_timestep
        torch.ops.fbgemm.lru_cache_populate_byte(
            replay_cc.weights_uvm,
            replay_cc.cache_hash_size_cumsum,
            replay_cc.total_cache_hash_size,
            replay_cc.cache_index_table_map,
            replay_cc.weights_offsets,
            replay_cc.weights_tys,
            replay_cc.D_offsets,
            linear_indices,
            replay_cc.lxu_cache_state,
            replay_cc.lxu_cache_weights,
            replay_timestep,
            replay_cc.lxu_state,
        )
        replay_timestep += 1

    for warm_up_request in warm_up_requests:
        replay_populate(warm_up_request)

    total_rows = 0
    for request in requests:
        prev = replay_cc.lxu_cache_state.clone().detach()
        replay_populate(request)
        after = replay_cc.lxu_cache_state.clone().detach()

        diff = after - prev
        total_rows += diff.count_nonzero().item()

    logging.info(
        f"Across {iters} runs, T: {num_tables}, Cached T: {get_num_cached_tables(num_tables, cached_tables_ratio)}, "
        f"BS: {batch}, cache_load_factor: {cache_load_factor}, {t_ms * 1.0e3:.0f}us, "
        f"BW (just UVM accesses): {total_rows * embedding_dims / iters / t_ms * 1000 / 1024 / 1024} MB/s"
    )


@cli.command()
@click.option("--iters", default=100)
@click.option("--num-tables", default=50)
@click.option("--cached-tables-ratio", default=1.0)
@click.option("--batch", default=100)
@click.option("--avg-pooling-factor", default=100)
@click.option("--cache-load-factor", default=0.2)
def lfu_cache_populate_byte(
    iters: int,
    num_tables: int,
    cached_tables_ratio: float,
    batch: int,
    avg_pooling_factor: int,
    cache_load_factor: float,
) -> None:
    num_warm_ups: int = 5
    num_embeddings: int = 1000000
    embedding_dims: int = 128

    embedding_specs = create_embedding_specs(
        num_tables, cached_tables_ratio, num_embeddings, embedding_dims
    )

    cc: nn.Module = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs,
        cache_load_factor=cache_load_factor,
        cache_algorithm=CacheAlgorithm.LFU,
    )
    cc.fill_random_weights()

    warm_up_requests = []
    for _ in range(num_warm_ups):
        indices, offsets = create_request(
            num_tables, num_embeddings, batch, avg_pooling_factor
        )
        warm_up_requests.append(
            torch.ops.fbgemm.linearize_cache_indices(
                cc.cache_hash_size_cumsum, indices, offsets
            )
        )

    requests = []
    for _ in range(iters):
        indices, offsets = create_request(
            num_tables, num_embeddings, batch, avg_pooling_factor
        )
        requests.append(
            torch.ops.fbgemm.linearize_cache_indices(
                cc.cache_hash_size_cumsum, indices, offsets
            )
        )

    def populate(linear_indices: Tensor) -> None:
        torch.ops.fbgemm.lfu_cache_populate_byte(
            cc.weights_uvm,
            cc.cache_hash_size_cumsum,
            cc.total_cache_hash_size,
            cc.cache_index_table_map,
            cc.weights_offsets,
            cc.weights_tys,
            cc.D_offsets,
            linear_indices,
            cc.lxu_cache_state,
            cc.lxu_cache_weights,
            cc.lxu_state,
        )

    for warm_up_request in warm_up_requests:
        populate(warm_up_request)

    t_ms = benchmark_different_inputs(
        populate,
        requests,
    )

    # Replay to figure out UVM access BW, which would be PCIe bound.
    replay_cc: nn.Module = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs,
        cache_load_factor=cache_load_factor,
        cache_algorithm=CacheAlgorithm.LFU,
    )
    replay_cc.fill_random_weights()

    def replay_populate(linear_indices: Tensor) -> None:
        torch.ops.fbgemm.lfu_cache_populate_byte(
            replay_cc.weights_uvm,
            replay_cc.cache_hash_size_cumsum,
            replay_cc.total_cache_hash_size,
            replay_cc.cache_index_table_map,
            replay_cc.weights_offsets,
            replay_cc.weights_tys,
            replay_cc.D_offsets,
            linear_indices,
            replay_cc.lxu_cache_state,
            replay_cc.lxu_cache_weights,
            replay_cc.lxu_state,
        )

    for warm_up_request in warm_up_requests:
        replay_populate(warm_up_request)

    total_rows = 0
    for request in requests:
        prev = replay_cc.lxu_cache_state.clone().detach()
        replay_populate(request)
        after = replay_cc.lxu_cache_state.clone().detach()

        diff = after - prev
        total_rows += diff.count_nonzero().item()

    logging.info(
        f"Across {iters} runs, T: {num_tables}, Cached T: {get_num_cached_tables(num_tables, cached_tables_ratio)}, "
        f"BS: {batch}, cache_load_factor: {cache_load_factor}, {t_ms * 1.0e3:.0f}us, "
        f"BW (just UVM accesses): {total_rows * embedding_dims / iters / t_ms * 1000 / 1024 / 1024} MB/s"
    )


if __name__ == "__main__":
    cli()
