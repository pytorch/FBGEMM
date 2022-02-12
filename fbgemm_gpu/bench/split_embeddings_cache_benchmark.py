# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import random
from typing import Tuple

import click
import numpy as np
import torch
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:split_table_batched_embeddings"
    )


ASSOC: int = 32


# pyre-ignore
def benchmark_torch_function(iters: int, f, *args) -> float:
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


def create_table_offsets(
    num_tables: int, cached_tables_ratio: float, num_embeddings: int
) -> Tensor:
    """
    Returns "table size cumsum", which is information of UVM caching for tables.
    """
    num_cached_tables = round(num_tables * cached_tables_ratio)
    np_list = np.arange(0, num_embeddings * num_cached_tables, num_embeddings)
    num_uncached_tables = num_tables - num_cached_tables
    while num_uncached_tables > 0:
        added = random.randint(1, num_uncached_tables)
        pos = random.randint(0, len(np_list) - 1)
        np_list = np.insert(np_list, pos, [np_list[pos]] * added)
        num_uncached_tables -= added
    cache_hash_size_cumsum: Tensor = torch.tensor(np_list).cuda()
    return cache_hash_size_cumsum


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

    t_ms = benchmark_torch_function(
        iters,
        lambda indices, offsets: torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum, indices, offsets
        ),
        indices,
        offsets,
    )
    logging.info(
        f"Across {iters} runs, T: {num_tables} BS: {batch}, {t_ms * 1.0e3:.0f}us"
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
    cache_hash_size_cumsum = create_table_offsets(
        num_tables, cached_tables_ratio, num_embeddings
    )
    indices, offsets = create_request(
        num_tables, num_embeddings, batch, avg_pooling_factor
    )

    linearized_indices = torch.ops.fbgemm.linearize_cache_indices(
        cache_hash_size_cumsum, indices, offsets
    )

    lxu_cache_state: Tensor = torch.empty(
        math.ceil(cache_hash_size_cumsum[-1] * cache_load_factor / ASSOC),
        ASSOC,
        device="cuda",
        dtype=torch.int64,
    ).fill_(-1)

    t_ms = benchmark_torch_function(
        iters,
        lambda linearized_indices, lxu_cache_state: torch.ops.fbgemm.lxu_cache_lookup(
            linearized_indices, lxu_cache_state
        ),
        linearized_indices,
        lxu_cache_state,
    )
    logging.info(
        f"Across {iters} runs, T: {num_tables} BS: {batch}, {t_ms * 1.0e3:.0f}us"
    )


if __name__ == "__main__":
    cli()
