# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random

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


@click.command()
@click.option("--iters", default=100)
@click.option("--num-tables", default=50)
@click.option("--batch", default=100)
@click.option("--avg-pooling-factor", default=400)
def main(
    iters: int,
    num_tables: int,
    batch: int,
    avg_pooling_factor: int,
) -> None:
    num_embeddings: int = 1000000
    cache_hash_size_cumsum: Tensor = torch.tensor(
        np.arange(0, num_embeddings * num_tables, num_embeddings)
    ).cuda()
    indices: Tensor = torch.randint(
        0, num_embeddings, (num_tables * batch * avg_pooling_factor,), dtype=torch.int32
    ).cuda()

    # Pooling factors are intentionally diversified between provided [1, avg_pooling_factor, avg_pooling_factor * 2].
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
                    ],
                    weights=[2, 10, 20, 1],
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


if __name__ == "__main__":
    main()
