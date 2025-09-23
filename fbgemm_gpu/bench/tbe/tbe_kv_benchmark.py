# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import gc
import logging
import time
from typing import Callable

import click
import numpy as np
import psutil
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import benchmark_requests
from fbgemm_gpu.tbe.cache import KVEmbeddingInference
from fbgemm_gpu.tbe.utils import generate_requests, round_up, TBERequest

OptionCommandType = Callable[..., Callable[..., None]]

iters: OptionCommandType = click.option(
    "--iters",
    default=200,
    type=int,
    help="Number of iterations to benchmark",
)
num_embeddings: OptionCommandType = click.option(
    "--num-embeddings",
    default=int(1e8),
    type=int,
    help="Number of embedding to benchmark",
)
dim: OptionCommandType = click.option(
    "--dim", default=256, type=int, help="Dimension of embedding to benchmark"
)
num_tables: OptionCommandType = click.option(
    "--num-tables", default=4, type=int, help="Number of tables to benchmark"
)
output_dtype: OptionCommandType = click.option(
    "--output-dtype", type=SparseType, default=SparseType.FP16
)
weights_precision: OptionCommandType = click.option(
    "--weights-precision", type=SparseType, default=SparseType.INT8
)
batch_size: OptionCommandType = click.option("--batch-size", default=128)
bag_size: OptionCommandType = click.option("--bag-size", default=1)
mixed_dim: OptionCommandType = click.option("--mixed-dim", is_flag=True, default=False)
tbe_class: OptionCommandType = click.option(
    "--tbe-class", type=str, default="KVEmbeddingInference"
)


TBE_CLASS_MAP: dict[str, type[IntNBitTableBatchedEmbeddingBagsCodegen]] = {
    "KVEmbeddingInference": KVEmbeddingInference,
    "IntNBitTableBatchedEmbeddingBagsCodegen": IntNBitTableBatchedEmbeddingBagsCodegen,
}


@click.group()
def cli() -> None:
    pass


@cli.command()
@iters
@num_embeddings
@dim
@num_tables
@output_dtype
@weights_precision
@batch_size
@bag_size
@mixed_dim
@tbe_class
def forward_benchmark(
    iters: int,
    num_embeddings: int,
    dim: int,
    num_tables: int,
    output_dtype: SparseType,
    weights_precision: SparseType,
    batch_size: int,
    bag_size: int,
    mixed_dim: bool,
    tbe_class: str,
) -> None:
    logging.info(
        f"Running forward benchmark with {iters} iterations, {num_embeddings} embeddings, {dim} dim, {num_tables} tables, {output_dtype} output dtype, {weights_precision} weights precision, {batch_size} batch"
    )

    stats = []

    if mixed_dim:
        dimentions = [
            round_up(np.random.randint(low=int(0.5 * dim), high=int(1.5 * dim)), 4)
            for _ in range(num_tables)
        ]
    else:
        dimentions = [dim] * num_tables

    process = psutil.Process()

    clazz = TBE_CLASS_MAP[tbe_class]

    time.sleep(5)
    mem_util_before = process.memory_info().rss / (1024 * 1024)
    logging.info(f"Memory util before emb init: {mem_util_before} MB")
    tbe = clazz(
        [
            (
                "",
                num_embeddings,
                d,
                weights_precision,
                EmbeddingLocation.HOST,
            )
            for d in dimentions
        ],
        output_dtype=output_dtype,
        device="cpu",
    )
    tbe.fill_random_weights()

    gc.collect()
    time.sleep(5)
    mem_util_after = process.memory_info().rss / (1024 * 1024)
    logging.info(f"Memory util after emb fill: {mem_util_after} MB")
    logging.info(f"Memory util diff: {mem_util_after - mem_util_before} MB")

    for batch_size in [10240, 20480, 40960]:
        requests = generate_requests(
            iters,
            batch_size,
            num_tables,
            bag_size,
            num_embeddings,
            use_cpu=True,
        )

        requests_cpu = [
            TBERequest(
                req.indices.int().cpu(),
                req.offsets.int().cpu(),
                req.per_sample_weights,
            )
            for req in requests
        ]

        logging.info(f"Running forward benchmark with {len(requests_cpu)} requests")
        time_per_iter = benchmark_requests(
            requests_cpu,
            lambda indices, offsets, per_sample_weights: tbe.forward(
                indices.int().cpu(),
                offsets.int().cpu(),
                per_sample_weights,
            ),
            num_warmups=10,
        )
        logging.info(f"{clazz} CPU Time: {time_per_iter * 1.0e6:.0f}us")
        stats.append(
            [
                clazz,
                num_tables,
                batch_size,
                f"{time_per_iter * 1.0e6:.0f}us",
                f"{mem_util_after - mem_util_before} MB",
            ]
        )
    for stat in stats:
        logging.info(stat)


if __name__ == "__main__":
    cli()
