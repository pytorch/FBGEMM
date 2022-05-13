# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import random

import click
import fbgemm_gpu
import numpy as np
import torch

logging.basicConfig(level=logging.DEBUG)

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--world-size", default=128)
@click.option("--num-tables", default=10)
@click.option("--min-len", default=10000)
@click.option("--max-len", default=20000)
def device(
    world_size: int,
    num_tables: int,
    min_len: int,
    max_len: int,
) -> None:
    lengths = torch.randint(min_len, max_len, size=(num_tables * world_size,))
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    permute = list(range(num_tables * world_size))
    random.shuffle(permute)
    permute_tensor = torch.tensor(permute)
    permuted_length = torch.index_select(lengths, 0, permute_tensor)
    permuted_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(permuted_length)
    jagged_size = offsets[-1]

    if torch.cuda.is_available():
        permute_tensor = permute_tensor.cuda()
        offsets = offsets.cuda()
        permuted_offsets = permuted_offsets.cuda()

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.expand_into_jagged_permute,
        (permute_tensor, offsets, permuted_offsets, jagged_size),
    )

    num_bytes = (
        permute_tensor.numel() * permute_tensor.element_size()
        + offsets.numel() * offsets.element_size()
        + permuted_offsets.numel() * permuted_offsets.element_size()
        + output.numel() * output.element_size()
    )
    logging.info(f"expand_into_jagged_permute {time} sec {num_bytes / time / 1e9} GB/s")


@cli.command()
@click.option("--row-size", default=25600)
@click.option("--batch-size", default=4096)
@click.option("--unique-batch-size", default=1024)
@click.option("--input-precision", type=str, default="fp32")
def batch_reuse_index_select_device(
    row_size: int, batch_size: int, unique_batch_size: int, input_precision: str
) -> None:
    # A function for generating indices in batch_reuse
    # pyre-fixme[11]: Annotation `array` is not defined as a type.
    def gen_inverse_index(curr_size: int, final_size: int) -> np.array:
        inverse_index = list(range(curr_size))
        np_arr = np.array(inverse_index)
        for _ in range(final_size - curr_size):
            inverse_index.append(np.random.randint(0, curr_size))
            np_arr = np.array(inverse_index)
            np.random.shuffle(np_arr)
        return np_arr

    dtype = torch.float
    if input_precision == "fp32":
        dtype = torch.float
    elif input_precision == "fp16":
        dtype = torch.half
    else:
        raise RuntimeError(f"Does not support data type {input_precision}")

    indices = torch.cuda.IntTensor(gen_inverse_index(unique_batch_size, batch_size))

    input = torch.rand(unique_batch_size, row_size, dtype=dtype, device="cuda")
    input.requires_grad = True
    num_bytes = 2 * batch_size * row_size * input.element_size()
    time, output = benchmark_torch_function(
        torch.ops.fbgemm.index_select_dim0, (input, indices, 0, unique_batch_size)
    )
    logging.info(
        f"index_select_dim0 forward: {dtype}, {num_bytes} bytes read/write, {time * 1e3} ms, {num_bytes / time / 1e9} GB/s"
    )

    grad = torch.rand_like(output, dtype=dtype, device="cuda")
    num_bytes = (input.numel() + output.numel()) * input.element_size()
    time, _ = benchmark_torch_function(
        functools.partial(output.backward, retain_graph=True), (grad,)
    )
    logging.info(
        f"index_select_dim0 backward: {dtype}, {num_bytes} bytes read/write, {time * 1e3} ms, {num_bytes / time / 1e9} GB/s"
    )


if __name__ == "__main__":
    cli()
