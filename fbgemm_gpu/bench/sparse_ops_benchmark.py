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

    # pyre-fixme[16]: Module `cuda` has no attribute `IntTensor`.
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


@cli.command()
@click.option("--max-seq-length", default=500)
@click.option("--batch-size", default=4096)
@click.option("--num-cols", default=256)
@click.option("--num-jagged-tensor-rows", default=4096)
@click.option("--num-zero-padding", default=1024)
@click.option("--index-dtype", type=click.Choice(["int", "long"]), default="int")
@click.option(
    "--jagged-tensor-dtype", type=click.Choice(["float", "half"]), default="float"
)
def jagged_index_select_2d_bench(
    max_seq_length: int,
    batch_size: int,
    num_cols: int,
    num_jagged_tensor_rows: int,
    num_zero_padding: int,
    index_dtype: str,
    jagged_tensor_dtype: str,
) -> None:
    def jagged_index_select_2d_ref(
        values: torch.Tensor, lengths: torch.Tensor, inverse_lookup: torch.Tensor
    ) -> torch.Tensor:
        offsets = torch.ops.fbgemm.asynchronous_exclusive_cumsum(lengths)
        end_offsets = offsets + lengths
        full_start_offset = torch.index_select(offsets, 0, inverse_lookup)
        full_end_offset = torch.index_select(end_offsets, 0, inverse_lookup)
        index_ranges = torch.stack(
            (full_start_offset, full_end_offset), dim=0
        ).transpose(0, 1)

        to_be_merged_tensors = []
        for row in index_ranges:
            to_be_merged_tensors.append(torch.arange(row[0], row[1], device="cuda"))
        all_indices = torch.cat(to_be_merged_tensors, dim=0)
        new_embeddings = torch.index_select(values, 0, all_indices)
        return new_embeddings

    index_t = {"int": torch.int, "long": torch.long}[index_dtype]
    scalar_t = {"float": torch.float, "half": torch.half}[jagged_tensor_dtype]

    lengths = torch.randint(
        low=0,
        high=max_seq_length,
        size=(num_jagged_tensor_rows,),
        dtype=index_t,
        device="cuda",
    )
    indices, _ = torch.sort(
        torch.randint(
            low=0,
            high=num_jagged_tensor_rows,
            size=(batch_size,),
            dtype=index_t,
            device="cuda",
        )
    )
    values = torch.rand(
        int(lengths.sum().item()), num_cols, dtype=scalar_t, device="cuda"
    )
    values.requires_grad = True

    indices[batch_size - num_zero_padding :] = 0

    time, (output, _) = benchmark_torch_function(
        torch.ops.fbgemm.jagged_index_select,
        (values, lengths, indices),
        num_warmups=10,
        iters=100,
    )
    time_ref, output_ref = benchmark_torch_function(
        jagged_index_select_2d_ref,
        (values, lengths, indices),
        num_warmups=10,
        iters=100,
    )
    logging.info(
        f"jagged_index_select_2d_bench "
        f"(max_seq_length={max_seq_length}, "
        f"batch_size={batch_size}, "
        f"num_cols={num_cols}, "
        f"num_jagged_tensor_rows={num_jagged_tensor_rows}, "
        f"num_zero_padding={num_zero_padding}, "
        f"index_dtype={index_dtype}, "
        f"jagged_tensor_dtype={jagged_tensor_dtype})"
    )
    logging.info(f"forward: fbgemm {time * 1e3:.3f} ms, ref {time_ref * 1e3:.3f} ms")

    grad = torch.rand_like(output)
    time, _ = benchmark_torch_function(
        functools.partial(output.backward, retain_graph=True),
        (grad,),
        num_warmups=10,
        iters=100,
    )
    time_ref, _ = benchmark_torch_function(
        functools.partial(output_ref.backward, retain_graph=True),
        (grad,),
        num_warmups=10,
        iters=100,
    )
    logging.info(f"backward: fbgemm {time * 1e3:.3f} ms, ref {time_ref * 1e3:.3f} ms")


if __name__ == "__main__":
    cli()
