# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import contextlib
import functools
import logging
import math
import random
from typing import List

import click
import fbgemm_gpu
import numpy as np
import torch

from torch.profiler import profile

logging.basicConfig(level=logging.DEBUG)

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_hip")
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:index_select_ops")


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


@cli.command()
@click.option("--row-size", default=512)
@click.option("--batch-size", default=4096)
@click.option("--unique-batch-size", default=1024)
@click.option("--input-precision", type=str, default="fp32")
@click.option("--sort-indices", type=bool, default=True)
@click.option("--num-groups", default=32)
def group_index_select_2d_bench(
    row_size: int,
    batch_size: int,
    unique_batch_size: int,
    input_precision: str,
    sort_indices: bool,
    num_groups: int,
) -> None:
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

    offset_indices_group = []
    indices_group = []
    for i in range(num_groups):
        # pyre-fixme[16]: Module `cuda` has no attribute `IntTensor`.
        indices = torch.cuda.IntTensor(gen_inverse_index(unique_batch_size, batch_size))
        if sort_indices:
            indices, _ = indices.sort()
        indices_group.append(indices)
        indices = torch.add(indices, batch_size * i)
        offset_indices_group.append(indices)

    offset_indices = torch.concat(offset_indices_group)

    input = torch.rand(num_groups * batch_size, row_size, dtype=dtype, device="cuda")
    input.requires_grad = True

    num_bytes = 2 * batch_size * row_size * input.element_size() * num_groups

    bench_kwargs = {"num_warmups": 10, "iters": 100}

    # Benchmark forward
    time_ref, output_ref = benchmark_torch_function(
        torch.index_select, (input, 0, offset_indices), **bench_kwargs
    )

    input_group = input.split(batch_size, 0)
    time, output_group = benchmark_torch_function(
        torch.ops.fbgemm.group_index_select_dim0,
        (input_group, indices_group),
        **bench_kwargs,
    )
    logging.info(
        f"forward: PyTorch batch {time_ref:.5f} sec ({num_bytes / time_ref / 1e9:.5f} GB/s), "
        f"fbgemm group {time:5f} sec ({num_bytes / time / 1e9:.5f} GB/s)"
    )

    # Benchmark backward
    grad = torch.rand_like(output_ref)
    time_ref, _ = benchmark_torch_function(
        functools.partial(output_ref.backward, retain_graph=True),
        (grad,),
        **bench_kwargs,
    )

    cat_output = torch.cat(output_group)
    time, _ = benchmark_torch_function(
        functools.partial(cat_output.backward, retain_graph=True),
        (grad,),
        **bench_kwargs,
    )
    logging.info(
        f"backward: PyTorch batch {time_ref:.5f} sec ({num_bytes / time_ref / 1e9:.5f} GB/s), "
        f"fbgemm group {time:.5f} sec ({num_bytes / time / 1e9:.5f} GB/s)"
    )


@cli.command()
@click.option("--num-vecs", default=2048)
@click.option("--num-entries-per-vec", default=1024)
@click.option("--dtype", type=str, default="long")
def asynchronous_complete_cumsum_2d_bench(
    num_vecs: int,
    num_entries_per_vec: int,
    dtype: str,
) -> None:
    # Reference code from TorchRec https://github.com/pytorch/torchrec/pull/332
    @torch.jit.script
    def asynchronous_complete_cumsum_2d_ref(lengths: torch.Tensor) -> torch.Tensor:
        (f, b) = lengths.shape
        offsets_0 = lengths.new_zeros((f, 1))
        offsets_1 = torch.cumsum(lengths, dim=-1).to(lengths.dtype)
        offsets = torch.cat([offsets_0, offsets_1], dim=-1)
        return offsets

    assert dtype == "int" or dtype == "long", "Only int and long are supported"
    index_dtype = torch.int64 if dtype == "long" else torch.int32

    x = torch.randint(low=0, high=100, size=(num_vecs, num_entries_per_vec)).type(
        index_dtype
    )
    x = x.cuda()

    time_ref, _ = benchmark_torch_function(
        asynchronous_complete_cumsum_2d_ref, (x,), num_warmups=100, iters=1000
    )

    time, _ = benchmark_torch_function(
        torch.ops.fbgemm.asynchronous_complete_cumsum, (x,), num_warmups=100, iters=1000
    )

    logging.info(
        f"asynchronous_complete_cumsum_2d_bench: input shape {x.shape}, dtype {dtype}"
    )
    logging.info(f"ref time: {time_ref:.5f} sec")
    logging.info(f"fbgemm_gpu time: {time:.5f} sec")


@cli.command()
@click.option("--batch-size", default=8192)
@click.option("--table-size", default=20)
@click.option("--length", default=50)
@click.option("--num-ads", default=100)
@click.option("--dtype", type=click.Choice(["float", "long"]), default="long")
@click.option("--itype", type=click.Choice(["int", "long"]), default="int")
@click.option("--broadcast-indices", type=bool, default=True)
@click.option("--device", type=str, default="cpu")
def reorder_batched_ad_indices_bench(
    batch_size: int,
    table_size: int,
    length: int,
    num_ads: int,
    dtype: str,
    itype: str,
    broadcast_indices: bool,
    device: str,
) -> None:
    assert dtype == "float" or dtype == "long", "Only int and long are supported"
    data_type = torch.int64 if dtype == "long" else torch.float
    data_size = 8 if dtype == "long" else 4

    assert itype == "int" or itype == "long", "Only int and long are supported"
    index_type = torch.int64 if itype == "long" else torch.int32

    if broadcast_indices:
        cat_ad_indices = (
            torch.randint(
                low=0,
                high=100,
                size=(batch_size * table_size * length,),
            )
            .int()
            .to(device)
            .to(data_type)
        )
        cat_ad_lengths = (
            torch.cat(
                [
                    torch.tensor([length for _ in range(table_size)])
                    for _ in range(batch_size)
                ],
                0,
            )
            .int()
            .to(device)
        )
    else:
        cat_ad_indices = (
            torch.randint(
                low=0,
                high=100,
                size=(batch_size * table_size * num_ads * length,),
            )
            .int()
            .to(device)
            .to(data_type)
        )
        cat_ad_lengths = (
            torch.cat(
                [
                    torch.tensor([length for _ in range(table_size * num_ads)])
                    for _ in range(batch_size)
                ],
                0,
            )
            .int()
            .to(device)
        )

    batch_offsets = (
        torch.tensor([num_ads * b for b in range(batch_size + 1)]).int().cuda()
    ).to(device)
    num_ads_in_batch = batch_size * num_ads
    reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
        cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
    ).to(device)

    cat_ad_offsets = (
        torch.ops.fbgemm.asynchronous_complete_cumsum(cat_ad_lengths)
        .to(index_type)
        .to(device)
    )
    reordered_cat_ad_offsets = (
        torch.ops.fbgemm.asynchronous_complete_cumsum(reordered_cat_ad_lengths)
        .to(index_type)
        .to(device)
    )
    time, _ = benchmark_torch_function(
        torch.ops.fbgemm.reorder_batched_ad_indices,
        (
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
            batch_size * table_size * num_ads * length,
        ),
        num_warmups=100,
        iters=1000,
    )
    num_bytes = batch_size * table_size * (num_ads + 1) * length * data_size
    logging.info(
        f"fbgemm_gpu time: {time * 1000:.5f} ms ({num_bytes / time / 1e9:.5f} GB/s)"
    )


@cli.command()
@click.option("--batch-size", default=8192)
@click.option("--table-size", default=20)
@click.option("--length", default=50)
@click.option("--num-ads", default=100)
@click.option("--broadcast-indices", type=bool, default=True)
@click.option("--device", type=str, default="cpu")
def reorder_batched_ad_lengths_bench(
    batch_size: int,
    table_size: int,
    length: int,
    num_ads: int,
    broadcast_indices: bool,
    device: str,
) -> None:
    if broadcast_indices:
        cat_ad_lengths = (
            torch.cat(
                [
                    torch.tensor([length for _ in range(table_size)])
                    for _ in range(batch_size)
                ],
                0,
            )
            .int()
            .to(device)
        )
    else:
        cat_ad_lengths = (
            torch.cat(
                [
                    torch.tensor([length for _ in range(table_size * num_ads)])
                    for _ in range(batch_size)
                ],
                0,
            )
            .int()
            .to(device)
        )

    batch_offsets = (
        torch.tensor([num_ads * b for b in range(batch_size + 1)]).int().cuda()
    ).to(device)
    num_ads_in_batch = batch_size * num_ads
    time, _ = benchmark_torch_function(
        torch.ops.fbgemm.reorder_batched_ad_lengths,
        (
            cat_ad_lengths,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
        ),
        num_warmups=100,
        iters=1000,
    )
    num_bytes = batch_size * table_size * (num_ads + 1) * length * 4
    logging.info(
        f"fbgemm_gpu time: {time * 1000:.5f} ms ({num_bytes / time / 1e9:.5f} GB/s)"
    )


@cli.command()
@click.option(
    "--batch-size", default=32
)  # 32 is the representative inference batch size
@click.option("--table-size", default=20)
@click.option("--length", default=512)  # long sequence representative case
@click.option("--num-items", default=100)
@click.option("--dim", default=256)
@click.option("--dtype", type=click.Choice(["half", "float"]), default="half")
@click.option("--itype", type=click.Choice(["int", "long"]), default="int")
@click.option("--device", type=str, default="cpu")
def reorder_batched_sequence_embeddings_bench(
    batch_size: int,
    table_size: int,
    length: int,
    num_items: int,
    dim: int,
    dtype: str,
    itype: str,
    device: str,
) -> None:
    assert (
        dtype == "float" or dtype == "half"
    ), "Only 32/16bits floating point number are supported"
    data_type = torch.half if dtype == "half" else torch.float

    assert itype == "int" or itype == "long", "Only int and long are supported"
    index_type = torch.int64 if itype == "long" else torch.int32

    cat_sequence_embeddings = torch.random(
        size=(batch_size * table_size * num_items * length * dim),
        dtype=data_type,
    ).to(device)
    cat_sequence_embeddings_lengths = (
        torch.cat(
            [
                torch.tensor([length for _ in range(table_size * num_items)])
                for _ in range(batch_size)
            ],
            0,
        )
        .to(index_type)
        .to(device)
    )

    batch_offsets = (
        (torch.tensor([num_items * b for b in range(batch_size + 1)]).cuda())
        .to(index_type)
        .to(device)
    )
    num_items_in_batch = batch_size * num_items
    reordered_cat_sequence_embeddings_lengths = (
        torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_sequence_embeddings_lengths,
            batch_offsets,
            num_items_in_batch,
        ).to(device)
    )

    cat_sequence_embeddings_offsets = (
        torch.ops.fbgemm.asynchronous_complete_cumsum(cat_sequence_embeddings_lengths)
        .to(index_type)
        .to(device)
    )
    reordered_cat_sequence_embeddings_offsets = (
        torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_sequence_embeddings_lengths
        )
        .to(index_type)
        .to(device)
    )
    time, _ = benchmark_torch_function(
        torch.ops.fbgemm.reorder_batched_sequence_embeddings,
        (
            cat_sequence_embeddings_offsets,
            cat_sequence_embeddings,
            reordered_cat_sequence_embeddings_offsets,
            batch_offsets,
            num_items_in_batch,
            batch_size * table_size * num_items * length,
        ),
        num_warmups=100,
        iters=1000,
    )
    num_bytes = (
        batch_size
        * table_size
        * num_items
        * length
        * cat_sequence_embeddings.element_size()
    )
    logging.info(
        f"fbgemm_gpu time: {time * 1000:.5f} ms ({num_bytes / time / 1e9:.5f} GB/s)"
    )


@cli.command()
@click.option("--num-inputs", default=1024)
@click.option("--rows", default=100)
@click.option("--columns", default=128)
@click.option("--num-indices", default=2048)
@click.option("--timeline", is_flag=True, default=False)
def index_select_bench(
    num_inputs: int, rows: int, columns: int, num_indices: int, timeline: bool
) -> None:
    input_rows = [rows] * num_inputs
    input_columns = [columns] * num_inputs
    input_num_indices = [num_indices] * num_inputs
    inputs = [
        torch.rand(rows, cols, dtype=torch.float, device="cuda")
        for rows, cols in zip(input_rows, input_columns)
    ]
    for i in range(len(inputs)):
        inputs[i].requires_grad = True
    indices = [
        torch.randint(low=0, high=rows, size=(num,), dtype=torch.long, device="cuda")
        for num, rows in zip(input_num_indices, input_rows)
    ]

    concat_inputs = torch.concat([input.flatten().clone().detach() for input in inputs])
    concat_inputs.requires_grad = True
    concat_indices = torch.concat(indices)

    gis_inputs = [input.clone().detach() for input in inputs]
    for i in range(len(gis_inputs)):
        gis_inputs[i].requires_grad = True

    # Add optimizer to perform zero grad in order to reset gradients
    # before the accumulation phase
    optim_index: torch.optim.Optimizer = torch.optim.SGD(inputs, lr=0.1)
    optim_batch: torch.optim.Optimizer = torch.optim.SGD([concat_inputs], lr=0.1)
    optim_group: torch.optim.Optimizer = torch.optim.SGD(gis_inputs, lr=0.1)

    def index_select_fwd_ref(
        inputs: List[torch.Tensor], indices: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        outputs = []
        for input, index in zip(inputs, indices):
            optim_index.zero_grad()
            outputs.append(torch.index_select(input, 0, index))
        return outputs

    def index_select_bwd_ref(
        outputs: List[torch.Tensor], grads: List[torch.Tensor]
    ) -> None:
        for output, grad in zip(outputs, grads):
            optim_index.zero_grad()
            output.backward(grad, retain_graph=True)

    def batch_index_select_fwd(
        concat_inputs: List[torch.Tensor],
        concat_indices: List[int],
        input_num_indices: List[int],
        input_rows: List[int],
        input_columns: List[int],
    ) -> torch.autograd.Variable:
        optim_batch.zero_grad()
        return torch.ops.fbgemm.batch_index_select_dim0(
            concat_inputs, concat_indices, input_num_indices, input_rows, input_columns
        )

    def group_index_select_fwd(
        gis_inputs: List[torch.Tensor], indices: List[int]
    ) -> torch.autograd.Variable:
        optim_group.zero_grad()
        return torch.ops.fbgemm.group_index_select_dim0(gis_inputs, indices)

    def batch_group_index_select_bwd(
        output: torch.autograd.Variable,
        grads: List[torch.Tensor],
        optim: torch.optim.Optimizer,
    ) -> torch.autograd.Variable:
        optim.zero_grad()
        return output.backward(grads, retain_graph=True)

    bench_kwargs = {"num_warmups": 10, "iters": 10 if timeline else 100}
    profile_ctx = profile if timeline else contextlib.nullcontext

    with profile_ctx() as prof:
        time_pyt, out_pyt = benchmark_torch_function(
            index_select_fwd_ref,
            (inputs, indices),
            **bench_kwargs,
        )

        time_bis, out_bis = benchmark_torch_function(
            batch_index_select_fwd,
            (
                concat_inputs,
                concat_indices,
                input_num_indices,
                input_rows,
                input_columns,
            ),
            **bench_kwargs,
        )

        time_gis, out_gis = benchmark_torch_function(
            group_index_select_fwd,
            (gis_inputs, indices),
            **bench_kwargs,
        )

    if timeline:
        prof.export_chrome_trace("index_select_fwd_trace.json")

    grads = [torch.rand_like(out) for out in out_pyt]
    concat_grads = torch.concat([grad.flatten() for grad in grads])
    concat_out_gis = torch.concat([out.flatten() for out in out_gis])

    with profile_ctx() as prof:
        time_bwd_pyt, _ = benchmark_torch_function(
            index_select_bwd_ref,
            (out_pyt, grads),
            **bench_kwargs,
        )

        time_bwd_bis, _ = benchmark_torch_function(
            batch_group_index_select_bwd,
            (
                out_bis,
                concat_grads,
                optim_batch,
            ),
            **bench_kwargs,
        )

        time_bwd_gis, _ = benchmark_torch_function(
            batch_group_index_select_bwd,
            (
                concat_out_gis,
                concat_grads,
                optim_group,
            ),
            **bench_kwargs,
        )

    if timeline:
        prof.export_chrome_trace("index_select_bwd_trace.json")

    logging.info(
        f"torch.index_select forward {time_pyt * 1e6:.2f} us, backward {time_bwd_pyt * 1e6:.2f} us\n"
        f"torch.ops.fbgemm.batch_index_select forward {time_bis * 1e6:.2f} us, backward {time_bwd_bis * 1e6:.2f} us\n"
        f"torch.ops.fbgemm.group_index_select_dim0 forward {time_gis * 1e6:.2f} us, backward {time_bwd_gis * 1e6:.2f} us"
    )


@cli.command()
@click.option("--batch-size", default=8192)
@click.option("--table-size", default=20)
@click.option("--length", default=50)
@click.option("--num-ads", default=100)
@click.option("--dtype", type=click.Choice(["float", "long"]), default="long")
@click.option("--itype", type=click.Choice(["int", "long"]), default="int")
@click.option("--broadcast-indices", type=bool, default=True)
def cat_reorder_batched_ad_indices_bench(
    batch_size: int,
    table_size: int,
    length: int,
    num_ads: int,
    dtype: str,
    itype: str,
    broadcast_indices: bool,
) -> None:
    assert dtype == "float" or dtype == "long", "Only int and long are supported"
    data_type = torch.int64 if dtype == "long" else torch.float
    data_size = 8 if dtype == "long" else 4

    assert itype == "int" or itype == "long", "Only int and long are supported"

    if broadcast_indices:
        ad_indices = [
            (
                torch.randint(
                    low=0,
                    high=100,
                    size=(table_size * length,),
                )
                .int()
                .to(data_type)
            )
            for _ in range(batch_size)
        ]
        ad_lengths = [
            torch.tensor([length for _ in range(table_size)]).int()
            for _ in range(batch_size)
        ]
    else:
        ad_indices = [
            (
                torch.randint(
                    low=0,
                    high=100,
                    size=(table_size * num_ads * length,),
                )
                .int()
                .to(data_type)
            )
            for _ in range(batch_size)
        ]
        ad_lengths = [
            torch.tensor([length for _ in range(table_size * num_ads)]).int()
            for _ in range(batch_size)
        ]

    batch_offsets = torch.tensor([num_ads * b for b in range(batch_size + 1)]).int()
    num_ads_in_batch = batch_size * num_ads

    # pyre-ignore
    def pass_1(ad_indices, ad_lengths, batch_offsets, num_ads_in_batch):
        cat_ad_lengths = torch.cat(ad_lengths, 0).to("cuda", non_blocking=True)
        cat_ad_indices = torch.cat(ad_indices, 0).to("cuda", non_blocking=True)
        batch_offsets = batch_offsets.to("cuda", non_blocking=True)
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
        )
        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(cat_ad_lengths)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        )
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
            batch_size * table_size * num_ads * length,
        )

        return reordered_cat_ad_indices, reordered_cat_ad_lengths

    # process length on device and process indice on device
    # pyre-ignore
    def pass_2(ad_indices, ad_lengths, batch_offsets, num_ads_in_batch):
        cat_ad_lengths = torch.cat(ad_lengths, 0)

        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
        )
        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(cat_ad_lengths)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        )
        cat_ad_indices = torch.cat(ad_indices, 0)

        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets.to("cuda", non_blocking=True),
            cat_ad_indices.to("cuda", non_blocking=True),
            reordered_cat_ad_offsets.to("cuda", non_blocking=True),
            batch_offsets.to("cuda", non_blocking=True),
            num_ads_in_batch,
            broadcast_indices,
            batch_size * table_size * num_ads * length,
        )

        return reordered_cat_ad_indices, reordered_cat_ad_lengths.to(
            "cuda", non_blocking=True
        )

    # minimize GPU workload + unfused cat + reorder
    # pyre-ignore
    def pass_3(ad_indices, ad_lengths, batch_offsets, num_ads_in_batch):
        cat_ad_lengths = torch.cat(ad_lengths, 0)
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
        )

        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(cat_ad_lengths)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        )
        cat_ad_indices = torch.cat(ad_indices, 0)

        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
            batch_size * table_size * num_ads * length,
        )

        return reordered_cat_ad_indices.to(
            "cuda", non_blocking=True
        ), reordered_cat_ad_lengths.to("cuda", non_blocking=True)

    # minimize GPU workload + fuse cat + reorder
    # pyre-ignore
    def pass_4(ad_indices, ad_lengths, batch_offsets, num_ads_in_batch):
        cat_ad_lengths = torch.cat(ad_lengths, 0)
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
        )

        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(cat_ad_lengths)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        )

        reordered_cat_ad_indices = torch.ops.fbgemm.cat_reorder_batched_ad_indices(
            cat_ad_offsets,
            ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
            broadcast_indices,
            batch_size * table_size * num_ads * length,
        )

        return reordered_cat_ad_indices.to(
            "cuda", non_blocking=True
        ), reordered_cat_ad_lengths.to("cuda", non_blocking=True)

    num_bytes = batch_size * table_size * (num_ads + 1) * length * data_size

    # pyre-ignore
    def ben(fn, name, ad_indices, ad_lengths, batch_offsets, num_ads_in_batch):
        time, _ = benchmark_torch_function(
            fn,
            (ad_indices, ad_lengths, batch_offsets, num_ads_in_batch),
            num_warmups=50,
            iters=500,
        )
        logging.info(
            f"{name} fbgemm_gpu time: {time * 1000:.5f} ms ({num_bytes / time / 1e9:.5f} GB/s)"
        )

    ben(pass_1, "pass_1", ad_indices, ad_lengths, batch_offsets, num_ads_in_batch)
    ben(pass_2, "pass_2", ad_indices, ad_lengths, batch_offsets, num_ads_in_batch)
    ben(pass_3, "pass_3", ad_indices, ad_lengths, batch_offsets, num_ads_in_batch)
    ben(pass_4, "pass_4", ad_indices, ad_lengths, batch_offsets, num_ads_in_batch)


@cli.command()
@click.option("--row-size", default=2560000)
@click.option("--batch-size", default=2048)
@click.option("--incidices-num", default=300000)
@click.option("--lengths-num", default=300000)
@click.option("--bucket-num", default=16)
@click.option("--input-precision", type=str, default="long")
@click.option("--sequence/--no-sequence", default=False)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
def block_bucketize_sparse_features_bench(
    row_size: int,
    batch_size: int,
    incidices_num: int,
    lengths_num: int,
    bucket_num: int,
    input_precision: str,
    sequence: bool,
    device: str,
) -> None:
    dtype = torch.int
    if input_precision == "int":
        dtype = torch.int
    elif input_precision == "long":
        dtype = torch.long
    else:
        raise RuntimeError(f"Does not support data type {input_precision}")

    lengths_num = lengths_num // batch_size * (batch_size)
    assert lengths_num <= incidices_num
    avg_len = incidices_num // lengths_num
    indices = torch.randint(0, row_size, (incidices_num,), dtype=dtype)
    weights = torch.randint(0, row_size, (incidices_num,), dtype=torch.float)
    lengths = [0] * lengths_num
    total = 0
    for i in range(lengths_num):
        length = int(random.gauss(mu=avg_len, sigma=1.0))
        lengths[i] = min(length, incidices_num - total)
        total += lengths[i]
        if total > incidices_num:
            break
    if total < incidices_num:
        lengths[-1] += incidices_num - total
    lengths = torch.tensor(lengths, dtype=dtype)
    bucket_size = math.ceil(row_size / bucket_num)
    block_sizes = torch.tensor([bucket_size] * batch_size, dtype=dtype)

    bucket_pos = [j * bucket_size for j in range(bucket_num + 1)]
    block_bucketize_pos = [torch.tensor(bucket_pos, device=device)] * batch_size
    test_param = {"uneven": block_bucketize_pos, "even": None}
    print(f"device {device}")
    for name, is_block_bucketize_pos in test_param.items():
        time, output = benchmark_torch_function(
            torch.ops.fbgemm.block_bucketize_sparse_features,
            (
                lengths if device == "cpu" else lengths.to(device),
                indices if device == "cpu" else indices.to(device),
                False,
                sequence,
                block_sizes if device == "cpu" else block_sizes.to(device),
                bucket_num,
                (
                    weights
                    if device == "cpu"
                    else (weights.to(device) if weights is not None else None)
                ),
                None,
                -1,  # unused
                (
                    is_block_bucketize_pos
                    if device == "cpu"
                    else (
                        [i.to(device) for i in is_block_bucketize_pos]
                        if is_block_bucketize_pos is not None
                        else None
                    )
                ),
            ),
            iters=100,
            device=device,
        )

        num_bytes = 0
        for tensor in [lengths, indices, weights, *block_bucketize_pos, *output]:
            if isinstance(tensor, torch.Tensor):
                num_bytes += (tensor.numel()) * tensor.element_size()

        logging.info(
            f"{name}_block_bucketize_sparse_features forward: {dtype}, {num_bytes} bytes read/write, {time * 1e3} ms, {num_bytes / time / 1e9} GB/s"
        )


if __name__ == "__main__":
    cli()
