# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import random
from typing import List, Tuple

import click
import fbgemm_gpu
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

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--batch-size", type=int, default=128)
@click.option("--embedding-dim", type=int, default=128)
@click.option("--max-len", type=int, default=128)
@click.option("--elem-type", type=str, default="half")
def device(
    batch_size: int,
    embedding_dim: int,
    max_len: int,
    elem_type: str,
) -> None:
    lengths = torch.randint(max_len, size=(batch_size,))
    total_lengths = lengths.sum().item()
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

    dtype = (
        torch.float16
        if elem_type == "half" or elem_type == "float16"
        else torch.float32
    )

    # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool, float, int]`.
    values_2d = torch.rand(total_lengths, embedding_dim, dtype=dtype)

    if torch.cuda.is_available():
        offsets = offsets.cuda()
        values_2d = values_2d.cuda()

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_2d_to_dense, (values_2d, offsets, max_len), iters=1000
    )

    offsets_nbytes = offsets.numel() * offsets.element_size()
    values_nbytes = values_2d.numel() * values_2d.element_size()
    dense_nbytes = output.numel() * output.element_size()

    num_bytes = offsets_nbytes + values_nbytes + dense_nbytes
    logging.info(f"jagged_2d_to_dense {time} sec {num_bytes / time / 1e9} GB/s")

    total_L = values_2d.size(0)
    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.dense_to_jagged, (output, [offsets], total_L), iters=1000
    )

    num_bytes = offsets_nbytes + 2 * values_nbytes
    logging.info(f"dense_to_jagged (2d) {time} sec {num_bytes / time / 1e9} GB/s")

    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output,
        (values_2d, [offsets], output),
        iters=1000,
    )
    num_bytes = offsets_nbytes + 3 * values_nbytes
    logging.info(
        f"jagged_dense_elementwise_add_jagged_output {time} sec {num_bytes / time / 1e9} GB/s"
    )

    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_dense_elementwise_mul,
        (values_2d, [offsets], output),
        iters=1000,
    )
    num_bytes = offsets_nbytes + 3 * values_nbytes
    logging.info(
        f"jagged_dense_elementwise_mul {time} sec {num_bytes / time / 1e9} GB/s"
    )

    output_sq = output * output
    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output,
        (values_2d, [offsets], output, output_sq),
        iters=1000,
    )
    num_bytes = offsets_nbytes + 4 * values_nbytes
    logging.info(
        f"jagged_dense_dense_elementwise_add_jagged_output {time} sec {num_bytes / time / 1e9} GB/s"
    )

    # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
    #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
    values_1d = torch.rand(total_lengths)
    if torch.cuda.is_available():
        values_1d = values_1d.cuda()
    values_nbytes = values_1d.numel() * values_1d.element_size()

    time, output = benchmark_torch_function(
        lambda: torch.ops.fbgemm.jagged_1d_to_dense(
            values_1d, offsets, max_len, padding_value=0
        ),
        (),
        iters=1000,
    )
    dense_nbytes = output.numel() * output.element_size()

    num_bytes = offsets_nbytes + values_nbytes + dense_nbytes
    logging.info(f"jagged_1d_to_dense {time} sec {num_bytes / time / 1e9} GB/s")

    total_L = values_1d.size(0)
    output_1d = torch.unsqueeze(output, -1)
    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.dense_to_jagged, (output_1d, [offsets], total_L), iters=1000
    )

    num_bytes = offsets_nbytes + 2 * values_nbytes
    logging.info(f"dense_to_jagged (1d) {time} sec {num_bytes / time / 1e9} GB/s")


@cli.command()
@click.option("--batch-size", type=int, default=1)
@click.option("--h-dim", type=int, default=3)
@click.option("--embedding-dim", type=int, default=16)
@click.option("--max-len", type=int, default=10)
@click.option("--elem-type", type=str, default="half")
def batched_dense_vec_jagged_2d_mul(
    batch_size: int,
    h_dim: int,
    embedding_dim: int,
    max_len: int,
    elem_type: str,
) -> None:
    lengths = torch.randint(2 * max_len, size=(batch_size,))  # Allow for truncation
    total_lengths = lengths.sum().item()
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    dtype = (
        torch.float16
        if elem_type == "half" or elem_type == "float16"
        else torch.float32
    )
    # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool, float, int]`.
    values_2d = torch.rand(total_lengths, h_dim * embedding_dim, dtype=dtype)
    dense = torch.rand(batch_size * h_dim, max_len, dtype=dtype)
    if torch.cuda.is_available():
        offsets = offsets.cuda()
        values_2d = values_2d.cuda()
        dense = dense.cuda()

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
        (dense, values_2d, offsets),
        iters=1000,
    )

    # Account for the fact that each matmul inner dim was limited to max_len
    computed_lengths = torch.minimum(lengths, torch.ones(batch_size) * max_len)
    total_computed_lengths = computed_lengths.sum().item()
    num_flops = total_computed_lengths * h_dim * embedding_dim * 2.0
    logging.info(
        f"batched_dense_vec_jagged_2d_mul {time} sec {num_flops / time / 1e9} GFLOP/s"
    )


@cli.command()
@click.option("--batch-size", type=int, default=1024)
@click.option("--max-len", type=int, default=10)
@click.option("--dtype", type=str, default="float")
def jagged_1d_to_truncated_values(
    batch_size: int,
    max_len: int,
    dtype: str,
) -> None:
    lengths = torch.randint(2 * max_len, size=(batch_size,))  # Allow for truncation
    total_lengths = lengths.sum().item()
    torch_dtype = torch.float16 if dtype in ["half", "float16"] else torch.float32
    # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool, float, int]`.
    values = torch.rand(total_lengths, dtype=torch_dtype)

    def ref(values: torch.Tensor, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        dense_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)],
            [max_len],
            padding_value=0,
        )
        truncated_lengths = torch.clamp(lengths, max=max_len)
        mask2d = torch.arange(max_len).expand(
            batch_size, -1
        ) < truncated_lengths.unsqueeze(-1)
        return dense_values[mask2d].view(-1)

    time_ref, output_ref = benchmark_torch_function(
        ref,
        (values, lengths, max_len),
    )

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_1d_to_truncated_values,
        (values, lengths, max_len),
    )

    torch.testing.assert_close(output, output_ref)

    bytes = (values.numel() + output.numel()) * (
        4 if torch_dtype == torch.float else 2
    ) + lengths.numel() * 4

    logging.info(f"reference {time_ref} sec {bytes / time_ref / 1e9} GB/s")
    logging.info(f"truncate_jagged_1d {time} sec {bytes / time / 1e9} GB/s")


@cli.command()
@click.option("--batch-size", type=int, default=1024)
@click.option("--max-len", type=int, default=256)
def masked_select_jagged_1d(
    batch_size: int,
    max_len: int,
) -> None:
    lengths = torch.randint(2 * max_len, size=(batch_size,))  # Allow for truncation
    total_lengths = int(lengths.sum().item())
    dtype = torch.long
    values = torch.randint(2**16, (total_lengths,), dtype=dtype)
    mask = torch.randint(2, (total_lengths,)) > 0

    def ref(
        values: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_values_ref = values[mask]
        cum_count = torch.cumsum(mask, 0)
        cum_count = torch.cat((cum_count, torch.tensor([0])))
        cum_length = cum_count[torch.cumsum(lengths, 0) - 1]
        cum_length_shift_right = torch.roll(cum_length, 1)
        cum_length_shift_right[0] = 0
        masked_lengths_ref = cum_length - cum_length_shift_right
        return masked_values_ref, masked_lengths_ref

    time_ref, (masked_values_ref, masked_lengths_ref) = benchmark_torch_function(
        ref,
        (values, lengths, mask),
    )

    time, (masked_values, masked_lengths) = benchmark_torch_function(
        torch.ops.fbgemm.masked_select_jagged_1d,
        (values, lengths, mask),
    )

    torch.testing.assert_close(masked_values, masked_values_ref)
    torch.testing.assert_close(masked_lengths, masked_lengths_ref)

    bytes = (2 * values.numel() + 2 * lengths.numel() + 2 * masked_values.numel()) * 4

    logging.info(f"reference {time_ref} sec {bytes / time_ref / 1e9} GB/s")
    logging.info(f"masked_select_jagged_1d {time} sec {bytes / time / 1e9} GB/s")


@cli.command()
@click.option("--num-batches", type=int, default=40)
@click.option("--max-seq-length", type=int, default=400)
@click.option("--input-batch-size", type=int, default=1024)
@click.option("--output-batch-size", type=int, default=512)
@click.option("--jagged-tensor-type", type=str, default="float")
@click.option("--has-weights", is_flag=True, default=False)
@click.option("--weight-type", type=str, default="float")
def keyed_jagged_index_select_dim1(
    num_batches: int,
    max_seq_length: int,
    input_batch_size: int,
    output_batch_size: int,
    jagged_tensor_type: str,
    has_weights: bool,
    weight_type: str,
) -> None:
    jagged_tensor_types = {
        "float": torch.float,
        "half": torch.half,
        "int": torch.int,
        "long": torch.long,
    }
    weight_types = {"float": torch.float, "half": torch.half}

    if jagged_tensor_type not in jagged_tensor_types.keys():
        raise AssertionError(
            f"--jagged-tensor-type ({jagged_tensor_type}) is not supported"
        )
    if weight_type not in weight_types.keys():
        raise AssertionError(f"--weight-type ({weight_type}) is not supported")

    jagged_tensor_dtype = jagged_tensor_types[jagged_tensor_type]
    is_float = jagged_tensor_dtype in [torch.float, torch.half]
    weight_dtype = weight_types[weight_type]

    lengths = torch.randint(
        low=0,
        high=max_seq_length,
        size=(input_batch_size * num_batches,),
        dtype=torch.long,
        device="cuda",
    )
    # Imitate KeyedJaggedTensor offsets
    offsets = torch.concat(
        [torch.zeros(1, dtype=torch.long, device="cuda"), lengths.cumsum(0)]
    )
    indices = torch.randint(
        low=0,
        high=1,
        size=(output_batch_size,),
        dtype=torch.long,
        device="cuda",
    )
    if is_float:
        values = torch.rand(
            int(offsets[-1].item()),
            dtype=jagged_tensor_dtype,
            device="cuda",
        )
    else:
        values = torch.randint(
            2**16,
            (int(offsets[-1].item()),),
            dtype=jagged_tensor_dtype,
            device="cuda",
        )
    weights = (
        torch.rand(int(offsets[-1].item()), dtype=weight_dtype, device="cuda")
        if has_weights
        else None
    )

    # Only float tensors can require grad
    if is_float:
        values.requires_grad = True

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.keyed_jagged_index_select_dim1,
        (values, lengths, offsets, indices, input_batch_size, weights),
        iters=1000,
    )
    output = output[0]

    # Prepare inputs for the reference run
    ref_inputs = []
    for k in range(num_batches):
        key_lengths = lengths[k * input_batch_size : (k + 1) * input_batch_size]
        start_offset = offsets[k * input_batch_size]
        end_offset = offsets[(k + 1) * input_batch_size]
        key_values = values[start_offset:end_offset].view(-1, 1)
        if has_weights:
            # pyre-ignore[16]
            key_weights = weights[start_offset:end_offset].view(-1, 1)
        else:
            key_weights = torch.empty(0)
        ref_inputs.append((key_values, key_lengths, indices, key_weights))

    def keyed_jagged_index_select_dim1_ref(
        inputs: List[torch.Tensor],
        has_weights: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        output_weights = []
        for key_values, key_lengths, indices, _ in inputs:
            outputs.append(
                torch.ops.fbgemm.jagged_index_select(key_values, key_lengths, indices)[
                    0
                ].view(-1)
            )
        if has_weights:
            for _, key_lengths, indices, key_weights in inputs:
                output_weights.append(
                    torch.ops.fbgemm.jagged_index_select(
                        key_weights, key_lengths, indices
                    )[0].view(-1)
                )
        return torch.concat(outputs), torch.concat(
            output_weights
        ) if has_weights else torch.empty(0)

    time_ref, output_ref = benchmark_torch_function(
        keyed_jagged_index_select_dim1_ref, (ref_inputs, has_weights)
    )
    output_ref = output_ref[0]

    logging.info(
        f"keyed_jagged_index_select_dim1 forward time: {time * 1e3} ms, ref {time_ref * 1e3}"
    )

    if not is_float:
        return

    grad = torch.rand_like(output)
    time, _ = benchmark_torch_function(
        functools.partial(output.backward, retain_graph=True), (grad,), iters=1000
    )
    time_ref, _ = benchmark_torch_function(
        functools.partial(output_ref.backward, retain_graph=True), (grad,), iters=1000
    )
    logging.info(
        f"keyed_jagged_index_select_dim1 backward time: {time * 1e3} ms, ref {time_ref * 1e3}"
    )


@cli.command()
@click.option("--max-seq-length", type=int, default=400)
@click.option("--input-batch-size", type=int, default=1024)
@click.option("--slice-length", type=int, default=10)
@click.option("--jagged-tensor-type", type=str, default="float")
def jagged_slice_cpu(
    max_seq_length: int,
    input_batch_size: int,
    slice_length: int,
    jagged_tensor_type: str,
) -> None:
    jagged_tensor_types = {
        "float": torch.float,
        "half": torch.half,
        "int": torch.int,
        "long": torch.long,
    }

    if jagged_tensor_type not in jagged_tensor_types.keys():
        raise AssertionError(
            f"--jagged-tensor-type ({jagged_tensor_type}) is not supported"
        )

    jagged_tensor_dtype = jagged_tensor_types[jagged_tensor_type]
    is_float = jagged_tensor_dtype in [torch.float, torch.half]

    lengths = torch.randint(
        low=0,
        high=max_seq_length,
        size=(input_batch_size,),
        dtype=torch.long,
    )
    start_list = [random.randint(0, max(len_ - 1, 0)) for len_ in lengths.tolist()]
    start = torch.tensor(start_list)

    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    if is_float:
        values = torch.rand(
            int(offsets[-1].item()),
            dtype=jagged_tensor_dtype,
        )
    else:
        values = torch.randint(
            2**16,
            (int(offsets[-1].item()),),
            dtype=jagged_tensor_dtype,
        )

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_slice,
        (values, lengths, start, slice_length),
        iters=1000,
    )

    def jagged_slice_ref(
        x_values: torch.Tensor,
        offsets: torch.Tensor,
        start: torch.Tensor,
        max_L: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        end_offsets_ = max_L + start + offsets[:-1]
        end_offsets = torch.where(end_offsets_ > offsets[1:], offsets[1:], end_offsets_)
        start_offsets = start + offsets[:-1]
        indices_to_select: List[torch.Tensor] = []
        for i in range(end_offsets.size(0)):
            indices_to_select.append(
                torch.arange(start_offsets[i].item(), end_offsets[i].item())
            )
        output_ref = torch.index_select(x_values, 0, torch.cat(indices_to_select))
        new_lengths = end_offsets - start_offsets
        new_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(new_lengths)
        return output_ref, new_offsets

    time_ref, output = benchmark_torch_function(
        jagged_slice_ref, (values, offsets, start, slice_length)
    )

    logging.info(f"jagged_slice forward time: {time * 1e3} ms, ref {time_ref * 1e3} ms")

    profiler = profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=200,
            warmup=100,
            active=100,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )

    profiler.start()
    for _ in range(500):
        torch.ops.fbgemm.jagged_slice(values, lengths, start, slice_length)
        profiler.step()
    profiler.stop()

    logging.info(
        "\n"
        + profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    )

    flops = sum(e.flops for e in profiler.events())
    logging.info(f"Total Compute: {flops / 1e9} gflops")


if __name__ == "__main__":
    cli()
