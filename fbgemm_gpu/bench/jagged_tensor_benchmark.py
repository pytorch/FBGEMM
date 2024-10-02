# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import functools
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple

import click
import fbgemm_gpu
import torch
from torch.profiler import profile

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_cpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_multi_embedding_ops_cpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_multi_embedding_ops_gpu"
    )


@click.group()
def cli() -> None:
    pass


@dataclass
class JaggedTensor:
    """
    A simple wrapper class around jagged tensors for benchmarking purposes.
    Jagged tensors are a tensor of variable length vectors.  They are
    represented as a tuple of (values, lengths, offsets) where values is a 2D
    tensor of shape (total_lengths, embedding_dim) and lengths is a 1D tensor
    of shape (batch_size,) containing the length of each row in the batch.
    Offsets is a 1D tensor of shape (batch_size + 1,) containing the offset of
    each row.
    """

    values: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor
    batch_size: int
    embedding_dim: int
    max_len: int

    @property
    def total_lengths(self) -> int:
        return int(self.lengths.sum().item())

    @staticmethod
    def rand_2d(
        batch_size: int, embedding_dim: int, max_len: int, elem_type: str
    ) -> JaggedTensor:
        """
        Generate a random JaggedTensor with 2D values.
        """
        # Each row in the batch has different length
        lengths = torch.randint(max_len, size=(batch_size,))
        total_lengths = lengths.sum().item()

        # Compute the offsets
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        # Set dtype
        dtype = (
            torch.float16
            if elem_type == "half" or elem_type == "float16"
            else torch.float32
        )

        # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool, float, int]`.
        values_2d = torch.rand(total_lengths, embedding_dim, dtype=dtype)

        if torch.cuda.is_available():
            values_2d = values_2d.cuda()
            offsets = offsets.cuda()

        return JaggedTensor(
            values_2d, lengths, offsets, batch_size, embedding_dim, max_len
        )

    def to_dense(self) -> torch.Tensor:
        """
        Convert the JaggedTensor into a dense tensor.
        """
        if self.values.dim() == 2:
            return torch.ops.fbgemm.jagged_2d_to_dense(
                self.values, self.offsets, self.max_len
            )
        elif self.values.dim() == 1:
            return torch.ops.fbgemm.jagged_1d_to_dense(
                self.values, self.offsets, self.max_len, padding_value=0
            )
        else:
            raise RuntimeError(f"Unsupported JaggedTensor dim {self.values.dim()}")

    def as_nested(self) -> torch.Tensor:
        """
        Convert the JaggedTensor into a PyTorch NestedTensor.
        """
        tensors = []

        for i in range(1, len(self.offsets)):
            tensors.append(self.values[self.offsets[i - 1] : self.offsets[i],])

        return torch.nested.nested_tensor(tensors)

    def nbytes(self) -> int:
        """
        Return the number of bytes used by the JaggedTensor.
        """
        offsets_nbytes = self.offsets.numel() * self.offsets.element_size()
        values_nbytes = self.values.numel() * self.values.element_size()
        return offsets_nbytes + values_nbytes


def dense_to_nested(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert a dense tensor into a PyTorch NestedTensor.
    """
    return torch.nested.nested_tensor(
        [values[i][: lengths[i],] for i in range(len(lengths))]
    )


def bench_jagged_2d_to_dense(jten: JaggedTensor) -> None:
    logging.info("######## Jagged (2D) to Dense ########")

    time, output = benchmark_torch_function(
        jten.to_dense,
        (),
        iters=1000,
    )

    dense_nbytes = output.numel() * output.element_size()
    num_bytes = jten.nbytes() + dense_nbytes
    logging.info(f"FBGEMM JaggedTensor: {time} sec {num_bytes / time / 1e9} GB/s")

    nten = jten.as_nested()
    time, output = benchmark_torch_function(
        torch.nested.to_padded_tensor,
        (nten, 0.0, (jten.batch_size, jten.max_len, jten.embedding_dim)),
        iters=1000,
    )

    nten_bytes = nten.numel() * nten.element_size()
    dense_nbytes = output.numel() * output.element_size()
    num_bytes = nten_bytes + dense_nbytes
    logging.info(f"PyTorch NestedTensor: {time} sec {num_bytes / time / 1e9} GB/s")
    logging.info("")


def bench_dense_to_jagged_2d(jten: JaggedTensor) -> None:
    logging.info("######## Dense to Jagged (2D) ########")

    dense_values = jten.to_dense()

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.dense_to_jagged,
        (dense_values, [jten.offsets], jten.total_lengths),
        iters=1000,
    )

    dense_nbytes = dense_values.numel() * dense_values.element_size()
    output_nbytes = output[0].numel() * output[0].element_size()
    offsets_nbytes = jten.offsets.numel() * jten.offsets.element_size()
    num_bytes = dense_nbytes + output_nbytes + offsets_nbytes
    logging.info(f"FBGEMM JaggedTensor: {time} sec {num_bytes / time / 1e9} GB/s")

    time, output = benchmark_torch_function(
        dense_to_nested,
        (dense_values, jten.lengths),
        iters=1000,
    )

    output_nbytes = output.numel() * output.element_size()
    num_bytes = dense_nbytes + output_nbytes
    logging.info(f"PyTorch NestedTensor: {time} sec {num_bytes / time / 1e9} GB/s")
    logging.info("")


def bench_jagged_dense_elementwise_op_jagged_output(jten: JaggedTensor) -> None:
    logging.info("######## Jagged (x) Dense -> Jagged ########")

    def nested_tensor_add(
        jagged_x: JaggedTensor, nested_x: torch.Tensor, dense_y: torch.Tensor
    ) -> torch.Tensor:
        return nested_x + dense_to_nested(
            dense_y,
            jagged_x.lengths,
        )

    def nested_tensor_mul(
        jagged_x: JaggedTensor, nested_x: torch.Tensor, dense_y: torch.Tensor
    ) -> torch.Tensor:
        return nested_x * dense_to_nested(
            dense_y,
            jagged_x.lengths,
        )

    offsets_nbytes = jten.offsets.numel() * jten.offsets.element_size()
    values_nbytes = jten.values.numel() * jten.values.element_size()
    num_bytes = offsets_nbytes + 3 * values_nbytes

    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output,
        (jten.values, [jten.offsets], jten.to_dense()),
        iters=1000,
    )
    logging.info(f"(Add) FBGEMM JaggedTensor: {time} sec {num_bytes / time / 1e9} GB/s")

    time, nested_output = benchmark_torch_function(
        nested_tensor_add,
        (jten, jten.as_nested(), jten.to_dense()),
        iters=1000,
    )
    logging.info(
        f"(Add) PyTorch NestedTensor: {time} sec {num_bytes / time / 1e9} GB/s"
    )

    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_dense_elementwise_mul,
        (jten.values, [jten.offsets], jten.to_dense()),
        iters=1000,
    )
    logging.info(f"(Mul) FBGEMM JaggedTensor: {time} sec {num_bytes / time / 1e9} GB/s")

    time, nested_output = benchmark_torch_function(
        nested_tensor_mul,
        (jten, jten.as_nested(), jten.to_dense()),
        iters=1000,
    )
    logging.info(
        f"(Mul) PyTorch NestedTensor: {time} sec {num_bytes / time / 1e9} GB/s"
    )
    logging.info("")


def bench_jagged_dense_dense_elementwise_add_jagged_output(jten: JaggedTensor) -> None:
    logging.info("######## Jagged + Dense + Dense -> Jagged ########")

    def nested_tensor_add(
        jagged_x: JaggedTensor,
        nested_x: torch.Tensor,
        dense_y0: torch.Tensor,
        dense_y1: torch.Tensor,
    ) -> torch.Tensor:
        return (
            nested_x
            + dense_to_nested(
                dense_y0,
                jagged_x.lengths,
            )
            + dense_to_nested(
                dense_y1,
                jagged_x.lengths,
            )
        )

    offsets_nbytes = jten.offsets.numel() * jten.offsets.element_size()
    values_nbytes = jten.values.numel() * jten.values.element_size()
    num_bytes = offsets_nbytes + 4 * values_nbytes

    output = jten.to_dense()
    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output,
        (jten.values, [jten.offsets], output, output * output),
        iters=1000,
    )
    logging.info(f"FBGEMM JaggedTensor: {time} sec {num_bytes / time / 1e9} GB/s")

    time, nested_output = benchmark_torch_function(
        nested_tensor_add,
        (jten, jten.as_nested(), output, output * output),
        iters=1000,
    )
    logging.info(f"PyTorch NestedTensor: {time} sec {num_bytes / time / 1e9} GB/s")
    logging.info("")


def bench_jagged_1d_to_dense(jten: JaggedTensor) -> None:
    logging.info("######## Jagged (1D) to Dense ########")

    # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
    #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
    jten.values = torch.rand(jten.total_lengths)
    if torch.cuda.is_available():
        jten.values = jten.values.cuda()

    time, output = benchmark_torch_function(
        jten.to_dense,
        (),
        iters=1000,
    )

    dense_nbytes = output.numel() * output.element_size()
    num_bytes = jten.nbytes() + dense_nbytes
    logging.info(f"FBGEMM JaggedTensor: {time} sec {num_bytes / time / 1e9} GB/s")

    nten = jten.as_nested()
    time, output = benchmark_torch_function(
        torch.nested.to_padded_tensor,
        (nten, 0.0, (jten.batch_size, jten.embedding_dim)),
        iters=1000,
    )

    nten_bytes = nten.numel() * nten.element_size()
    num_bytes = nten_bytes + dense_nbytes
    logging.info(f"PyTorch NestedTensor: {time} sec {num_bytes / time / 1e9} GB/s")
    logging.info("")


def bench_dense_to_jagged_1d(jten: JaggedTensor) -> None:
    logging.info("######## Dense to Jagged (1D) ########")

    # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
    #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
    jten.values = torch.rand(jten.total_lengths)
    if torch.cuda.is_available():
        jten.values = jten.values.cuda()
    dense_values = jten.to_dense()

    dense_1d = torch.unsqueeze(dense_values, -1)
    time, jagged_output = benchmark_torch_function(
        torch.ops.fbgemm.dense_to_jagged,
        (dense_1d, [jten.offsets], jten.total_lengths),
        iters=1000,
    )

    dense_1d_nbytes = dense_1d.numel() * dense_1d.element_size()
    offsets_nbytes = jten.offsets.numel() * jten.offsets.element_size()
    jagged_output_bytes = jagged_output[0].numel() * jagged_output[0].element_size()
    num_bytes = offsets_nbytes + dense_1d_nbytes + jagged_output_bytes
    logging.info(f"FBGEMM JaggedTensor: {time} sec {num_bytes / time / 1e9} GB/s")

    time, output = benchmark_torch_function(
        dense_to_nested,
        (dense_1d, jten.lengths),
        iters=1000,
    )

    nten_nbytes = output.numel() * output.element_size()
    num_bytes = dense_1d_nbytes + nten_nbytes
    logging.info(f"PyTorch NestedTensor: {time} sec {num_bytes / time / 1e9} GB/s")
    logging.info("")


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
    jtensor = JaggedTensor.rand_2d(batch_size, embedding_dim, max_len, elem_type)

    bench_jagged_2d_to_dense(jtensor)

    bench_dense_to_jagged_2d(jtensor)

    bench_jagged_dense_elementwise_op_jagged_output(jtensor)

    bench_jagged_dense_dense_elementwise_add_jagged_output(jtensor)

    bench_jagged_1d_to_dense(jtensor)

    bench_dense_to_jagged_1d(jtensor)


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
@click.option("--use-selected-lengths-sum", is_flag=True, default=False)
def keyed_jagged_index_select_dim1(
    num_batches: int,
    max_seq_length: int,
    input_batch_size: int,
    output_batch_size: int,
    jagged_tensor_type: str,
    has_weights: bool,
    weight_type: str,
    use_selected_lengths_sum: bool,
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

    if use_selected_lengths_sum:
        length_indices = torch.cat(
            [indices + i * input_batch_size for i in range(num_batches)]
        )
        selected_lengths_sum = (
            torch.index_select(lengths, 0, length_indices).sum().item()
        )
    else:
        selected_lengths_sum = None

    # Only float tensors can require grad
    if is_float:
        values.requires_grad = True

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.keyed_jagged_index_select_dim1,
        (
            values,
            lengths,
            offsets,
            indices,
            input_batch_size,
            weights,
            selected_lengths_sum,
        ),
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
        return torch.concat(outputs), (
            torch.concat(output_weights) if has_weights else torch.empty(0)
        )

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
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            torch.profiler.ProfilerActivity.CPU,
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
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


@cli.command()
@click.option("--batch-size", type=int, default=1024)
@click.option("--num-of-features", type=int, default=256)
@click.option("--num-of-tensors", type=int, default=4)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cuda")
def permute_pooled_embs_bench(
    num_of_features: int,
    num_of_tensors: int,
    batch_size: int,
    device: str,
) -> None:
    in_lengths = []
    out_lengths = []
    permute_list = []
    i = 0
    for in_tensor in range(num_of_tensors):
        lengths = []
        for _ in range(num_of_features):
            lengths.append(2 << random.randint(3, 10))
            permute_list.append([in_tensor, i, sum(lengths[:-1]), 0, lengths[-1], 0])
            i += 1
        in_lengths.append(lengths)
    offsets = [0]
    for permute in permute_list:
        offsets.append(offsets[-1] + permute[4])
    random.shuffle(permute_list)
    inv_offsets = [0]
    permutes = []
    for i, permute in enumerate(permute_list):
        permutes.append(permute[1])
        inv_offsets.append(inv_offsets[-1] + permute[4])
        permute[1] = i // num_of_features
        if i % num_of_features == 0:
            out_lengths.append([])
        permute[3] = sum(out_lengths[-1])
        out_lengths[-1].append(permute[4])
    inv_permutes = [0] * len(permutes)
    for i, p in enumerate(permutes):
        inv_permutes[p] = i

    in_lengths = [sum(len) for len in in_lengths]
    out_lengths = [sum(len) for len in out_lengths]
    in_shape = torch.tensor(in_lengths, dtype=torch.int32, device=torch.device(device))
    out_shape = torch.tensor(
        out_lengths, dtype=torch.int32, device=torch.device(device)
    )
    permutes_tensor = torch.tensor(
        permute_list, dtype=torch.int32, device=torch.device(device)
    )

    values = torch.rand((batch_size, offsets[-1]), device=torch.device(device))
    offsets = torch.tensor(offsets, device=torch.device(device))
    permutes = torch.tensor(permutes, device=torch.device(device))
    inv_offsets = torch.tensor(inv_offsets, device=torch.device(device))
    inv_permutes = torch.tensor(inv_permutes, device=torch.device(device))

    m_values = [
        torch.empty([batch_size, length], device=torch.device(device))
        for length in in_lengths
    ]
    for i, v in enumerate(torch.split(values, in_lengths, dim=1)):
        m_values[i].copy_(v)
    permute_list = [i for p in permute_list for i in p]

    time_ref, output_ref = benchmark_torch_function(
        torch.ops.fbgemm.permute_pooled_embs_auto_grad,
        (
            values,
            offsets,
            permutes,
            inv_offsets,
            inv_permutes,
        ),
        num_warmups=20,
        iters=100,
    )

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.permute_multi_embedding,
        (
            m_values,
            permutes_tensor,
            in_shape,
            out_shape,
            out_lengths,
        ),
        num_warmups=20,
        iters=100,
    )

    logging.info(
        f"size: {batch_size} x {offsets[-1]}; "
        "permute_multi_embedding: %.3g ms; permute_pooled_embs: %.3g ms; delta: %.1f%%"
        % (time * 1e3, time_ref * 1e3, (time - time_ref) / time_ref * 100),
    )

    for i, out in enumerate(output_ref.split(out_lengths, dim=1)):
        assert torch.allclose(out, output[i])


if __name__ == "__main__":
    cli()
