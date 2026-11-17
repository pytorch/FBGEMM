# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Callable, List, Tuple

import click
import torch
from torch.profiler import profile, ProfilerActivity

logger: logging.Logger = logging.getLogger(__name__)

torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


TensorTuple = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _balanced_lengths(num_rows: int, num_values: int) -> List[int]:
    base, remainder = divmod(num_values, num_rows)
    return [base + int(row < remainder) for row in range(num_rows)]


def _make_inputs(
    batch_size: int,
    num_indices: int,
    target_indices: int,
    num_embeddings: int,
    total_num_blocks: int,
    world_size: int,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if target_indices >= num_indices:
        raise click.UsageError("--target-indices must be smaller than --num-indices")

    lengths = torch.tensor(
        _balanced_lengths(batch_size, num_indices - target_indices)
        + _balanced_lengths(batch_size, target_indices),
        dtype=torch.long,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(20260928)
    indices = torch.randint(
        0,
        num_embeddings,
        (num_indices,),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    raw_ids = torch.randint(
        0,
        2**61,
        (num_indices,),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    weights_2d = raw_ids.view(torch.float32).reshape(-1, 2)
    block_size = math.ceil(num_embeddings / total_num_blocks)
    block_sizes = torch.full((2,), block_size, dtype=torch.long, device=device)
    total_num_blocks_tensor = torch.full(
        (2,), total_num_blocks, dtype=torch.long, device=device
    )
    return lengths, indices, weights_2d, block_sizes, total_num_blocks_tensor


def _lookup(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    weights_2d: torch.Tensor,
    block_sizes: torch.Tensor,
    total_num_blocks: torch.Tensor,
    world_size: int,
) -> TensorTuple:
    return torch.ops.fbgemm.block_bucketize_sparse_features(
        lengths=lengths,
        indices=indices,
        bucketize_pos=False,
        sequence=True,
        block_sizes=block_sizes,
        my_size=world_size,
        weights=weights_2d,
        batch_size_per_feature=None,
        max_B=-1,
        block_bucketize_pos=None,
        keep_orig_idx=False,
        total_num_blocks=total_num_blocks,
        keep_orig_idx_per_feature=None,
    )


def _write(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    weights_2d: torch.Tensor,
    block_sizes: torch.Tensor,
    total_num_blocks: torch.Tensor,
    world_size: int,
) -> TensorTuple:
    return torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
        lengths=lengths,
        indices=indices,
        bucketize_pos=False,
        sequence=True,
        block_sizes=block_sizes,
        my_size=world_size,
        weights=weights_2d,
        weights_dim=2,
        batch_size_per_feature=None,
        max_B=-1,
        block_bucketize_pos=None,
        keep_orig_idx=False,
        total_num_blocks=total_num_blocks,
        keep_orig_idx_per_feature=None,
    )


def _reuse_lookup_routing(
    weights_2d: torch.Tensor,
    lookup_output: TensorTuple,
) -> TensorTuple:
    new_lengths, new_indices, _, new_pos, source_to_bucketized_permute = lookup_output
    bucketized_weights = torch.empty_like(weights_2d)
    bucketized_weights.index_copy_(
        0,
        source_to_bucketized_permute,
        weights_2d,
    )
    return (
        new_lengths,
        new_indices,
        bucketized_weights,
        new_pos,
        source_to_bucketized_permute,
    )


def _time_cuda(
    fn: Callable[[], TensorTuple], warmups: int, iterations: int
) -> Tuple[List[float], TensorTuple]:
    output = fn()
    for _ in range(warmups):
        output = fn()
    torch.cuda.synchronize()

    samples_ms: List[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))
    return samples_ms, output


def _verify_output(
    name: str,
    output: TensorTuple,
    indices: torch.Tensor,
    weights_2d: torch.Tensor,
    block_sizes: torch.Tensor,
    total_num_blocks: torch.Tensor,
    world_size: int,
    check_weights: bool,
) -> None:
    new_lengths, new_indices, new_weights, _, unbucketize_permute = output
    torch.testing.assert_close(
        new_lengths.sum(), torch.tensor(indices.numel(), device=indices.device)
    )
    if unbucketize_permute.numel() != indices.numel():
        raise AssertionError(f"{name}: wrong permutation size")

    local_num_blocks = total_num_blocks[0] // world_size
    expected_local_indices = torch.remainder(indices, block_sizes[0] * local_num_blocks)
    torch.testing.assert_close(new_indices[unbucketize_permute], expected_local_indices)
    sorted_permute = torch.sort(unbucketize_permute).values
    torch.testing.assert_close(
        sorted_permute,
        torch.arange(indices.numel(), dtype=torch.long, device=indices.device),
    )
    if check_weights:
        torch.testing.assert_close(
            new_weights[unbucketize_permute].view(torch.int32),
            weights_2d.view(torch.int32),
            rtol=0,
            atol=0,
        )


def _summary(samples_ms: List[float]) -> str:
    ordered = sorted(samples_ms)
    p10 = ordered[max(0, int(0.1 * (len(ordered) - 1)))]
    p90 = ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]
    return (
        f"median={statistics.median(samples_ms):.3f} ms "
        f"mean={statistics.mean(samples_ms):.3f} ms "
        f"p10={p10:.3f} ms p90={p90:.3f} ms"
    )


def _export_trace(
    lookup_fn: Callable[[], TensorTuple],
    write_fn: Callable[[], TensorTuple],
    reuse_fn: Callable[[], TensorTuple],
    trace_file: str,
) -> None:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            lookup_fn()
            write_fn()
            reuse_fn()
            prof.step()
    prof.export_chrome_trace(trace_file)


@click.command()
@click.option("--batch-size", default=384, show_default=True)
@click.option("--num-indices", default=2_807_700, show_default=True)
@click.option("--target-indices", default=1_619, show_default=True)
@click.option("--world-size", default=160, show_default=True)
@click.option("--num-embeddings", default=1_125_899_902_955_520, show_default=True)
@click.option("--total-num-blocks", default=3_991_680, show_default=True)
@click.option("--warmups", default=5, show_default=True)
@click.option("--iterations", default=20, show_default=True)
@click.option("--verify/--no-verify", default=True, show_default=True)
@click.option("--trace-file", default="", show_default=True)
def main(
    batch_size: int,
    num_indices: int,
    target_indices: int,
    world_size: int,
    num_embeddings: int,
    total_num_blocks: int,
    warmups: int,
    iterations: int,
    verify: bool,
    trace_file: str,
) -> None:
    """Benchmark the CU-embedding block-bucketize production shape."""
    if not torch.cuda.is_available():
        raise click.UsageError("A CUDA/HIP GPU is required")
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise click.UsageError("A CUDA/HIP GPU is required")
    device = torch.device(accelerator)
    lengths, indices, weights_2d, block_sizes, total_num_blocks_tensor = _make_inputs(
        batch_size,
        num_indices,
        target_indices,
        num_embeddings,
        total_num_blocks,
        world_size,
        device,
    )

    def lookup_fn() -> TensorTuple:
        return _lookup(
            lengths,
            indices,
            weights_2d,
            block_sizes,
            total_num_blocks_tensor,
            world_size,
        )

    def write_fn() -> TensorTuple:
        return _write(
            lengths,
            indices,
            weights_2d,
            block_sizes,
            total_num_blocks_tensor,
            world_size,
        )

    lookup_ms, lookup_output = _time_cuda(lookup_fn, warmups, iterations)

    def reuse_fn() -> TensorTuple:
        return _reuse_lookup_routing(weights_2d, lookup_output)

    write_ms, write_output = _time_cuda(write_fn, warmups, iterations)
    reuse_ms, reuse_output = _time_cuda(reuse_fn, warmups, iterations)
    if verify:
        # The legacy lookup operator accepts the Nx2 payload but historically
        # only scatters one scalar per ID. The dedicated write operator owns the
        # Nx2 payload contract and is checked end-to-end here.
        _verify_output(
            "lookup",
            lookup_output,
            indices,
            weights_2d,
            block_sizes,
            total_num_blocks_tensor,
            world_size,
            False,
        )
        _verify_output(
            "write",
            write_output,
            indices,
            weights_2d,
            block_sizes,
            total_num_blocks_tensor,
            world_size,
            True,
        )
        _verify_output(
            "reuse",
            reuse_output,
            indices,
            weights_2d,
            block_sizes,
            total_num_blocks_tensor,
            world_size,
            True,
        )

    result = {
        "pid": os.getpid(),
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "num_features": 2,
        "num_segments": batch_size * 2,
        "num_indices": num_indices,
        "target_indices": target_indices,
        "world_size": world_size,
        "num_embeddings": num_embeddings,
        "total_num_blocks": total_num_blocks,
        "lookup_ms": lookup_ms,
        "write_ms": write_ms,
        "reuse_ms": reuse_ms,
    }
    print(f"lookup: {_summary(lookup_ms)}")
    print(f"write:  {_summary(write_ms)}")
    print(f"reuse:  {_summary(reuse_ms)}")
    print(
        "lookup + independent write median: "
        f"{statistics.median(lookup_ms) + statistics.median(write_ms):.3f} ms"
    )
    print(
        "lookup + reused routing median: "
        f"{statistics.median(lookup_ms) + statistics.median(reuse_ms):.3f} ms"
    )
    print(json.dumps(result, sort_keys=True))

    if trace_file:
        trace_path = str(Path(trace_file).resolve())
        _export_trace(lookup_fn, write_fn, reuse_fn, trace_path)
        print(f"trace: {trace_path}")


if __name__ == "__main__":
    main()
