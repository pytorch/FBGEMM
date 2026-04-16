# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import os
from contextlib import nullcontext
from math import sqrt

import click
import fbgemm_gpu
import fbgemm_gpu.batched_unary_embeddings_ops as batched_unary_embeddings_ops
import numpy as np
import torch
from torch.profiler import profile

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


def generate_unary_feature(
    batch_size: int,
    num_embeddings: int,
    # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
    #  `typing.List[<element type>]` to avoid runtime subscripting errors.
) -> tuple[list, list, list]:
    lengths = []
    offsets = []
    indices = []
    offset = 0
    for _ in range(batch_size):
        n_indices = 1
        indices += (
            np.round(np.random.random(n_indices) * (num_embeddings - 1))
            .astype(int)
            .tolist()
        )
        offsets.append(offset)
        offset += 1
        lengths.append(n_indices)
    offsets.append(offset)
    return (lengths, offsets, indices)


class MyModule(torch.nn.Module):
    def __init__(self, num_tasks: int, hash_sizes: list[int]) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.hash_sizes = hash_sizes
        self.emb_modules = torch.nn.ModuleList()
        for _ in range(num_tasks):
            for h in self.hash_sizes:
                emb = torch.nn.EmbeddingBag(
                    num_embeddings=h,
                    embedding_dim=1,
                    mode="sum",
                    sparse=False,
                    include_last_offset=True,
                )
                emb.weight = torch.nn.Parameter(
                    torch.empty([h, 1]).uniform_(-sqrt(1 / h), sqrt(1 / h))
                )
                self.emb_modules.append(emb)

    def forward(
        self, offsets: list[torch.Tensor], indices: list[torch.Tensor]
    ) -> torch.Tensor:
        tt_list = []
        for n in range(self.num_tasks):
            t_list = []
            for i in range(len(self.hash_sizes)):
                t = self.emb_modules[n * len(self.hash_sizes) + i](
                    offsets=offsets[i].long(), input=indices[i].long()
                )
                t_list.append(t)
            tt = torch.cat(t_list, dim=1)
            tt_list.append(tt)
        return torch.cat(tt_list).view(self.num_tasks, -1, len(self.hash_sizes))


@click.command()
@click.option("--batch-size", default=512)
@click.option("--num-tables", default=2)
@click.option("--num-tasks", default=3)
@click.option("--repeats", default=100)
@click.option(
    "--manual-seed/--skip-manual-seed",
    default=False,
    help="Use manual seed for reproduction.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run on (cuda or cpu). Default is cuda.",
)
@click.option(
    "--export-trace",
    is_flag=True,
    default=False,
    help="Enable export of trace for profiling. Default is False.",
)
@click.option(
    "--trace-url",
    type=str,
    default="batched_unary_embeddings_{phase}_trace_{ospid}.json",
)
def cli(
    batch_size: int,
    num_tables: int,
    num_tasks: int,
    repeats: int,
    manual_seed: bool,
    device: str,
    export_trace: bool,
    trace_url: str,
) -> None:
    # set manual seed for reproducibility
    if manual_seed:
        torch.manual_seed(42)
        np.random.seed(42)

    dev = torch.device(device, 0) if device == "cuda" else torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    hash_sizes = list(np.random.choice(range(50, 250), size=(num_tables)))
    lengths = []
    offsets = []
    indices = []
    for h in hash_sizes:
        l, o, i = generate_unary_feature(batch_size, h)
        lengths.append(torch.tensor(l, dtype=torch.int64, device=dev))
        offsets.append(torch.tensor(o, dtype=torch.int64, device=dev))
        indices.append(torch.tensor(i, dtype=torch.int64, device=dev))
    lengths_tensor = torch.cat(lengths)
    indices_tensor = torch.cat(indices)
    offsets_tensor = torch.zeros(
        lengths_tensor.numel() + 1,
        dtype=lengths_tensor.dtype,
        device=lengths_tensor.device,
    )
    offsets_tensor[1:] = torch.ops.fbgemm.asynchronous_inclusive_cumsum(
        lengths_tensor.view(-1)
    )

    # forward
    ref_emb = MyModule(num_tasks, hash_sizes).to(dev)
    unary_emb = batched_unary_embeddings_ops.BatchedUnaryEmbeddingBag(
        num_tasks, hash_sizes, long_index=True
    ).to(dev)
    for i, param in enumerate(unary_emb.split_embedding_weights()):
        param.detach().copy_(ref_emb.emb_modules[i].weight)
    output_ref = ref_emb(offsets, indices)
    output = unary_emb(offsets_tensor, indices_tensor)
    torch.testing.assert_close(output_ref, output)
    # backward
    d_output = torch.randn([num_tasks, batch_size, len(hash_sizes)]).to(dev) * 0.1
    output_ref.backward(d_output)
    output.backward(d_output)
    d_weight_ref = []
    for emb in ref_emb.emb_modules:
        d_weight_ref.append(emb.weight.grad)
    d_weight_ref = torch.cat(d_weight_ref).view(num_tasks, -1)
    d_weight = unary_emb.weight.grad
    # pyre-fixme[16]: Optional type has no attribute `squeeze`.
    torch.testing.assert_close(d_weight_ref, d_weight.squeeze())

    # Kineto trace helpers
    def _kineto_trace_handler(p: profile, phase: str) -> None:
        p.export_chrome_trace(trace_url.format(phase=phase, ospid=os.getpid()))

    # pyre-ignore[3, 2]
    def context_factory(on_trace_ready):
        return profile(on_trace_ready=on_trace_ready) if export_trace else nullcontext()

    # A100 40MB L2 cache
    elapse, _ = benchmark_torch_function(ref_emb, (offsets, indices), iters=repeats)
    print("PyTorch EmbeddingBag forward", elapse)

    elapse, _ = benchmark_torch_function(
        unary_emb,
        (offsets_tensor, indices_tensor),
        iters=repeats,
    )
    print("Batched Unary Emb forward", elapse)

    output = ref_emb(offsets, indices)
    output.backward(d_output, retain_graph=True)
    elapse, _ = benchmark_torch_function(
        functools.partial(output.backward, retain_graph=True),
        (d_output,),
        iters=repeats,
    )
    print("PyTorch EmbeddingBag backward", elapse)

    output = unary_emb(offsets_tensor, indices_tensor)
    with context_factory(lambda p: _kineto_trace_handler(p, "fwd")):
        elapse, _ = benchmark_torch_function(
            unary_emb,
            (offsets_tensor, indices_tensor),
            iters=repeats,
        )
    print("Batched Unary Emb forward (traced)", elapse)

    output = unary_emb(offsets_tensor, indices_tensor)
    with context_factory(lambda p: _kineto_trace_handler(p, "bwd")):
        elapse, _ = benchmark_torch_function(
            functools.partial(output.backward, retain_graph=True),
            (d_output,),
            iters=repeats,
        )
    print("Batched Unary Emb backward (traced)", elapse)


if __name__ == "__main__":
    cli()
