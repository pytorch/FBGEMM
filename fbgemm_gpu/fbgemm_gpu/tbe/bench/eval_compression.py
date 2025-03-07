# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
import statistics
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

logging.basicConfig(level=logging.DEBUG)


@dataclass
class EvalCompressionBenchmarkOutput:
    avg: float
    fwd: float
    bwd: float
    compressed_avg: float
    compressed_fwd: float
    reindex: float
    compressed_bwd: float


def benchmark_eval_compression(
    baseline_requests: List[Tuple[torch.Tensor, torch.Tensor]],
    compressed_requests: List[Tuple[torch.Tensor, torch.Tensor]],
    baseline_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    compressed_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reindex: torch.Tensor,
    embedding_dim: int,
) -> EvalCompressionBenchmarkOutput:
    times = []
    fwd_times = []
    bwd_times = []
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for indices, offsets in baseline_requests:
        time = 0.0
        start_event.record()
        # forward
        out = baseline_func(indices, offsets)
        end_event.record()
        torch.cuda.synchronize()
        it_time = start_event.elapsed_time(end_event) * 1.0e-3
        fwd_times.append(it_time)
        time += it_time

        grad = torch.rand_like(out)
        start_event.record()
        # backward
        out.backward(grad)
        end_event.record()
        torch.cuda.synchronize()
        it_time = start_event.elapsed_time(end_event) * 1.0e-3
        bwd_times.append(it_time)
        time += it_time
        times.append(time)

    avg = statistics.median(times)
    fwd = statistics.median(fwd_times)
    bwd = statistics.median(bwd_times)

    times.clear()
    fwd_times.clear()
    bwd_times.clear()
    reindex_times = []

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for indices, offsets in compressed_requests:
        time = 0.0
        start_event.record()
        # forward
        out = compressed_func(indices, offsets)
        end_event.record()
        torch.cuda.synchronize()
        it_time = start_event.elapsed_time(end_event) * 1.0e-3
        fwd_times.append(it_time)
        time += it_time

        start_event.record()
        # reindex
        out = out.reshape(-1, embedding_dim)
        out = torch.ops.fbgemm.index_select_dim0(out, reindex)
        end_event.record()
        torch.cuda.synchronize()
        it_time = start_event.elapsed_time(end_event) * 1.0e-3
        reindex_times.append(it_time)
        time += it_time

        grad = torch.rand_like(out)
        start_event.record()
        # backward
        out.backward(grad)
        end_event.record()
        torch.cuda.synchronize()
        it_time = start_event.elapsed_time(end_event) * 1.0e-3
        bwd_times.append(it_time)
        time += it_time
        times.append(time)

    compressed_avg = statistics.median(times)
    compressed_fwd = statistics.median(fwd_times)
    reindex = statistics.median(reindex_times)
    compressed_bwd = statistics.median(bwd_times)

    return EvalCompressionBenchmarkOutput(
        avg, fwd, bwd, compressed_avg, compressed_fwd, reindex, compressed_bwd
    )
