# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.utils import (  # noqa: F401
    b_indices,
    generate_requests,  # noqa: F401
    get_device,  # noqa: F401
    round_up,  # noqa: F401
    TBERequest,
)
from torch import nn

logging.basicConfig(level=logging.DEBUG)


def benchmark_torch_function(  # noqa: C901
    # pyre-fixme[2]: Parameter must be annotated.
    f,
    # pyre-fixme[2]: Parameter must be annotated.
    args,
    # pyre-fixme[2]: Parameter must be annotated.
    kwargs={},
    flush_gpu_cache_size_mb: int = 40,
    iters: int = 10,
    num_warmups: int = 2,
    device: str = "cuda",
    name: str = "",
    num_threads: int = 1,
    copy_f_for_multi_thread_test: bool = False,
) -> Tuple[float, torch.Tensor]:
    logging.debug(f"Start to benchmark {name}...")
    if device != "cpu" and device != "" and device != "cuda":
        torch.cuda.set_device(device)
    for _ in range(num_warmups):
        output = f(*args, **kwargs)

    assert num_threads > 0
    if device != "cpu" and torch.cuda.is_available() and (num_threads == 1):
        cache = torch.empty(
            int(flush_gpu_cache_size_mb * 1024 * 1024 // 4),
            dtype=torch.float,
            device=device,
        )
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(iters)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(iters)]
        torch.cuda.synchronize(device)
        for i in range(iters):
            # flush the cache
            if flush_gpu_cache_size_mb:
                cache.zero_()
            start_event[i].record()
            with torch.cuda.nvtx.range(f"RunCudaModule_{name}"):
                output = f(*args)
            end_event[i].record()
        torch.cuda.synchronize(device)
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
        )
        elapsed_time = torch.mean(times).item() * 1.0e-3
    elif device != "cpu" and torch.cuda.is_available() and (num_threads > 1):
        cache = torch.empty(
            int(flush_gpu_cache_size_mb * 1024 * 1024 // 4),
            dtype=torch.float,
            device=device,
        )
        duration_ms_list: List[float] = []

        f_list = [f]
        # make deepcopy of f if necessary
        for _ in range(num_threads - 1):
            f_list.append(copy.deepcopy(f) if copy_f_for_multi_thread_test else f)

        @torch.inference_mode()
        # pyre-ignore[53]
        def forward(idx: int) -> None:
            stream = torch.cuda.Stream()
            f_temp = f_list[idx]
            start_event = [
                torch.cuda.Event(enable_timing=True)
                for i in range(iters // num_threads)
            ]
            end_event = [
                torch.cuda.Event(enable_timing=True)
                for i in range(iters // num_threads)
            ]
            torch.cuda.synchronize(device)
            with torch.cuda.stream(stream):
                for i in range(iters // num_threads):
                    # flush the cache
                    if flush_gpu_cache_size_mb:
                        cache.zero_()
                    start_event[i].record()
                    with torch.cuda.nvtx.range(f"RunCudaModule_{name}"):
                        _ = f_temp(*args)
                    end_event[i].record()
                torch.cuda.synchronize(device)
                times = torch.tensor(
                    [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
                )
                duration_ms = torch.sum(times).item()
                duration_ms_list.append(duration_ms)

        threads = [
            threading.Thread(target=forward, args=(idx,)) for idx in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed_time = sum(duration_ms_list) * 1.0e-3 / num_threads / iters

        torch.cuda.synchronize(device)
        if copy_f_for_multi_thread_test:
            # clean the copies of f and clean the HBM cache
            for idx in reversed(range(num_threads - 1)):
                del f_list[idx + 1]
        torch.cuda.empty_cache()

    else:
        start_time = time.time()
        for _ in range(iters):
            with torch.cuda.nvtx.range(f"RunCPUModule_{name}"):
                output = f(*args)
        elapsed_time = (time.time() - start_time) / iters

    # pyre-fixme[61]: `output` is undefined, or not always defined.
    return float(elapsed_time), output


def benchmark_requests(
    requests: List[TBERequest],
    func: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    flush_gpu_cache_size_mb: int = 0,
    check_median: bool = False,
    num_warmups: int = 0,
    bwd_only: bool = False,
    grad: Optional[torch.Tensor] = None,
    # Used to label benchmark iterations differently in nsys profile result
    # so that we can compare performance of two different models for example.
    # If empty string is provided, it won't have any effect.
    nvtx_range: str = "",
    # Can be used to clear model's stats after warmup for example.
    callback_after_warmup: Optional[Callable[[], None]] = None,
    periodic_logs: bool = False,
    warmup_ms: Optional[int] = None,
) -> float:
    times = []

    # Run at least one warmup iteration to avoid the long cudaLaunchKernel time
    # for the first kernel
    num_warmups = num_warmups + 1 if num_warmups >= 0 else 1

    if warmup_ms:
        indices, offsets, weights = requests[0].unpack_3()
        start_time_ms = time.time() * 1000
        while time.time() * 1000 - start_time_ms < warmup_ms:
            out = func(indices, offsets, weights)
            if bwd_only:
                out.backward(grad)


    if num_warmups > 0:
        indices, offsets, weights = requests[0].unpack_3()
        for _ in range(num_warmups):
            out = func(indices, offsets, weights)
            if bwd_only:
                out.backward(grad)

    if callback_after_warmup is not None:
        callback_after_warmup()

    num_iters = len(requests)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    else:
        start_events = []
        end_events = []

    for it, req in enumerate(requests):
        indices, offsets, weights = req.unpack_3()
        if bwd_only:
            # Run forward before profiling if does backward only
            out = func(indices, offsets, weights)
        start_time = time.time()
        if torch.cuda.is_available():
            if flush_gpu_cache_size_mb:
                _ = torch.rand(
                    flush_gpu_cache_size_mb * 1024 * 1024 // 4,
                    dtype=torch.float,
                    device="cuda",
                )
            start_events[it].record()

        if nvtx_range:
            torch.cuda.nvtx.range_push(f"{nvtx_range}-{it}")

        if bwd_only:
            out.backward(grad)
        else:
            func(indices, offsets, weights)

        if nvtx_range:
            torch.cuda.nvtx.range_pop()

        if torch.cuda.is_available():
            end_events[it].record()
        else:
            it_time = time.time() - start_time
            times.append(it_time)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        times = [
            start.elapsed_time(end) * 1.0e-3
            for start, end in zip(start_events, end_events)
        ]

    if periodic_logs:
        for it in range(100, num_iters + 1, 100):
            times_ = times[0:it]
            avg_time = sum(times_) / len(times_) * 1.0e6
            last_100_avg = sum(times_[-100:]) / 100 * 1.0e6
            logging.info(
                f"Iteration [{it}/{len(requests)}]: Last 100: {last_100_avg:.2f} us, Running avg: {avg_time:.2f} us"
            )

    avg_time = sum(times) / len(requests)
    median_time = statistics.median(times)
    return median_time if check_median else avg_time


def benchmark_requests_refer(
    requests: List[TBERequest],
    T: int,
    B: int,
    L: int,
    E: int,
    D: int,
    pooling_mode: str,
    weighted: bool,
    flush_gpu_cache_size_mb: int = 0,
    check_median: bool = False,
) -> float:
    do_pooling = pooling_mode in ["sum", "mean"]

    if do_pooling:
        nn_embedding_list = [
            torch.nn.EmbeddingBag(E, D, mode=pooling_mode, sparse=True).cuda()
        ] * T
    else:
        nn_embedding_list = [torch.nn.Embedding(E, D, sparse=True).cuda()] * T

    times = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    for req in requests:
        indices, _, weights = req.unpack_3()
        indices_list = indices.view(T, B, L).split(1)

        if weighted:
            assert weights is not None
            weights_list = weights.view(T, B, L).split(1)

        start_time = time.time()
        if torch.cuda.is_available():
            if flush_gpu_cache_size_mb:
                _ = torch.rand(
                    flush_gpu_cache_size_mb * 1024 * 1024 // 4,
                    dtype=torch.float,
                    device="cuda",
                )
                torch.cuda.synchronize()
            start_event.record()

        nn_embedding_output = (
            [
                b_indices(nn_embedding, x, use_cpu=False, do_pooling=do_pooling)
                for (nn_embedding, x) in zip(nn_embedding_list, indices_list)
            ]
            if not weighted
            else [
                b_indices(
                    nn_embedding,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=False,
                    do_pooling=do_pooling,
                )
                for (nn_embedding, x, xw) in zip(
                    nn_embedding_list,
                    indices_list,
                    # pyre-fixme[61]: `weights_list` is undefined, or not always
                    #  defined.
                    weights_list,
                )
            ]
        )

        if do_pooling:
            final_output = torch.cat(
                [f.view(B, -1) for f in nn_embedding_output], dim=1
            )
        else:
            final_output = torch.cat(nn_embedding_output, dim=0).view(  # noqa: F841
                -1, D
            )

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            # pyre-fixme[61]: `end_event` is undefined, or not always defined.
            it_time = start_event.elapsed_time(end_event) * 1.0e-3
            times.append(it_time)
        else:
            it_time = time.time() - start_time
            times.append(it_time)
    avg_time = sum(times) / len(requests)
    median_time = statistics.median(times)
    return median_time if check_median else avg_time


def benchmark_pipelined_requests(
    requests: List[TBERequest],
    func1: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], None],
    func2: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], None],
    flush_gpu_cache_size_mb: int = 0,
    check_median: bool = False,
) -> Tuple[float, float]:
    torch.cuda.synchronize()
    start_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in requests
    ]
    end_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in requests
    ]
    for req, start_event, end_event in zip(requests, start_events, end_events):
        indices, offsets, indices_weights = req.unpack_3()
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4,
                dtype=torch.float,
                device="cuda",
            )
            torch.cuda.synchronize()
        start_event[0].record()
        func1(indices, offsets, indices_weights)
        end_event[0].record()
        start_event[1].record()
        func2(indices, offsets, indices_weights)
        end_event[1].record()
    torch.cuda.synchronize()
    avg_time = (
        sum(
            start_event[0].elapsed_time(end_event[0]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        )
        / len(requests),
        sum(
            start_event[1].elapsed_time(end_event[1]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        )
        / len(requests),
    )
    median_time = (
        statistics.median(
            start_event[0].elapsed_time(end_event[0]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        ),
        statistics.median(
            start_event[1].elapsed_time(end_event[1]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        ),
    )
    return median_time if check_median else avg_time


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


def benchmark_vbe(
    requests: List[Tuple[torch.Tensor, torch.Tensor]],
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[float, float]:
    """
    A benchmark function to return the average execution time in seconds of
    forward and backward of VBE kernels.

    Args:
        requests (List[Tuple[torch.Tensor, torch.Tensor]]):
            A list of requests.  Each request is a tuple
            of indices and offsets.

        func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            A function that takes in indices and offsets
            and returns the output of the VBE kernel.

    Returns:
        Tuple[float, float]:
            A tuple of average execution time in seconds of forward and
            backward of VBE kernels.
    """

    fwd_times = []
    bwd_times = []

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for indices, offsets in requests:
        # forward
        start_event.record()
        out = func(indices, offsets)
        end_event.record()
        torch.cuda.synchronize()
        it_time = start_event.elapsed_time(end_event) * 1.0e-3
        fwd_times.append(it_time)

        grad = torch.rand_like(out)
        start_event.record()
        # backward
        out.backward(grad)
        end_event.record()
        torch.cuda.synchronize()
        it_time = start_event.elapsed_time(end_event) * 1.0e-3
        bwd_times.append(it_time)

    fwd_time_sec = statistics.median(fwd_times)
    bwd_time_sec = statistics.median(bwd_times)

    return fwd_time_sec, bwd_time_sec


def fill_random_scale_bias(
    emb: nn.Module,
    T: int,
    weights_precision: SparseType,
) -> None:
    for t in range(T):
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        (weights, scale_shift) = emb.split_embedding_weights()[t]
        if scale_shift is not None:
            (E, R) = scale_shift.shape
            assert R == 4
            scales = None
            shifts = None
            if weights_precision == SparseType.INT8:
                scales = np.random.uniform(0.001, 0.01, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            elif weights_precision == SparseType.INT4:
                scales = np.random.uniform(0.01, 0.1, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            elif weights_precision == SparseType.INT2:
                scales = np.random.uniform(0.1, 1, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            scale_shift.copy_(
                torch.tensor(
                    np.stack([scales, shifts], axis=1)
                    .astype(np.float16)
                    .view(np.uint8),
                    device=scale_shift.device,
                )
            )
