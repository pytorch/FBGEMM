# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import statistics
import threading
import time
from subprocess import Popen
from typing import Callable, List, Optional, Tuple

import torch

from fbgemm_gpu.tbe.utils import b_indices, TBERequest


logging.basicConfig(level=logging.DEBUG)


def bench_warmup(
    request: TBERequest,
    warmup_ms: int,
    warmup_runs: int,
    func: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    bwd_only: bool = False,
    grad: Optional[torch.Tensor] = None,
) -> None:
    indices, offsets, weights = request.unpack_3()
    if warmup_ms:
        start_time_ms = time.time() * 1000
        while time.time() * 1000 - start_time_ms < warmup_ms:
            out = func(indices, offsets, weights)
            if bwd_only:
                out.backward(grad)
    else:
        for _ in range(warmup_runs):
            out = func(indices, offsets, weights)
            if bwd_only:
                out.backward(grad)


class BMBarrier:

    def __init__(self) -> None:
        self.bar: Optional[threading.Barrier] = None

    def create_barrier(self, party_size: int) -> None:
        if self.bar is not None:
            self.bar.reset()
            self.bar = None
        self.bar = torch.multiprocessing.Barrier(party_size)

    def wait(self) -> None:
        if self.bar is not None:
            self.bar.wait()


# This barrier ensures all CPU TBE workers start the embedding workload
# together so that we get the most accurate measurement. This needs to be
# a global variable because it will be shared among worker processes.
cpu_bm_barrier = BMBarrier()


def cpu_tbe_worker(
    requests_: List[TBERequest],
    func_: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    use_barrier: bool = False,
) -> float:
    """
    Worker function to process CPU TBE workload.

    Args:
        requests_ (List[TBERequest]): A list of TBERequest objects to be processed. Namely, the dataset.
        func_ (Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]):
            The function to process each request, usually the `.forward()` method
            n the embedding module instance.
        use_barrier (bool, optional): Whether to use a barrier to synchronize the
            start of embedding workload. Defaults to False.

    Returns:
        float: The average runtime per iteration in seconds.
    """
    import time

    if use_barrier:
        cpu_bm_barrier.wait()

    start_time = time.perf_counter()
    for req in requests_:
        func_(*(req.unpack_3()))
    end_time = time.perf_counter()

    return (end_time - start_time) / len(requests_)


def benchmark_cpu_requests_mp(
    requests: List[TBERequest],
    emb_module: torch.nn.Module,
    num_warmups: int = 0,
    num_copies: int = 1,
    start_script: str = "",
    end_script: str = "",
) -> float:
    """
    CPU benchmark request handler with multi-processing support

    Args:
        requests (List[TBERequest]): A list of TBERequest objects to be processed.
        emb_module (torch.nn.Module): The embedding module to be used for processing requests,
            for example, an instance of `IntNBitTableBatchedEmbeddingBagsCodegen` module.
        num_warmups (int, optional): Number of warm-up iterations to perform before benchmarking. Defaults to 0.
        num_copies (int, optional): Number of parallel copies of the workloads. By `copies`,
            we mean the number of parallel processes working on the same dataset described in `requests`.
            Defaults to 1 (which means single threaded). Increasing this will enable the benchmark to use
            more CPU cores and push higher memory bandwidth.
        start_script (str, optional): Path to a script to be executed before starting the benchmark.
            Defaults to empty (not running anything). This can be used to collect perf counters.
            The script will be terminated upon benchmark finishing.
        end_script (str, optional): Path to a script to be executed after completing the benchmark.
            Defaults to empty (not running anything). This can be used to post-process perf counters.

    Returns:
        float: The average runtime per iteration in seconds.

    """
    cpu_bm_barrier.create_barrier(num_copies)
    worker_pool = torch.multiprocessing.Pool(num_copies)

    if num_warmups > 0:
        asyncres = []
        for _ in range(num_copies):
            asyncres.append(
                worker_pool.apply_async(
                    cpu_tbe_worker,
                    args=(
                        [requests[0]],
                        emb_module.forward,
                        False,
                        num_warmups,
                    ),
                )
            )
        for res in asyncres:
            res.wait()

    if start_script:
        p_start = Popen([start_script, str(num_copies)])

    asyncres = []
    for _ in range(num_copies):
        asyncres.append(
            worker_pool.apply_async(
                cpu_tbe_worker,
                args=(
                    requests,
                    emb_module.forward,
                    True,
                ),
            )
        )
    runtime_per_iter = 0.0
    for res in asyncres:
        res.wait()
        runtime_per_iter += res.get()
    worker_pool.close()
    worker_pool.join()
    worker_pool.terminate()

    if start_script:
        p_start.terminate()

    if end_script:
        p_end = Popen([end_script, str(num_copies)])
        p_end.wait()

    return runtime_per_iter / num_copies


def benchmark_cpu_requests(
    requests: List[TBERequest],
    func: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    num_warmups: int = 0,
) -> float:
    import time

    if num_warmups > 0:
        for _ in range(num_warmups):
            func(*(requests[0].unpack_3()))

    start_time = time.perf_counter()
    for req in requests:
        func(*(req.unpack_3()))
    end_time = time.perf_counter()
    return (end_time - start_time) / len(requests)


def benchmark_requests(  # noqa: C901
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
    iters: int = -1,
) -> float:
    times = []
    # Run at least one warmup iteration to avoid the long cudaLaunchKernel time
    # for the first kernel if warmup_ms > 0
    # warmup_ms is prioritized over num_warmups

    if warmup_ms is None:
        num_warmups = num_warmups + 1 if num_warmups >= 0 else 1

    # warm-up the GPU before profiling
    bench_warmup(
        requests[0],
        # pyre-ignore[6]
        warmup_ms,
        num_warmups,
        lambda indices, offsets, per_sample_weights: func(
            indices,
            offsets,
            per_sample_weights,
        ),
        bwd_only=bwd_only,
        grad=grad,
    )

    if callback_after_warmup is not None:
        callback_after_warmup()

    num_reqs = len(requests)
    iters = num_reqs if iters == -1 else iters

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    else:
        start_events = []
        end_events = []

    for it in range(iters):
        req = requests[it % num_reqs]

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
        for it in range(100, iters + 1, 100):
            times_ = times[0:it]
            avg_time = sum(times_) / len(times_) * 1.0e6
            last_100_avg = sum(times_[-100:]) / 100 * 1.0e6
            logging.info(
                f"Iteration [{it}/{len(requests)}]: Last 100: {last_100_avg:.2f} us, Running avg: {avg_time:.2f} us"
            )

    avg_time = sum(times) / iters
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


def benchmark_vbe(
    requests: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    func: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    num_warmups: int = 0,
) -> Tuple[float, float]:
    """
    A benchmark function to return the average execution time in seconds of
    forward and backward of VBE kernels.

    Args:
        requests (List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]):
            A list of requests.  Each request is a tuple
            of indices, offsets and weights.

        func (Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]):
            A function that takes in indices, offsets, and weights
            and returns the output of the VBE kernel.

        num_warmups (int):
            The number of warm-up iterations before measuring performance.

    Returns:
        Tuple[float, float]:
            A tuple of average execution time in seconds of forward and
            backward of VBE kernels.
    """

    use_cuda = torch.cuda.is_available()

    # Warm-ups.
    for _ in range(num_warmups):
        # Warm-up using the first request as done in benchmark_requests
        indices, offsets, weights = requests[0]
        out = func(indices, offsets, weights)
        grad = torch.rand_like(out)
        out.backward(grad)

    iters = len(requests)
    if use_cuda:
        fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        torch.cuda.synchronize()
    else:
        # Actual measurement in seconds.
        fwd_times_sec = []
        bwd_times_sec = []

    for i, (indices, offsets, weights) in enumerate(requests):
        # forward
        if use_cuda:
            # pyre-ignore[61]
            fwd_start_events[i].record()
        else:
            start_time = time.time()

        out = func(indices, offsets, weights)
        if use_cuda:
            # pyre-ignore[61]
            fwd_end_events[i].record()
        else:
            # pyre-ignore[61]
            fwd_times_sec.append(time.time() - start_time)

        grad = torch.rand_like(out)

        if use_cuda:
            # pyre-ignore[61]
            bwd_start_events[i].record()
        else:
            start_time = time.time()
        # backward
        out.backward(grad)
        if use_cuda:
            # pyre-ignore[61]
            bwd_end_events[i].record()
        else:
            # pyre-ignore[61]
            bwd_times_sec.append(time.time() - start_time)

    if use_cuda:
        torch.cuda.synchronize()

    if use_cuda:
        fwd_times_sec = [
            start_event.elapsed_time(end_event) * 1.0e-3
            # pyre-ignore[61]
            for start_event, end_event in zip(fwd_start_events, fwd_end_events)
        ]
        bwd_times_sec = [
            start_event.elapsed_time(end_event) * 1.0e-3
            # pyre-ignore[61]
            for start_event, end_event in zip(bwd_start_events, bwd_end_events)
        ]

    # pyre-ignore[61]
    fwd_time_sec = statistics.median(fwd_times_sec)
    # pyre-ignore[61]
    bwd_time_sec = statistics.median(bwd_times_sec)

    return fwd_time_sec, bwd_time_sec
