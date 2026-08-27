# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from functools import partial
from typing import Any, TypedDict

import torch

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BenchmarkWorkerDiagnostics(TypedDict):
    stream_handle: int
    device_span_ms: float


class BenchmarkDiagnostics(TypedDict, total=False):
    timing_method: str
    uses_provided_streams: bool
    per_thread_iters: int
    wall_s: float
    workers: list[BenchmarkWorkerDiagnostics]
    max_device_span_ms: float
    min_device_span_ms: float
    event_mean_ms: float
    device_span_ms: float


def _multi_stream_event_amortized_timing(
    # pyre-fixme[2]: Parameter must be annotated.
    f_list: list[Any],
    # pyre-fixme[2]: Parameter must be annotated.
    args: Any,
    num_threads: int,
    per_thread_iters: int,
    iters: int,
    flush_gpu_cache_size_mb: int,
    device: str,
    name: str,
) -> float:
    """Estimate concurrent throughput from per-call device event durations."""
    cache = torch.empty(
        int(flush_gpu_cache_size_mb * 1024 * 1024 // 4),
        dtype=torch.float,
        device=device,
    )
    duration_ms_list: list[float] = []

    @torch.inference_mode()
    # pyre-ignore[53]
    def forward(idx: int) -> None:
        stream = torch.cuda.Stream()
        f_temp = f_list[idx]
        start_event = [
            torch.cuda.Event(enable_timing=True) for i in range(per_thread_iters)
        ]
        end_event = [
            torch.cuda.Event(enable_timing=True) for i in range(per_thread_iters)
        ]
        torch.cuda.synchronize(device)
        with torch.cuda.stream(stream):
            for i in range(per_thread_iters):
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
            duration_ms_list.append(torch.sum(times).item())

    threads = [
        threading.Thread(target=forward, args=(idx,)) for idx in range(num_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sum(duration_ms_list) * 1.0e-3 / num_threads / iters


def _run_wall_clock_worker(
    f: Any,
    args: Any,
    kwargs: Any,
    stream: Any,
    start_barrier: threading.Barrier,
    per_thread_iters: int,
    num_warmups: int,
    flush_gpu_cache_size_mb: int,
    device: str,
    name: str,
    collect_diagnostics: bool,
) -> tuple[Any, BenchmarkWorkerDiagnostics | None]:
    torch.cuda.set_device(stream.device)
    span_start = torch.cuda.Event(enable_timing=True) if collect_diagnostics else None
    span_end = torch.cuda.Event(enable_timing=True) if collect_diagnostics else None
    # Citrine C3: allocate the flush buffer directly on device.
    cache = (
        torch.empty(
            int(flush_gpu_cache_size_mb * 1024 * 1024 // 4),
            dtype=torch.float,
            device=device,
        )
        if flush_gpu_cache_size_mb
        else None
    )
    output = None
    with torch.cuda.stream(stream):
        for _ in range(num_warmups):
            output = f(*args, **kwargs)
        if cache is not None:
            cache.zero_()
        stream.synchronize()

    start_barrier.wait()
    with torch.cuda.stream(stream):
        if span_start is not None:
            span_start.record()
        for _ in range(per_thread_iters):
            with torch.cuda.nvtx.range(f"RunCudaModule_{name}"):
                output = f(*args, **kwargs)
        if span_end is not None:
            span_end.record()
        stream.synchronize()

    if span_start is None or span_end is None:
        return output, None
    return output, {
        "stream_handle": int(stream.cuda_stream),
        "device_span_ms": span_start.elapsed_time(span_end),
    }


def _record_wall_clock_diagnostics(
    diagnostics: BenchmarkDiagnostics,
    worker_diagnostics: list[BenchmarkWorkerDiagnostics | None],
    streams_provided: bool,
    per_thread_iters: int,
    wall_s: float,
) -> None:
    workers = [record for record in worker_diagnostics if record is not None]
    diagnostics.update(
        {
            "timing_method": "multi_stream_wall_clock",
            "uses_provided_streams": streams_provided,
            "per_thread_iters": per_thread_iters,
            "wall_s": wall_s,
            "workers": workers,
        }
    )
    if not workers:
        return
    device_spans_ms = [float(record["device_span_ms"]) for record in workers]
    diagnostics.update(
        {
            "max_device_span_ms": max(device_spans_ms),
            "min_device_span_ms": min(device_spans_ms),
        }
    )


def _prepare_wall_clock_streams(
    streams: list[Any] | None,
    num_threads: int,
    device: str,
) -> list[Any]:
    stream_list = (
        streams
        if streams is not None
        else [torch.cuda.Stream() for _ in range(num_threads)]
    )
    caller_stream = torch.cuda.current_stream(device)
    for stream in stream_list:
        stream.wait_stream(caller_stream)
    return stream_list


@torch.inference_mode()
# pyre-ignore[53]
def _run_wall_clock_worker_safely(
    idx: int,
    f_list: list[Any],
    args: Any,
    kwargs: Any,
    stream_list: list[Any],
    start_barrier: threading.Barrier,
    per_thread_iters: int,
    num_warmups: int,
    flush_gpu_cache_size_mb: int,
    device: str,
    name: str,
    collect_diagnostics: bool,
    worker_outputs: list[Any],
    worker_diagnostics: list[BenchmarkWorkerDiagnostics | None],
    worker_errors: queue.SimpleQueue[Exception],
) -> None:
    try:
        output, worker_diagnostic = _run_wall_clock_worker(
            f=f_list[idx],
            args=args,
            kwargs=kwargs,
            stream=stream_list[idx],
            start_barrier=start_barrier,
            per_thread_iters=per_thread_iters,
            num_warmups=num_warmups,
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            device=device,
            name=name,
            collect_diagnostics=collect_diagnostics,
        )
        worker_outputs[idx] = output
        worker_diagnostics[idx] = worker_diagnostic
    except Exception as error:
        worker_errors.put(error)
        start_barrier.abort()


def _execute_wall_clock_workers(
    f_list: list[Any],
    args: Any,
    kwargs: Any,
    stream_list: list[Any],
    start_barrier: threading.Barrier,
    per_thread_iters: int,
    num_warmups: int,
    flush_gpu_cache_size_mb: int,
    device: str,
    name: str,
    collect_diagnostics: bool,
    worker_outputs: list[Any],
    worker_diagnostics: list[BenchmarkWorkerDiagnostics | None],
    worker_errors: queue.SimpleQueue[Exception],
) -> None:
    worker = partial(
        _run_wall_clock_worker_safely,
        f_list=f_list,
        args=args,
        kwargs=kwargs,
        stream_list=stream_list,
        start_barrier=start_barrier,
        per_thread_iters=per_thread_iters,
        num_warmups=num_warmups,
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        device=device,
        name=name,
        collect_diagnostics=collect_diagnostics,
        worker_outputs=worker_outputs,
        worker_diagnostics=worker_diagnostics,
        worker_errors=worker_errors,
    )
    threads = [
        threading.Thread(target=worker, args=(idx,)) for idx in range(len(f_list))
    ]
    for thread in threads:
        thread.start()
    try:
        start_barrier.wait()
    except threading.BrokenBarrierError:
        for thread in threads:
            thread.join()
        if not worker_errors.empty():
            raise worker_errors.get()
        raise
    for thread in threads:
        thread.join()
    if not worker_errors.empty():
        raise worker_errors.get()


def _multi_stream_wall_clock_timing(
    # pyre-fixme[2]: Parameter must be annotated.
    f_list: list[Any],
    # pyre-fixme[2]: Parameter must be annotated.
    args: Any,
    # pyre-fixme[2]: Parameter must be annotated.
    kwargs: Any,
    num_threads: int,
    per_thread_iters: int,
    num_warmups: int,
    flush_gpu_cache_size_mb: int,
    device: str,
    name: str,
    diagnostics: BenchmarkDiagnostics | None,
    streams: list[Any] | None,
) -> tuple[float, Any]:
    """Wall-clock multi-stream timing with per-stream warmup."""
    stream_list = _prepare_wall_clock_streams(
        streams,
        num_threads,
        device,
    )
    start_time_ns: list[int] = []
    start_barrier = threading.Barrier(
        num_threads + 1,
        action=lambda: start_time_ns.append(time.perf_counter_ns()),
    )
    worker_diagnostics: list[BenchmarkWorkerDiagnostics | None] = [
        None for _ in range(num_threads)
    ]
    worker_outputs: list[Any] = [None for _ in range(num_threads)]
    worker_errors: queue.SimpleQueue[Exception] = queue.SimpleQueue()

    # Prime lazy runtime initialization on a measured stream so affinity-based
    # runtime pools never observe an extra default-stream key.
    with torch.inference_mode(), torch.cuda.stream(stream_list[0]):
        worker_outputs[0] = f_list[0](*args, **kwargs)
        stream_list[0].synchronize()

    _execute_wall_clock_workers(
        f_list=f_list,
        args=args,
        kwargs=kwargs,
        stream_list=stream_list,
        start_barrier=start_barrier,
        per_thread_iters=per_thread_iters,
        num_warmups=num_warmups,
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        device=device,
        name=name,
        collect_diagnostics=diagnostics is not None,
        worker_outputs=worker_outputs,
        worker_diagnostics=worker_diagnostics,
        worker_errors=worker_errors,
    )
    torch.cuda.synchronize(device)
    end_time_ns = time.perf_counter_ns()
    wall_s = (end_time_ns - start_time_ns[0]) * 1.0e-9
    if diagnostics is not None:
        _record_wall_clock_diagnostics(
            diagnostics,
            worker_diagnostics,
            streams is not None,
            per_thread_iters,
            wall_s,
        )
    return wall_s / (per_thread_iters * num_threads), worker_outputs[0]


def _validate_benchmark_inputs(
    num_threads: int,
    num_warmups: int,
    wall_clock_multi_stream_timing: bool,
    streams: list[Any] | None,
) -> None:
    assert num_threads > 0
    if (
        wall_clock_multi_stream_timing
        and (num_threads > 1 or streams is not None)
        and num_warmups < 1
    ):
        raise ValueError("wall-clock multi-stream timing requires num_warmups >= 1")
    if streams is None:
        return
    if len(streams) != num_threads:
        raise ValueError(f"Expected {num_threads} streams, received {len(streams)}")
    if not wall_clock_multi_stream_timing:
        raise ValueError("Provided streams require wall-clock timing")


def _warm_up(
    f: Any,
    args: Any,
    kwargs: Any,
    num_warmups: int,
) -> Any:
    output = None
    for _ in range(num_warmups):
        output = f(*args, **kwargs)
    return output


def _single_stream_event_timing(
    f: Any,
    args: Any,
    flush_gpu_cache_size_mb: int,
    iters: int,
    device: str,
    name: str,
    diagnostics: BenchmarkDiagnostics | None,
) -> tuple[float, Any]:
    cache = torch.empty(
        int(flush_gpu_cache_size_mb * 1024 * 1024 // 4),
        dtype=torch.float,
        device=device,
    )
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize(device)
    diagnostic_start_ns = time.perf_counter_ns()
    output = None
    for i in range(iters):
        if flush_gpu_cache_size_mb:
            cache.zero_()
        start_event[i].record()
        with torch.cuda.nvtx.range(f"RunCudaModule_{name}"):
            output = f(*args)
        end_event[i].record()
    torch.cuda.synchronize(device)
    diagnostic_end_ns = time.perf_counter_ns()
    times = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(start_event, end_event)]
    )
    elapsed_time = torch.mean(times).item() * 1.0e-3
    if diagnostics is not None:
        diagnostics.update(
            {
                "timing_method": "single_stream_events",
                "per_thread_iters": iters,
                "wall_s": (diagnostic_end_ns - diagnostic_start_ns) * 1.0e-9,
                "event_mean_ms": elapsed_time * 1.0e3,
                "device_span_ms": start_event[0].elapsed_time(end_event[-1]),
            }
        )
    return elapsed_time, output


def _multi_stream_timing(
    f: Any,
    args: Any,
    kwargs: Any,
    output: Any,
    flush_gpu_cache_size_mb: int,
    iters: int,
    num_warmups: int,
    device: str,
    name: str,
    num_threads: int,
    copy_f_for_multi_thread_test: bool,
    wall_clock_multi_stream_timing: bool,
    diagnostics: BenchmarkDiagnostics | None,
    streams: list[Any] | None,
) -> tuple[float, Any]:
    if wall_clock_multi_stream_timing and (
        iters < num_threads or iters % num_threads != 0
    ):
        raise ValueError(
            f"iters ({iters}) must be a positive multiple of "
            f"num_threads ({num_threads})"
        )
    per_thread_iters = iters // num_threads
    f_list = [f]
    for _ in range(num_threads - 1):
        f_list.append(copy.deepcopy(f) if copy_f_for_multi_thread_test else f)

    if wall_clock_multi_stream_timing:
        elapsed_time, output = _multi_stream_wall_clock_timing(
            f_list=f_list,
            args=args,
            kwargs=kwargs,
            num_threads=num_threads,
            per_thread_iters=per_thread_iters,
            num_warmups=num_warmups,
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            device=device,
            name=name,
            diagnostics=diagnostics,
            streams=streams,
        )
    else:
        elapsed_time = _multi_stream_event_amortized_timing(
            f_list=f_list,
            args=args,
            num_threads=num_threads,
            per_thread_iters=per_thread_iters,
            iters=iters,
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
            device=device,
            name=name,
        )
        if diagnostics is not None:
            diagnostics["timing_method"] = "multi_stream_event_amortized"

    torch.cuda.synchronize(device)
    if copy_f_for_multi_thread_test:
        for idx in reversed(range(num_threads - 1)):
            del f_list[idx + 1]
    torch.cuda.empty_cache()
    return elapsed_time, output


def _cpu_timing(
    f: Any,
    args: Any,
    kwargs: Any,
    iters: int,
    name: str,
) -> tuple[float, Any]:
    use_nvtx = torch.cuda.is_available()
    start_time = time.time()
    output = None
    for _ in range(iters):
        if use_nvtx:
            with torch.cuda.nvtx.range(f"RunCPUModule_{name}"):
                output = f(*args, **kwargs)
        else:
            output = f(*args, **kwargs)
    return (time.time() - start_time) / iters, output


def benchmark_torch_function(
    # pyre-fixme[2]: Parameter must be annotated.
    f,
    # pyre-fixme[2]: Parameter must be annotated.
    args,
    # pyre-fixme[2]: Parameter must be annotated.
    kwargs={},  # noqa: B006
    flush_gpu_cache_size_mb: int = 40,
    iters: int = 10,
    num_warmups: int = 2,
    device: str = "cuda",
    name: str = "",
    num_threads: int = 1,
    copy_f_for_multi_thread_test: bool = False,
    wall_clock_multi_stream_timing: bool = False,
    diagnostics: BenchmarkDiagnostics | None = None,
    streams: list[Any] | None = None,
) -> tuple[float, torch.Tensor]:
    logging.debug(f"Start to benchmark {name}...")
    if device != "cpu" and device != "" and device != "cuda":
        torch.cuda.set_device(device)
    _validate_benchmark_inputs(
        num_threads,
        num_warmups,
        wall_clock_multi_stream_timing,
        streams,
    )
    is_cuda_benchmark = device != "cpu" and torch.cuda.is_available()
    is_multi_stream_benchmark = num_threads > 1 or streams is not None
    uses_wall_clock_multi_stream_timing = (
        is_cuda_benchmark
        and is_multi_stream_benchmark
        and wall_clock_multi_stream_timing
    )
    output = (
        None
        if uses_wall_clock_multi_stream_timing
        else _warm_up(f, args, kwargs, num_warmups)
    )

    if not is_cuda_benchmark:
        elapsed_time, output = _cpu_timing(f, args, kwargs, iters, name)
    elif not is_multi_stream_benchmark:
        elapsed_time, output = _single_stream_event_timing(
            f,
            args,
            flush_gpu_cache_size_mb,
            iters,
            device,
            name,
            diagnostics,
        )
    else:
        elapsed_time, output = _multi_stream_timing(
            f,
            args,
            kwargs,
            output,
            flush_gpu_cache_size_mb,
            iters,
            num_warmups,
            device,
            name,
            num_threads,
            copy_f_for_multi_thread_test,
            wall_clock_multi_stream_timing,
            diagnostics,
            streams,
        )

    return float(elapsed_time), output
