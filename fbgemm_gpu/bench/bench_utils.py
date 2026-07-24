# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import threading
import time
from typing import Any, List

import torch

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def _multi_stream_legacy_timing(
    # pyre-fixme[2]: Parameter must be annotated.
    f_list: List[Any],
    # pyre-fixme[2]: Parameter must be annotated.
    args: Any,
    num_threads: int,
    per_thread_iters: int,
    iters: int,
    flush_gpu_cache_size_mb: int,
    device: str,
    name: str,
) -> float:
    """Legacy event-amortized multi-stream timing path."""
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


def _multi_stream_wall_clock_timing(
    # pyre-fixme[2]: Parameter must be annotated.
    f_list: List[Any],
    # pyre-fixme[2]: Parameter must be annotated.
    args: Any,
    num_threads: int,
    per_thread_iters: int,
    num_warmups: int,
    flush_gpu_cache_size_mb: int,
    device: str,
    name: str,
) -> float:
    """Wall-clock multi-stream timing with per-stream warmup."""
    start_barrier = threading.Barrier(num_threads + 1)

    @torch.inference_mode()
    # pyre-ignore[53]
    def forward(idx: int) -> None:
        stream = torch.cuda.Stream()
        f_temp = f_list[idx]
        # Citrine C3: allocate the flush buffer directly on device.
        # Per-stream (not shared) to avoid cross-stream contention.
        cache = (
            torch.empty(
                int(flush_gpu_cache_size_mb * 1024 * 1024 // 4),
                dtype=torch.float,
                device=device,
            )
            if flush_gpu_cache_size_mb
            else None
        )
        with torch.cuda.stream(stream):
            for _ in range(num_warmups):
                _ = f_temp(*args)
            stream.synchronize()
        start_barrier.wait()
        with torch.cuda.stream(stream):
            for _ in range(per_thread_iters):
                if cache is not None:
                    cache.zero_()
                with torch.cuda.nvtx.range(f"RunCudaModule_{name}"):
                    _ = f_temp(*args)
            stream.synchronize()

    threads = [
        threading.Thread(target=forward, args=(idx,)) for idx in range(num_threads)
    ]
    for thread in threads:
        thread.start()
    # Set the start AFTER workers reach the barrier (post-warmup) so the
    # timed window excludes warmup; the barrier release is the t0.
    start_barrier.wait()
    start_time = time.perf_counter()
    for thread in threads:
        thread.join()
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - start_time
    # Achieved concurrent throughput as per-iter wall time: caller's
    # batch/elapsed = (per_thread_iters*num_threads*batch)/wall.
    return wall_s / (per_thread_iters * num_threads)


def benchmark_torch_function(  # noqa: C901
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
    legacy_multi_stream_timing: bool = False,
) -> tuple[float, torch.Tensor]:
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
        # Multi-stream throughput. Each thread runs its own non-default stream.
        # Defaults are tuned for stable cross-run measurement:
        #  (1) warm up EACH stream before timing -- the timed iters run on
        #      non-default streams, so the default-stream warmup above leaves
        #      cold first-iters that inflate and destabilize the result; and
        #  (2) time the whole concurrent run with a single wall clock + one
        #      final sync, reporting achieved throughput as per-iter wall time
        #      (wall / total_iters). The older approach summed each stream's
        #      per-iter CUDA-event times and divided by num_threads, which
        #      reconstructs throughput under an idealized-overlap assumption
        #      that fluctuates run-to-run (large T=2 variance). Set
        #      legacy_multi_stream_timing=True to restore it.
        # The returned value keeps the same "seconds per iter" contract either
        # way, so callers computing batch/elapsed are unaffected.
        per_thread_iters = max(1, iters // num_threads)

        f_list = [f]
        # make deepcopy of f if necessary
        for _ in range(num_threads - 1):
            f_list.append(copy.deepcopy(f) if copy_f_for_multi_thread_test else f)

        if legacy_multi_stream_timing:
            elapsed_time = _multi_stream_legacy_timing(
                f_list=f_list,
                args=args,
                num_threads=num_threads,
                per_thread_iters=per_thread_iters,
                iters=iters,
                flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
                device=device,
                name=name,
            )
        else:
            elapsed_time = _multi_stream_wall_clock_timing(
                f_list=f_list,
                args=args,
                num_threads=num_threads,
                per_thread_iters=per_thread_iters,
                num_warmups=num_warmups,
                flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
                device=device,
                name=name,
            )

        torch.cuda.synchronize(device)
        if copy_f_for_multi_thread_test:
            # clean the copies of f and clean the HBM cache
            for idx in reversed(range(num_threads - 1)):
                del f_list[idx + 1]
        torch.cuda.empty_cache()

    else:
        use_nvtx = torch.cuda.is_available()
        start_time = time.time()
        for _ in range(iters):
            if use_nvtx:
                with torch.cuda.nvtx.range(f"RunCPUModule_{name}"):
                    output = f(*args)
            else:
                output = f(*args)
        elapsed_time = (time.time() - start_time) / iters

    # pyre-fixme[61]: `output` is undefined, or not always defined.
    return float(elapsed_time), output
