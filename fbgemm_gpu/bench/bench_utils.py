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
from typing import List, Tuple

import torch

logging.basicConfig(level=logging.DEBUG)


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
