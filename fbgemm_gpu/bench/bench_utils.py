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
import typing

import torch

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def _setup_device(
    device: str,
    name: str,
    # pyre-fixme[2]: Parameter annotation for return type.
) -> tuple[typing.Any, typing.Any, str]:
    """Initialize the target device and return (device_mod, create_event, nvtx_label).

    Returns (None, None, label) for CPU — callers should check device_mod
    before using accelerator-specific APIs.
    """
    if device.startswith("mtia"):
        import mtia.host_runtime.torch_mtia.dynamic_library  # noqa  # pyre-fixme[21]

        if not torch.mtia.is_available():
            torch.mtia.init()
        if device != "mtia":
            torch.mtia.set_device(device)
        return torch.mtia, torch.mtia.Event, f"RunMtiaModule_{name}"

    if device not in ("cpu", ""):
        if device != "cuda":
            torch.cuda.set_device(device)
        return (
            torch.cuda,
            lambda: torch.cuda.Event(enable_timing=True),
            f"RunCudaModule_{name}",
        )

    return None, None, f"RunCPUModule_{name}"


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
) -> tuple[float, torch.Tensor]:
    logging.debug(f"Start to benchmark {name}...")

    device_mod, create_event, nvtx_label = _setup_device(device, name)

    for _ in range(num_warmups):
        output = f(*args, **kwargs)

    assert num_threads > 0

    is_device_available = device_mod is not None and device_mod.is_available()

    # Single-threaded GPU/accelerator benchmarking
    if is_device_available and num_threads == 1:
        cache = torch.empty(
            int(flush_gpu_cache_size_mb * 1024 * 1024 // 4),
            dtype=torch.float,
            device=device,
        )
        start_event = [create_event() for i in range(iters)]
        end_event = [create_event() for i in range(iters)]
        device_mod.synchronize(device)
        for i in range(iters):
            # flush the cache
            if flush_gpu_cache_size_mb:
                cache.zero_()
            start_event[i].record()
            with torch.cuda.nvtx.range(nvtx_label):
                output = f(*args)
            end_event[i].record()
        device_mod.synchronize(device)
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
        )
        elapsed_time = torch.mean(times).item() * 1.0e-3
    elif is_device_available and num_threads > 1:
        cache = torch.empty(
            int(flush_gpu_cache_size_mb * 1024 * 1024 // 4),
            dtype=torch.float,
            device=device,
        )
        duration_ms_list: list[float] = []

        f_list = [f]
        # make deepcopy of f if necessary
        for _ in range(num_threads - 1):
            f_list.append(copy.deepcopy(f) if copy_f_for_multi_thread_test else f)

        @torch.inference_mode()
        # pyre-ignore[53]
        def forward(idx: int) -> None:
            stream = device_mod.Stream()
            f_temp = f_list[idx]
            start_event = [create_event() for i in range(iters // num_threads)]
            end_event = [create_event() for i in range(iters // num_threads)]
            device_mod.synchronize(device)
            with device_mod.stream(stream):
                for i in range(iters // num_threads):
                    # flush the cache
                    if flush_gpu_cache_size_mb:
                        cache.zero_()
                    start_event[i].record()
                    with torch.cuda.nvtx.range(nvtx_label):
                        _ = f_temp(*args)
                    end_event[i].record()
                device_mod.synchronize(device)
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

        device_mod.synchronize(device)
        if copy_f_for_multi_thread_test:
            # clean the copies of f and clean the HBM cache
            for idx in reversed(range(num_threads - 1)):
                del f_list[idx + 1]
        if hasattr(device_mod, "empty_cache"):
            device_mod.empty_cache()

    else:
        start_time = time.time()
        for _ in range(iters):
            with torch.cuda.nvtx.range(nvtx_label):
                output = f(*args)
        elapsed_time = (time.time() - start_time) / iters

    # pyre-fixme[61]: `output` is undefined, or not always defined.
    return float(elapsed_time), output
