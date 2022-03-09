import time
from typing import Callable, Tuple, Any

import torch
from torch import Tensor


def benchmark_torch_function_with_output(
    flush_gpu_cache_size_mb: int,
    func: Callable,
    *args: Any,
) -> Tuple[float, Tensor]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # Flush the cache
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
            )
            torch.cuda.synchronize()
        start_event.record()
        # Benchmark code
        output = func(*args)
        # Accumulate the time for iters iteration
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
    else:
        start_time = time.time()
        output = func(*args)
        elapsed_time = time.time() - start_time
    return float(elapsed_time), output


def benchmark_torch_function(
    flush_gpu_cache_size_mb: int,
    iters: int,
    warmup_runs: int,
    func: Callable,
    *args: Any,
) -> float:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Flush the cache
    if flush_gpu_cache_size_mb:
        _ = torch.rand(flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float)
        torch.cuda.synchronize()
    for i in range(iters):
        if i == warmup_runs:
            start_event.record()
        func(*args)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters
