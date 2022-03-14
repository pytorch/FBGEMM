# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Tuple

import torch
from torch import Tensor


def benchmark_torch_function(
    # pyre-fixme[2]: Parameter must be annotated.
    f,
    # pyre-fixme[2]: Parameter must be annotated.
    args,
    flush_gpu_cache_size_mb: int = 40,
    iters: int = 10,
    num_warmups: int = 2,
) -> Tuple[float, Tensor]:
    for _ in range(num_warmups):
        output = f(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # flush the cache
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
            )
            torch.cuda.synchronize()
        start_event.record()
        for _ in range(iters):
            output = f(*args)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
    else:
        start_time = time.time()
        for _ in range(iters):
            output = f(*args)
        elapsed_time = time.time() - start_time

    # pyre-fixme[61]: `output` is undefined, or not always defined.
    return float(elapsed_time) / iters, output
