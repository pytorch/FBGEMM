# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Callable, Tuple

import click
import torch
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


def benchmark_fbgemm_function(
    func: Callable[[Tensor], Tuple[Tensor, Tensor]],
    input: Tensor,
) -> Tuple[float, Tensor]:
    if input.is_cuda:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # Benchmark code
        output, _ = func(input)
        # Accumulate the time for iters iteration
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
    else:
        start_time = time.time()
        output, _ = func(input)
        elapsed_time = time.time() - start_time
    return float(elapsed_time), output


@click.command()
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=2)
def main(
    iters: int,
    warmup_runs: int,
) -> None:

    total_time = {
        "fbgemm_cpu_half": 0.0,
        "fbgemm_cpu_float": 0.0,
        "fbgemm_gpu_half": 0.0,
        "fbgemm_gpu_float": 0.0,
    }

    input_data_cpu = torch.rand(5000, dtype=torch.float)

    bin_num_examples: Tensor = torch.empty([5000], dtype=torch.float64).fill_(0.0)
    bin_num_positives: Tensor = torch.empty([5000], dtype=torch.float64).fill_(0.0)
    lower_bound: float = 0.0
    upper_bound: float = 1.0

    def fbgemm_hbc_cpu(input: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.histogram_binning_calibration(
            input,
            bin_num_examples,
            bin_num_positives,
            0.4,
            lower_bound,
            upper_bound,
            0,
            0.9995,
        )

    for step in range(iters + warmup_runs):
        time, _ = benchmark_fbgemm_function(
            fbgemm_hbc_cpu,
            input_data_cpu.half(),
        )
        if step >= warmup_runs:
            total_time["fbgemm_cpu_half"] += time

        time, _ = benchmark_fbgemm_function(
            fbgemm_hbc_cpu,
            input_data_cpu.float(),
        )
        if step >= warmup_runs:
            total_time["fbgemm_cpu_float"] += time

        if torch.cuda.is_available():
            bin_num_examples_gpu: Tensor = bin_num_examples.cuda()
            bin_num_positives_gpu: Tensor = bin_num_positives.cuda()

            def fbgemm_hbc_gpu(input: Tensor) -> Tuple[Tensor, Tensor]:
                return torch.ops.fbgemm.histogram_binning_calibration(
                    input,
                    bin_num_examples_gpu,
                    bin_num_positives_gpu,
                    0.4,
                    lower_bound,
                    upper_bound,
                    0,
                    0.9995,
                )

            time, _ = benchmark_fbgemm_function(
                fbgemm_hbc_gpu,
                input_data_cpu.cuda().half(),
            )
            if step >= warmup_runs:
                total_time["fbgemm_gpu_half"] += time

            time, _ = benchmark_fbgemm_function(
                fbgemm_hbc_gpu,
                input_data_cpu.cuda().float(),
            )
            if step >= warmup_runs:
                total_time["fbgemm_gpu_float"] += time

    for k, t_time in total_time.items():
        logging.info(f"{k} time per iter: {t_time / iters * 1.0e6:.0f}us")


if __name__ == "__main__":
    main()
