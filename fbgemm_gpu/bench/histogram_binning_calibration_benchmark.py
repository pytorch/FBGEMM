# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
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
    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_hip")
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


def benchmark_hbc_function(
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
    data_types = [torch.half, torch.float, torch.double]

    total_time = {
        "hbc": {
            "cpu": {
                torch.half: 0.0,
                torch.float: 0.0,
                torch.double: 0.0,
            },
            "gpu": {
                torch.half: 0.0,
                torch.float: 0.0,
                torch.double: 0.0,
            },
        },
        "hbc_by_feature": {
            "cpu": {
                torch.half: 0.0,
                torch.float: 0.0,
                torch.double: 0.0,
            },
            "gpu": {
                torch.half: 0.0,
                torch.float: 0.0,
                torch.double: 0.0,
            },
        },
        "generic_hbc_by_feature": {
            "cpu": {
                torch.half: 0.0,
                torch.float: 0.0,
                torch.double: 0.0,
            },
            "gpu": {
                torch.half: 0.0,
                torch.float: 0.0,
                torch.double: 0.0,
            },
        },
    }

    num_bins: int = 5000
    num_segments: int = 42

    num_logits = 5000
    input_data_cpu = torch.rand(num_logits, dtype=torch.float)
    segment_lengths: Tensor = torch.randint(0, 2, (num_logits,))
    num_values: int = int(torch.sum(segment_lengths).item())
    segment_values: Tensor = torch.randint(
        0,
        num_segments,
        (num_values,),
    )

    lower_bound: float = 0.0
    upper_bound: float = 1.0
    w: float = (upper_bound - lower_bound) / num_bins

    bin_num_examples: Tensor = torch.empty([num_bins], dtype=torch.float64).fill_(0.0)
    bin_num_positives: Tensor = torch.empty([num_bins], dtype=torch.float64).fill_(0.0)
    bin_boundaries: Tensor = torch.arange(
        lower_bound + w, upper_bound - w / 2, w, dtype=torch.float64
    )

    by_feature_bin_num_examples: Tensor = torch.empty(
        [num_bins * (num_segments + 1)], dtype=torch.float64
    ).fill_(0.0)
    by_feature_bin_num_positives: Tensor = torch.empty(
        [num_bins * (num_segments + 1)], dtype=torch.float64
    ).fill_(0.0)

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

    def fbgemm_hbc_by_feature_cpu(input: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.histogram_binning_calibration_by_feature(
            input,
            segment_values,
            segment_lengths,
            num_segments,
            by_feature_bin_num_examples,
            by_feature_bin_num_positives,
            num_bins,
            0.4,
            lower_bound,
            upper_bound,
            0,
            0.9995,
        )

    def fbgemm_generic_hbc_by_feature_cpu(input: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            input,
            segment_values,
            segment_lengths,
            num_segments,
            by_feature_bin_num_examples,
            by_feature_bin_num_positives,
            bin_boundaries,
            0.4,
            0,
            0.9995,
        )

    for step in range(iters + warmup_runs):
        for data_type in data_types:
            curr_input = input_data_cpu.to(data_type)
            hbc_time, _ = benchmark_hbc_function(
                fbgemm_hbc_cpu,
                curr_input,
            )

            hbc_by_feature_time, _ = benchmark_hbc_function(
                fbgemm_hbc_by_feature_cpu, curr_input
            )

            generic_hbc_by_feature_time, _ = benchmark_hbc_function(
                fbgemm_generic_hbc_by_feature_cpu, curr_input
            )
            if step >= warmup_runs:
                total_time["hbc"]["cpu"][data_type] += hbc_time
                total_time["hbc_by_feature"]["cpu"][data_type] += hbc_by_feature_time
                total_time["generic_hbc_by_feature"]["cpu"][
                    data_type
                ] += generic_hbc_by_feature_time

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

            segment_values_gpu: Tensor = segment_values.cuda()
            segment_lengths_gpu: Tensor = segment_lengths.cuda()

            by_feature_bin_num_examples_gpu: Tensor = by_feature_bin_num_examples.cuda()
            by_feature_bin_num_positives_gpu: Tensor = (
                by_feature_bin_num_positives.cuda()
            )

            def fbgemm_hbc_by_feature_gpu(input: Tensor) -> Tuple[Tensor, Tensor]:
                return torch.ops.fbgemm.histogram_binning_calibration_by_feature(
                    input,
                    segment_values_gpu,
                    segment_lengths_gpu,
                    num_segments,
                    by_feature_bin_num_examples_gpu,
                    by_feature_bin_num_positives_gpu,
                    num_bins,
                    0.4,
                    lower_bound,
                    upper_bound,
                    0,
                    0.9995,
                )

            bin_boundaries_gpu: Tensor = bin_boundaries.cuda()

            def fbgemm_generic_hbc_by_feature_gpu(
                input: Tensor,
            ) -> Tuple[Tensor, Tensor]:
                return (
                    torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
                        input,
                        segment_values_gpu,
                        segment_lengths_gpu,
                        num_segments,
                        by_feature_bin_num_examples_gpu,
                        by_feature_bin_num_positives_gpu,
                        bin_boundaries_gpu,
                        0.4,
                        0,
                        0.9995,
                    )
                )

            for data_type in data_types:
                curr_input_gpu = input_data_cpu.cuda().to(data_type)
                hbc_time, _ = benchmark_hbc_function(
                    fbgemm_hbc_gpu,
                    curr_input_gpu,
                )

                hbc_by_feature_time, _ = benchmark_hbc_function(
                    fbgemm_hbc_by_feature_gpu,
                    curr_input_gpu,
                )

                generic_hbc_by_feature_time, _ = benchmark_hbc_function(
                    fbgemm_generic_hbc_by_feature_gpu,
                    curr_input_gpu,
                )
                if step >= warmup_runs:
                    total_time["hbc"]["gpu"][data_type] += hbc_time
                    total_time["hbc_by_feature"]["gpu"][
                        data_type
                    ] += hbc_by_feature_time
                    total_time["generic_hbc_by_feature"]["gpu"][
                        data_type
                    ] += generic_hbc_by_feature_time

    for op, curr_items in total_time.items():
        for platform, data_items in curr_items.items():
            for dtype, t_time in data_items.items():
                logging.info(
                    f"{op}_{platform}_{dtype} time per iter: {t_time / iters * 1.0e6:.0f}us"
                )


if __name__ == "__main__":
    main()
