#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import json
from enum import Enum
from typing import Any, Optional

import click


@dataclasses.dataclass(frozen=True)
class TBEBenchmarkingConfig:
    # Number of iterations
    iterations: int
    # Number of input TBE batches to generate for testing
    num_requests: int
    # Number of warmup iterations to run before making measurements
    warmup_iterations: int
    # Amount of memory to use for flushing the GPU cache after each iteration
    flush_gpu_cache_size_mb: int
    # If set, trace will be exported to the path specified in trace_url
    export_trace: bool
    # The path for exporting the trace
    trace_url: Optional[str]
    # If set and export_trace is true, the benchmark will upload performance data from the trace to Scuba
    upload_perf_data: bool

    @classmethod
    # pyre-ignore [3]
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)

    @classmethod
    # pyre-ignore [3]
    def from_json(cls, data: str):
        return cls.from_dict(json.loads(data))

    def dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def json(self, format: bool = False) -> str:
        return json.dumps(self.dict(), indent=(2 if format else -1), sort_keys=True)

    # pyre-ignore [3]
    def validate(self):
        assert self.iterations > 0, "iterations must be positive"
        assert self.num_requests > 0, "num_requests must be positive"
        assert self.warmup_iterations >= 0, "warmup_iterations must be non-negative"
        assert (
            self.flush_gpu_cache_size_mb >= 0
        ), "flush_gpu_cache_size_mb must be non-negative"
        return self


@dataclasses.dataclass(frozen=True)
class TBEBenchmarkingHelperText(Enum):
    BENCH_ITERATIONS = "Number of benchmark iterations to run"
    BENCH_NUM_REQUESTS = "Number of input batches to generate. If the value is smaller than the number of benchmark iterations, input batches will be re-used"
    BENCH_WARMUP_ITERATIONS = (
        "Number of warmup iterations to run before making measurements"
    )
    BENCH_FLUSH_GPU_CACHE_SIZE = (
        "Amount of memory to use for flushing the GPU cache after each iteration (MB)"
    )
    BENCH_EXPORT_TRACE = (
        "If set, trace will be exported to the path specified in trace url"
    )
    BENCH_TRACE_URL = "The path for exporting the trace"
    BENCH_UPLOAD_PERF_DATA = "If set and export_trace is true, the benchmark will upload performance data from the trace to Scuba"


class TBEBenchmarkingConfigLoader:
    @classmethod
    # pyre-ignore [2]
    def options(cls, func) -> click.Command:
        options = [
            click.option(
                "--bench-iterations",
                type=int,
                default=100,
                help=TBEBenchmarkingHelperText.BENCH_ITERATIONS.value,
            ),
            click.option(
                "--bench-num-requests",
                type=int,
                default=-1,
                help=TBEBenchmarkingHelperText.BENCH_NUM_REQUESTS.value,
            ),
            click.option(
                "--bench-warmup-iterations",
                type=int,
                default=0,
                help=TBEBenchmarkingHelperText.BENCH_WARMUP_ITERATIONS.value,
            ),
            click.option(
                "--bench-flush-gpu-cache-size",
                type=int,
                default=0,
                help=TBEBenchmarkingHelperText.BENCH_FLUSH_GPU_CACHE_SIZE.value,
            ),
            click.option(
                "--bench-export-trace",
                is_flag=True,
                default=False,
                help=TBEBenchmarkingHelperText.BENCH_EXPORT_TRACE.value,
            ),
            click.option(
                "--bench-trace-url",
                type=str,
                required=False,
                default="{emb_op_type}_tbe_{phase}_trace_{ospid}.json",
                help=TBEBenchmarkingHelperText.BENCH_TRACE_URL.value,
            ),
            click.option(
                "--upload-perf-data",
                is_flag=True,
                default=False,
                help=TBEBenchmarkingHelperText.BENCH_UPLOAD_PERF_DATA.value,
            ),
        ]

        for option in reversed(options):
            func = option(func)
        return func

    @classmethod
    def load(cls, context: click.Context) -> TBEBenchmarkingConfig:
        params = context.params

        iterations = params["bench_iterations"]
        num_requests = params["bench_num_requests"]
        warmup_iterations = params["bench_warmup_iterations"]
        flush_gpu_cache_size = params["bench_flush_gpu_cache_size"]
        export_trace = params["bench_export_trace"]
        trace_url = params["bench_trace_url"]
        upload_perf_data = params["upload_perf_data"]

        # Default the number of TBE requests to number of iterations specified
        num_requests = iterations if num_requests == -1 else num_requests

        return TBEBenchmarkingConfig(
            iterations,
            num_requests,
            warmup_iterations,
            flush_gpu_cache_size,
            export_trace,
            trace_url,
            upload_perf_data,
        ).validate()
