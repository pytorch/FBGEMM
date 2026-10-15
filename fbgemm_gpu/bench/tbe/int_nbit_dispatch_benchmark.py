#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode

Lookup = Callable[[], torch.Tensor]


def _summary(samples: list[float]) -> dict[str, float]:
    mean = statistics.mean(samples)
    sem = statistics.stdev(samples) / math.sqrt(len(samples))
    return {
        "mean_us": mean,
        "median_us": statistics.median(samples),
        "ci95_us": 1.96 * sem,
    }


def _make_lookup(
    embedding: IntNBitTableBatchedEmbeddingBagsCodegen,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache_weights: torch.Tensor | None,
    cache_locations: torch.Tensor | None,
) -> Lookup:
    def lookup() -> torch.Tensor:
        return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
            dev_weights=embedding.weights_dev,
            uvm_weights=embedding.weights_uvm,
            weights_placements=embedding.weights_placements,
            weights_offsets=embedding.weights_offsets,
            weights_tys=embedding.weights_tys,
            D_offsets=embedding.D_offsets,
            total_D=embedding.total_D,
            max_int2_D=embedding.max_int2_D,
            max_int4_D=embedding.max_int4_D,
            max_int8_D=embedding.max_int8_D,
            max_float16_D=embedding.max_float16_D,
            max_float32_D=embedding.max_float32_D,
            indices=indices,
            offsets=offsets,
            pooling_mode=int(PoolingMode.NONE),
            indice_weights=None,
            output_dtype=embedding.output_dtype,
            lxu_cache_weights=cache_weights,
            lxu_cache_locations=cache_locations,
            row_alignment=embedding.row_alignment,
            max_float8_D=embedding.max_float8_D,
            fp8_exponent_bits=embedding.fp8_exponent_bits,
            fp8_exponent_bias=embedding.fp8_exponent_bias,
        )

    return lookup


def _profile(lookup: Lookup, iterations: int) -> dict[str, Any]:
    for _ in range(25):
        lookup()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,  # pyre-ignore[16]
            torch.profiler.ProfilerActivity.CUDA,  # pyre-ignore[16]
        ]
    ) as prof:
        for _ in range(iterations):
            lookup()
        torch.cuda.synchronize()

    operator_events = [
        event
        for event in prof.key_averages()
        if event.key == "fbgemm::int_nbit_split_embedding_codegen_lookup_function"
    ]
    if len(operator_events) != 1:
        raise RuntimeError(f"Expected one lookup event, got {len(operator_events)}")
    operator_event = operator_events[0]
    empty_count = sum(
        event.count for event in prof.key_averages() if event.key == "aten::empty"
    )
    kernel_events = [
        event
        for event in prof.key_averages()
        if "nbit::" in event.key and "codegen_forward" in event.key
    ]
    return {
        "iterations": iterations,
        "operator_cpu_total_us": operator_event.cpu_time_total / iterations,
        "operator_self_cpu_us": operator_event.self_cpu_time_total / iterations,
        "aten_empty_count": empty_count,
        "aten_empty_per_invocation": empty_count / iterations,
        "gpu_kernel_count": sum(event.count for event in kernel_events),
        "gpu_kernels": {event.key: event.count for event in kernel_events},
        "gpu_kernel_time_us": sum(event.device_time for event in kernel_events),
    }


def _run_concurrent_round(
    executor: ThreadPoolExecutor,
    streams: list[torch.cuda.Stream],
    lookup: Lookup,
    iterations: int,
) -> tuple[float, list[float]]:
    barrier = threading.Barrier(len(streams))

    def worker(stream: torch.cuda.Stream) -> tuple[int, int, list[float]]:
        torch.cuda.set_device(stream.device)
        durations = []
        with torch.cuda.stream(stream):
            barrier.wait()
            batch_start = time.perf_counter_ns()
            for _ in range(iterations):
                start = time.perf_counter_ns()
                lookup()
                durations.append((time.perf_counter_ns() - start) / 1_000.0)
            batch_end = time.perf_counter_ns()
        return batch_start, batch_end, durations

    results = list(executor.map(worker, streams))
    torch.cuda.synchronize()
    elapsed_us = (
        max(result[1] for result in results) - min(result[0] for result in results)
    ) / 1_000.0
    per_call_us = elapsed_us / (len(streams) * iterations)
    durations = [duration for result in results for duration in result[2]]
    return per_call_us, durations


def _benchmark_concurrency_pair(
    lookups: dict[str, Lookup],
    thread_count: int,
    iterations: int,
    rounds: int,
) -> dict[str, dict[str, Any]]:
    streams = [torch.cuda.Stream() for _ in range(thread_count)]
    wall_samples: dict[str, list[float]] = {name: [] for name in lookups}
    latency_samples: dict[str, list[float]] = {name: [] for name in lookups}
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        for lookup in lookups.values():
            _run_concurrent_round(executor, streams, lookup, iterations=5)
        for round_index in range(rounds):
            names = list(lookups)
            if round_index % 2:
                names.reverse()
            for name in names:
                wall_us, durations = _run_concurrent_round(
                    executor, streams, lookups[name], iterations
                )
                wall_samples[name].append(wall_us)
                latency_samples[name].extend(durations)

    return {
        name: {
            "threads": thread_count,
            "calls": thread_count * iterations * rounds,
            "wall_per_invocation": _summary(wall_samples[name]),
            "dispatch_latency": _summary(latency_samples[name]),
        }
        for name in lookups
    }


def cli() -> None:
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise RuntimeError("An accelerator is required")
    device = torch.device(accelerator.type, torch.cuda.current_device())
    num_indices = 24_576
    num_bags = 24
    embedding = IntNBitTableBatchedEmbeddingBagsCodegen(
        [("", 100_000, 256, SparseType.INT8, EmbeddingLocation.DEVICE)],
        device=device,
        output_dtype=SparseType.FP16,
        pooling_mode=PoolingMode.NONE,
    )
    embedding.fill_random_weights()

    # Citrine C3: construct benchmark inputs directly on the accelerator.
    indices = torch.arange(num_indices, dtype=torch.int32, device=device) % 100_000
    offsets = torch.arange(
        0,
        num_indices + 1,
        num_indices // num_bags,
        dtype=torch.int32,
        device=device,
    )
    cache_weights = torch.empty((0, 0), dtype=torch.uint8, device=device)
    cache_locations = torch.empty((0,), dtype=torch.int32, device=device)

    lookups = {
        "supplied": _make_lookup(
            embedding, indices, offsets, cache_weights, cache_locations
        ),
        "nullopt": _make_lookup(embedding, indices, offsets, None, None),
    }
    supplied_output = lookups["supplied"]()
    nullopt_output = lookups["nullopt"]()
    torch.testing.assert_close(supplied_output, nullopt_output)
    if supplied_output.dtype != torch.float16 or supplied_output.device != device:
        raise RuntimeError(
            f"Unexpected output: {supplied_output.dtype} on {supplied_output.device}"
        )

    results: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(device),
        "shape": {
            "weights": "INT8",
            "embedding_dim": 256,
            "output": "FP16",
            "indices": num_indices,
            "offsets": offsets.numel(),
        },
        "correctness": True,
        "variants": {},
    }
    profiles = {
        name: _profile(lookup, iterations=1_000) for name, lookup in lookups.items()
    }
    concurrency_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in lookups
    }
    for thread_count in (1, 16, 32):
        results_for_thread_count = _benchmark_concurrency_pair(
            lookups, thread_count, iterations=20, rounds=50
        )
        for name, result in results_for_thread_count.items():
            concurrency_results[name].append(result)

    for name in lookups:
        for concurrency_result in concurrency_results[name]:
            concurrency_result["expected_aten_empty_calls"] = int(
                concurrency_result["calls"]
                * profiles[name]["aten_empty_per_invocation"]
            )
        results["variants"][name] = {
            "profile": profiles[name],
            "concurrency": concurrency_results[name],
        }

    print(json.dumps(results, indent=2, sort_keys=True))
