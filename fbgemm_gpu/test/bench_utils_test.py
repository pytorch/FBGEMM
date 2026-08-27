# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import math
import threading
import time
import unittest
from unittest.mock import patch

import torch
from fbgemm_gpu.bench.bench_utils import benchmark_torch_function, BenchmarkDiagnostics


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class BenchUtilsTest(unittest.TestCase):
    def _operands(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Citrine C3: allocate directly on device.
        a = torch.randn(512, 512, device=torch.accelerator.current_accelerator())
        b = torch.randn(512, 512, device=torch.accelerator.current_accelerator())
        return a, b

    def _assert_sane_per_iter(self, elapsed: float) -> None:
        # Returned value is seconds-per-iteration for both single- and
        # multi-stream paths (the contract callers rely on for batch/elapsed).
        self.assertTrue(math.isfinite(elapsed))
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 1.0)  # a 512x512 mm is well under 1s/iter

    def test_single_thread(self) -> None:
        a, b = self._operands()
        diagnostics: BenchmarkDiagnostics = {}
        elapsed, _ = benchmark_torch_function(
            _mm,
            (a, b),
            iters=20,
            num_warmups=5,
            device="cuda",
            num_threads=1,
            diagnostics=diagnostics,
        )
        self._assert_sane_per_iter(elapsed)
        self.assertEqual("single_stream_events", diagnostics["timing_method"])
        self.assertGreater(float(diagnostics["device_span_ms"]), 0.0)

    def test_multi_stream_wall_clock(self) -> None:
        a, b = self._operands()
        diagnostics: BenchmarkDiagnostics = {}
        call_count = 0
        observed_stream_handles: set[int] = set()
        call_lock = threading.Lock()

        def counted_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nonlocal call_count
            with call_lock:
                call_count += 1
                observed_stream_handles.add(torch.cuda.current_stream().cuda_stream)
            return torch.mm(a, b)

        elapsed, _ = benchmark_torch_function(
            counted_mm,
            (a, b),
            iters=40,
            num_warmups=5,
            device="cuda",
            num_threads=2,
            wall_clock_multi_stream_timing=True,
            diagnostics=diagnostics,
        )
        self._assert_sane_per_iter(elapsed)
        self.assertEqual("multi_stream_wall_clock", diagnostics["timing_method"])
        workers = diagnostics["workers"]
        self.assertEqual(2, len(workers))
        self.assertGreater(float(diagnostics["max_device_span_ms"]), 0.0)
        self.assertEqual(51, call_count)
        self.assertEqual(
            {int(worker["stream_handle"]) for worker in workers},
            observed_stream_handles,
        )

    def test_multi_stream_provided_streams(self) -> None:
        a, b = self._operands()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        expected_handles = {stream.cuda_stream for stream in streams}

        for _ in range(2):
            diagnostics: BenchmarkDiagnostics = {}
            elapsed, _ = benchmark_torch_function(
                _mm,
                (a, b),
                iters=40,
                num_warmups=5,
                device="cuda",
                num_threads=2,
                wall_clock_multi_stream_timing=True,
                diagnostics=diagnostics,
                streams=streams,
            )
            self._assert_sane_per_iter(elapsed)
            self.assertTrue(bool(diagnostics["uses_provided_streams"]))
            workers = diagnostics["workers"]
            self.assertEqual(
                expected_handles,
                {int(worker["stream_handle"]) for worker in workers},
            )

    def test_single_provided_stream_uses_wall_clock(self) -> None:
        a, b = self._operands()
        stream = torch.cuda.Stream()
        diagnostics: BenchmarkDiagnostics = {}
        elapsed, _ = benchmark_torch_function(
            _mm,
            (a, b),
            iters=20,
            num_warmups=5,
            device="cuda",
            num_threads=1,
            wall_clock_multi_stream_timing=True,
            diagnostics=diagnostics,
            streams=[stream],
        )
        self._assert_sane_per_iter(elapsed)
        self.assertEqual("multi_stream_wall_clock", diagnostics["timing_method"])
        workers = diagnostics["workers"]
        self.assertEqual(1, len(workers))
        self.assertEqual(stream.cuda_stream, int(workers[0]["stream_handle"]))

    def test_multi_stream_wall_clock_is_opt_in(self) -> None:
        a, b = self._operands()
        default_diagnostics: BenchmarkDiagnostics = {}
        elapsed_default, _ = benchmark_torch_function(
            _mm,
            (a, b),
            iters=40,
            num_warmups=5,
            device="cuda",
            num_threads=2,
            diagnostics=default_diagnostics,
        )
        wall_clock_diagnostics: BenchmarkDiagnostics = {}
        elapsed_wall_clock, _ = benchmark_torch_function(
            _mm,
            (a, b),
            iters=40,
            num_warmups=5,
            device="cuda",
            num_threads=2,
            wall_clock_multi_stream_timing=True,
            diagnostics=wall_clock_diagnostics,
        )
        self._assert_sane_per_iter(elapsed_default)
        self._assert_sane_per_iter(elapsed_wall_clock)
        self.assertEqual(
            "multi_stream_event_amortized", default_diagnostics["timing_method"]
        )
        self.assertEqual(
            "multi_stream_wall_clock", wall_clock_diagnostics["timing_method"]
        )

    def test_default_multi_stream_allows_uneven_iteration_partition(self) -> None:
        a, b = self._operands()
        elapsed, _ = benchmark_torch_function(
            _mm,
            (a, b),
            flush_gpu_cache_size_mb=0,
            iters=5,
            num_warmups=1,
            device="cuda",
            num_threads=2,
        )
        self._assert_sane_per_iter(elapsed)

    def test_wall_clock_requires_even_iteration_partition(self) -> None:
        a, b = self._operands()
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            benchmark_torch_function(
                _mm,
                (a, b),
                iters=5,
                num_warmups=1,
                device="cuda",
                num_threads=2,
                wall_clock_multi_stream_timing=True,
            )

    def test_wall_clock_requires_stream_warmup(self) -> None:
        a, b = self._operands()
        with self.assertRaisesRegex(ValueError, "requires num_warmups >= 1"):
            benchmark_torch_function(
                _mm,
                (a, b),
                iters=4,
                num_warmups=0,
                device="cuda",
                num_threads=2,
                wall_clock_multi_stream_timing=True,
            )

    def test_wall_clock_forwards_keyword_arguments(self) -> None:
        a, b = self._operands()

        def scaled_mm(
            a: torch.Tensor,
            b: torch.Tensor,
            *,
            scale: float,
        ) -> torch.Tensor:
            return torch.mm(a, b) * scale

        _, output = benchmark_torch_function(
            scaled_mm,
            (a, b),
            kwargs={"scale": 2.0},
            flush_gpu_cache_size_mb=0,
            iters=4,
            num_warmups=1,
            device="cuda",
            num_threads=2,
            wall_clock_multi_stream_timing=True,
        )
        self.assertTrue(torch.allclose(torch.mm(a, b) * 2.0, output))

    def test_single_stream_event_timing_preserves_legacy_kwargs_behavior(
        self,
    ) -> None:
        a, b = self._operands()

        def scaled_mm(
            a: torch.Tensor,
            b: torch.Tensor,
            *,
            scale: float = 1.0,
        ) -> torch.Tensor:
            return torch.mm(a, b) * scale

        _, output = benchmark_torch_function(
            scaled_mm,
            (a, b),
            kwargs={"scale": 2.0},
            flush_gpu_cache_size_mb=0,
            iters=2,
            num_warmups=1,
            device="cuda",
        )
        self.assertTrue(torch.allclose(torch.mm(a, b), output))

    def test_multi_stream_event_timing_preserves_legacy_kwargs_behavior(
        self,
    ) -> None:
        a, b = self._operands()
        observed_scales: list[float] = []
        call_lock = threading.Lock()

        def scaled_mm(
            a: torch.Tensor,
            b: torch.Tensor,
            *,
            scale: float = 1.0,
        ) -> torch.Tensor:
            with call_lock:
                observed_scales.append(scale)
            return torch.mm(a, b) * scale

        benchmark_torch_function(
            scaled_mm,
            (a, b),
            kwargs={"scale": 2.0},
            flush_gpu_cache_size_mb=0,
            iters=4,
            num_warmups=1,
            device="cuda",
            num_threads=2,
        )
        self.assertCountEqual([2.0, 1.0, 1.0, 1.0, 1.0], observed_scales)

    def test_wall_clock_waits_for_caller_stream(self) -> None:
        device = torch.accelerator.current_accelerator()
        # Citrine C3: allocate test inputs directly on device.
        a = torch.zeros(32, 32, device=device)
        b = torch.ones(32, 32, device=device)
        torch.cuda._sleep(100_000_000)
        a.fill_(2.0)

        _, output = benchmark_torch_function(
            _mm,
            (a, b),
            flush_gpu_cache_size_mb=0,
            iters=2,
            num_warmups=1,
            device="cuda",
            num_threads=1,
            wall_clock_multi_stream_timing=True,
            streams=[torch.cuda.Stream()],
        )
        expected = torch.full((32, 32), 64.0, device=device)
        self.assertTrue(torch.equal(expected, output))

    def test_wall_clock_records_start_before_releasing_workers(self) -> None:
        a, b = self._operands()
        timestamp_recorded = threading.Event()
        work_started_before_timestamp = threading.Event()
        call_count = 0
        clock_call_count = 0
        call_lock = threading.Lock()
        num_threads = 2
        num_warmups = 1
        expected_warmup_calls = 1 + num_threads * num_warmups

        def ordered_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nonlocal call_count
            with call_lock:
                call_count += 1
                is_measured_call = call_count > expected_warmup_calls
            if is_measured_call and not timestamp_recorded.is_set():
                work_started_before_timestamp.set()
            return torch.mm(a, b)

        def delayed_clock() -> int:
            nonlocal clock_call_count
            clock_call_count += 1
            if clock_call_count == 1:
                time.sleep(0.05)
                timestamp_recorded.set()
                return 1_000_000_000
            return 2_000_000_000

        with patch(
            "fbgemm_gpu.bench.bench_utils.time.perf_counter_ns",
            side_effect=delayed_clock,
        ):
            benchmark_torch_function(
                ordered_mm,
                (a, b),
                flush_gpu_cache_size_mb=0,
                iters=2,
                num_warmups=num_warmups,
                device="cuda",
                num_threads=num_threads,
                wall_clock_multi_stream_timing=True,
            )
        self.assertFalse(work_started_before_timestamp.is_set())

    def test_multi_stream_wall_clock_propagates_worker_error(self) -> None:
        a, b = self._operands()
        call_count = 0
        call_lock = threading.Lock()

        def fail_after_priming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nonlocal call_count
            with call_lock:
                call_count += 1
                should_fail = call_count == 2
            if should_fail:
                raise RuntimeError("worker failed")
            return torch.mm(a, b)

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            benchmark_torch_function(
                fail_after_priming,
                (a, b),
                iters=4,
                num_warmups=1,
                device="cuda",
                num_threads=2,
                wall_clock_multi_stream_timing=True,
            )
