# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import cast, List, Tuple

import fbgemm_gpu
import torch
from fbgemm_gpu.runtime_monitor import (
    AsyncSeriesTimer,
    StdLogStatsReporter,
    StdLogStatsReporterConfig,
    TBEStatsReporterConfig,
)

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class TesteeAsyncSeriesTimer(AsyncSeriesTimer):
    outputs: List[Tuple[str, float]]

    def __init__(self) -> None:
        self.outputs = []

        def report_callback(ctx: str, duration: float) -> None:
            self.outputs.append((ctx, duration))

        super().__init__(report_callback)


class RuntimeMonitorTest(unittest.TestCase):
    def assert_context(
        self, timer: TesteeAsyncSeriesTimer, context_list: List[str]
    ) -> None:
        timer._lazy_report()
        self.assertEqual([t[0] for t in timer.outputs], context_list)

    def expensive_work(self, iterations: int) -> torch.Tensor:
        t = torch.rand((2000, 2000), dtype=torch.float32, device="cuda")
        for _ in range(iterations):
            t2 = torch.rand((2000, 2000), dtype=torch.float32, device="cuda")
            t = torch.matmul(t, t2)
        return t

    # pyre-ignore
    @unittest.skipIf(*gpu_unavailable)
    def test_async_series_timer_multi_start(self) -> None:
        """
        Test the timer must fail if tried to start recording while already started
        """
        timer = TesteeAsyncSeriesTimer()
        timer.start()
        with self.assertRaises(AssertionError):
            timer.start()
        self.assertEqual(timer.outputs, [])

    # pyre-ignore
    @unittest.skipIf(*gpu_unavailable)
    def test_async_series_timer_multi_stop(self) -> None:
        """
        Test the timer must fail if tried to stop recording while not started
        """
        timer = TesteeAsyncSeriesTimer()
        with self.assertRaises(AssertionError):
            timer.stop("a")
        timer.start()
        timer.stop("b")
        with self.assertRaises(AssertionError):
            timer.stop("c")
        torch.cuda.synchronize()
        self.assert_context(timer, ["b"])

    # pyre-ignore
    @unittest.skipIf(*gpu_unavailable)
    def test_async_series_timer_recording(self) -> None:
        """
        Test the timer is timing properly and returning a nonzero duration for
        a relatively expensive kernel
        """
        timer = TesteeAsyncSeriesTimer()
        with timer.recording(context="alloc"):
            t = self.expensive_work(1)
        t.tolist()
        torch.cuda.synchronize()
        self.assert_context(timer, ["alloc"])
        self.assertGreaterEqual(timer.outputs[0][1], 0)

    # pyre-ignore
    @unittest.skipIf(*gpu_unavailable)
    def test_async_series_timer_recording_other_stream(self) -> None:
        """
        Test the timer is timing properly on customized stream
        """
        timer = TesteeAsyncSeriesTimer()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            with timer.recording(context="alloc"):
                t = self.expensive_work(1)
        t.tolist()
        torch.cuda.synchronize()
        self.assert_context(timer, ["alloc"])
        self.assertGreaterEqual(timer.outputs[0][1], 0)

    # pyre-ignore
    @unittest.skipIf(*gpu_unavailable)
    def test_async_series_timer_recording_multi_stream_load(self) -> None:
        """
        Test the timer is timing properly for events across multiple streams, including
        explicit stream synchronization
        """
        timer = TesteeAsyncSeriesTimer()

        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            with timer.recording(context="alloc"):
                t = self.expensive_work(1)
            with timer.recording(context="compute"):
                t2 = self.expensive_work(5)

        with timer.recording(context="wait"):
            torch.cuda.current_stream().wait_stream(s)
        t.tolist()
        t2.tolist()
        torch.cuda.synchronize()

        self.assert_context(timer, ["alloc", "compute", "wait"])
        self.assertGreaterEqual(timer.outputs[0][1], 0)
        self.assertGreaterEqual(timer.outputs[1][1], 0)
        self.assertGreaterEqual(timer.outputs[2][1], 0)

    def test_noop_reporter(self) -> None:
        """
        Test the base reporter config can only create noop-reporter (None) or
        fail if interval is positive (which requires a real reporting)
        """
        # This config can create a None reporter, because interval is non-positive
        config = TBEStatsReporterConfig()
        self.assertIsNone(config.create_reporter())

        # This config cannot, because it provides a positive interval without
        # giving any actual implementation
        config = TBEStatsReporterConfig(interval=100)
        with self.assertRaises(AssertionError):
            config.create_reporter()

    def test_stdlog_reporter(self) -> None:
        """
        Test std log reporter be created with different config and log as expected
        """
        config = StdLogStatsReporterConfig(interval=0)
        self.assertIsNone(config.create_reporter())

        config = StdLogStatsReporterConfig(interval=100)
        reporter = config.create_reporter()
        self.assertIsInstance(reporter, StdLogStatsReporter)
        r = cast(StdLogStatsReporter, reporter)
        self.assertTrue(r.should_report(500))
        self.assertFalse(r.should_report(101))
        r.report_duration(iteration_step=500, event_name="test_event", duration_ms=404)
        r.report_data_amount(iteration_step=500, event_name="event2", data_bytes=4096)
