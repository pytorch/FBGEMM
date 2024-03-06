#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import logging
from collections import deque
from dataclasses import dataclass
from types import TracebackType
from typing import Callable, Deque, Optional, Tuple, Type, TypeVar

import torch


class TBEStatsReporter(abc.ABC):
    """
    Interface for TBE runtime stats reporting. Actual implementation may do
    custome aggregation (on intended group-key) and reporting destination.

    All the report_XXX functions should be light weighted and fail-safe.
    """

    @abc.abstractmethod
    def should_report(self, iteration_step: int) -> bool:
        """
        Return whether we should report metrics during this step.
        This function should be cheap, side-effect free and return immediately.
        """
        ...

    @abc.abstractmethod
    def report_duration(
        self,
        iteration_step: int,
        event_name: str,
        duration_ms: float,
        embedding_id: str = "",
        tbe_id: str = "",
    ) -> None:
        """
        Report the duration of a timed event.
        """
        ...


class StdLogStatsReporter(TBEStatsReporter):
    def __init__(self, report_interval: int) -> None:
        assert report_interval > 0, "Report interval must be positive"
        self.report_interval = report_interval

    def should_report(self, iteration_step: int) -> bool:
        return iteration_step % self.report_interval == 0

    def report_duration(
        self,
        iteration_step: int,
        event_name: str,
        duration_ms: float,
        embedding_id: str = "",
        tbe_id: str = "",
    ) -> None:
        logging.info(
            f"[Batch #{iteration_step}][TBE:{tbe_id}][Table:{embedding_id}] The event {event_name} took {duration_ms} ms"
        )


@dataclass
class TBEStatsReporterConfig:
    """
    Configuration for TBEStatsReporter. It eventually instantiates the actual
    reporter, so it can be deep-copied without incurring the actual reporter
    getting copied.
    """

    # Collect required batches every given batches. Non-positive stands for
    # no collection or reporting
    interval: int = -1

    def create_reporter(self) -> Optional[TBEStatsReporter]:
        assert (
            self.interval <= 0
        ), "Cannot specify interval without an actual implementation of reporter"
        return None


@dataclass
class StdLogStatsReporterConfig(TBEStatsReporterConfig):
    def create_reporter(self) -> Optional[TBEStatsReporter]:
        if self.interval <= 0:
            return None
        return StdLogStatsReporter(report_interval=self.interval)


T = TypeVar("T")


class AsyncSeriesTimerRecordedContext:
    """
    An easier way to use AsyncSeriesTimer. Example:
    ```
    timer : AsyncSeriesTimer = ...
    with timer.recording(ctx):
        cuda_kernel1()
        cuda_kernel2()
        cuda_kernel3()
    ```
    """

    def __init__(
        self,
        timer: "AsyncSeriesTimer",
        context: T,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self._context = context
        self._stream = stream
        self._timer = timer

    def __enter__(self) -> None:
        self._timer.start(self._stream)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._timer.stop(self._context, self._stream)


class AsyncSeriesTimer:
    """
    A wrapper class on top of torch.cuda.Event to measure the time between a
    series of CUDA events. Once initiated, every start() and stop() call pair
    will measure the timing between them on GPU. Caller cannot inititate another
    recording if there's already one ongoing.

    Reporting is asynchronous as the timing result is not ready immediately at
    stop(). Instead, we do it in a lazy way -- we check the all unreported
    events at every start or stop call.
    """

    def __init__(self, report_functor: Callable[[T, float], None]) -> None:
        self._events_queue: Deque[Tuple[torch.cuda.Event, torch.cuda.Event, T]] = (
            deque()
        )
        self._active_start_event: Optional[torch.cuda.Event] = None
        self._report_functor = report_functor

    def start(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        assert self._active_start_event is None, "There's an active recording"
        self._active_start_event = torch.cuda.Event(enable_timing=True)
        self._active_start_event.record(stream)
        self._lazy_report()

    def stop(self, context: T, stream: Optional[torch.cuda.Stream] = None) -> None:
        assert self._active_start_event is not None, "There's no active recording"
        active_start_event: torch.cuda.Event = self._active_start_event

        active_stop_event = torch.cuda.Event(enable_timing=True)
        active_stop_event.record(stream)
        self._events_queue.append((active_start_event, active_stop_event, context))
        self._active_start_event = None
        self._lazy_report()

    def recording(
        self, context: T, stream: Optional[torch.cuda.Stream] = None
    ) -> AsyncSeriesTimerRecordedContext:
        return AsyncSeriesTimerRecordedContext(self, context, stream)

    def _lazy_report(self) -> None:
        # Since this is a series of timing event, the earlies recorded event
        # finishes earliest. So we only need to check the leftmost stop event
        # to decide if we need to report now.

        while len(self._events_queue):
            stop_event = self._events_queue[0][1]
            if not stop_event.query():
                # Even the earliest event hasn't completed in GPU. Don't do
                # report.
                return
            start_event, stop_event, context = self._events_queue.popleft()
            assert (
                start_event.query()
            ), "Recording has start event later than stop event"
            result = float(start_event.elapsed_time(stop_event))
            self._report_functor(context, result)
