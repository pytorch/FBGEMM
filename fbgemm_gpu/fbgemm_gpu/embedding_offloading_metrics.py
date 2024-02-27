#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import deque
from typing import Callable, Deque, Optional, Tuple, TypeVar

import torch


class IEmbeddingOffloadingMetricsReporter(abc.ABC):
    """
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


T = TypeVar("T")


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
        self._events_queue: Deque[
            Tuple[torch.cuda.Event, torch.cuda.Event, T]
        ] = deque()
        self._active_start_event: Optional[torch.cuda.Event] = None
        self._report_functor = report_functor

    def start(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        assert self._active_start_event is None, "There's an active recording"
        self._active_start_event = torch.cuda.Event(enable_timing=True)
        self._active_start_event.record(stream)
        self._lazy_report()

    def stop(self, context: T, stream: Optional[torch.cuda.Stream] = None) -> None:
        assert self._active_start_event is not None, "There's an active recording"
        active_start_event: torch.cuda.Event = self._active_start_event
        self._active_start_event = None

        active_stop_event = torch.cuda.Event(enable_timing=True)
        active_stop_event.record(stream)
        self._events_queue.append((active_start_event, active_stop_event, context))
        self._lazy_report()

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
