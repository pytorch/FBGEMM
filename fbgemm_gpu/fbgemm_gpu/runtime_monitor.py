#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc

from dataclasses import dataclass
from typing import Optional


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
