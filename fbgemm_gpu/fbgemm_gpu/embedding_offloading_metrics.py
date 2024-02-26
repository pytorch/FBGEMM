#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc


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
