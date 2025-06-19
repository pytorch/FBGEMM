#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
from dataclasses import dataclass

haveAIBench = False
try:
    from aibench_observer.utils.observer import emitMetric

    haveAIBench = True
except Exception:
    haveAIBench = False


@dataclass
class BenchmarkReporter:
    report: bool
    logger: logging.Logger = logging.getLogger()

    # pyre-ignore[3]
    def __post_init__(self):
        self.logger.setLevel(logging.INFO)

    # pyre-ignore[2]
    def emit_metric(self, **kwargs) -> None:
        if self.report and haveAIBench:
            self.logger.info(emitMetric(**kwargs))
