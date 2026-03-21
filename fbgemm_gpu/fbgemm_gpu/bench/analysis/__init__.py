#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""
Shared library for kernel trace analysis.

Provides common functionality for parsing Chrome traces, extracting kernel
durations, computing statistics, formatting output tables, and exporting to CSV.
"""

from fbgemm_gpu.bench.analysis.comparison import (  # noqa: F401
    combined_summary,
    print_ratio_table,
)
from fbgemm_gpu.bench.analysis.formatting import (  # noqa: F401
    fmt,
    pct,
    print_table,
    shorten_kernel_name,
)
from fbgemm_gpu.bench.analysis.trace import KinetoTrace  # noqa: F401
from fbgemm_gpu.bench.analysis.types import KernelStats, StatsMap  # noqa: F401
