#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Formatting and table-printing utilities for kernel statistics."""

from __future__ import annotations

import math


def fmt(val: float, precision: int = 2) -> str:
    """NaN-safe float formatting."""
    if math.isnan(val):
        return "N/A"
    return f"{val:.{precision}f}"


def pct(old_val: float, new_val: float) -> str:
    """Percentage diff string with sign."""
    if (
        old_val > 0
        and new_val > 0
        and not math.isnan(old_val)
        and not math.isnan(new_val)
    ):
        return f"{((new_val - old_val) / old_val) * 100:+.1f}%"
    return "N/A"


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    right_align_from: int = 1,
    group_by_col: int | None = None,
    footer_lines: list[str] | None = None,
) -> None:
    """
    Print an aligned table with | separators.

    Args:
        title: Title printed above the table
        headers: Column header names
        rows: List of rows (each row is a list of string values)
        right_align_from: Column index from which to right-align (default: 1)
        group_by_col: If set, insert a separator line when this column's value changes
        footer_lines: Optional footer lines printed below the table
    """
    if not rows:
        print(f"\n{title}: No data.")
        return

    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    total = sum(widths) + (len(widths) - 1) * 3

    print(f"\n{'=' * total}")
    print(title)
    print("=" * total)

    hdr = " | ".join(
        f"{h:>{w}}" if i >= right_align_from else f"{h:<{w}}"
        for i, (h, w) in enumerate(zip(headers, widths))
    )
    print(hdr)
    print("-" * total)

    prev_group = None
    for row in rows:
        if group_by_col is not None:
            if row[group_by_col] != prev_group and prev_group is not None:
                print("-" * total)
            prev_group = row[group_by_col]

        line = " | ".join(
            (
                f"{row[i]:>{widths[i]}}"
                if i >= right_align_from
                else f"{row[i]:<{widths[i]}}"
            )
            for i in range(len(headers))
        )
        print(line)

    print("=" * total)

    if footer_lines:
        for line in footer_lines:
            print(line)
