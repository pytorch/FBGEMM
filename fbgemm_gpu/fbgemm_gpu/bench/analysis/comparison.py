#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Side-by-side comparison and ratio tables for multi-source trace analysis."""

from __future__ import annotations

import glob
import os
import re
import statistics
from collections.abc import Callable
from typing import Any

from fbgemm_gpu.bench.analysis.formatting import shorten_kernel_name
from fbgemm_gpu.bench.analysis.trace import KinetoTrace


def combined_summary(
    sources: list[tuple[str, str, str]],
    parse_config_fn: Callable[[str], Any | None] | None = None,
    config_key_fn: Callable[[Any], str] | None = None,
    verbose: bool = False,
) -> None:
    """Print a combined table across multiple trace sources.

    Each *source* is ``(label, trace_dir, kernel_pattern)``.  For every
    source the function scans the trace directory and groups durations by
    (exact kernel name, config_key).  Each distinct kernel gets its own
    column; rows are config keys.

    If *parse_config_fn* and *config_key_fn* are not provided, the default
    behaviour groups by ``(output_dtype, managed)`` — the most common
    pattern for TBE benchmarks.

    Args:
        sources: List of ``(label, trace_dir, kernel_pattern)`` triples.
        parse_config_fn: ``(filename) -> config | None``.  When ``None``,
            a simple default extracts output_dtype and managed from the
            filename using common TBE naming conventions.
        config_key_fn: ``(config) -> str`` row key.  Ignored when
            *parse_config_fn* is ``None``.
        verbose: Print warnings for skipped files.
    """
    if parse_config_fn is None:
        parse_config_fn = _default_parse_config
        config_key_fn = _default_config_key

    col_data: dict[str, dict[str, list[float]]] = {}
    col_full_names: dict[str, set[str]] = {}
    col_keys: list[str] = []

    for label, trace_dir, kernel_pattern in sources:
        kernel_configs: dict[str, dict[str, list[float]]] = {}

        for trace_file in sorted(glob.glob(os.path.join(trace_dir, "*.json"))):
            filename = os.path.basename(trace_file)
            config = parse_config_fn(filename)
            if not config:
                continue

            row_key = config_key_fn(config) if config_key_fn else str(config)
            if not row_key:
                continue

            try:
                trace = KinetoTrace.from_file(trace_file)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to parse {trace_file}: {e}")
                continue

            by_name = trace.extract_durations(kernel_pattern)
            for kname, durs in by_name.items():
                kernel_configs.setdefault(kname, {}).setdefault(row_key, []).extend(
                    durs
                )

        unique_kernels = sorted(kernel_configs.keys())
        by_short: dict[str, list[str]] = {}
        for kname in unique_kernels:
            short = shorten_kernel_name(kname)
            by_short.setdefault(short, []).append(kname)

        for short, knames in sorted(by_short.items()):
            if len(knames) == 1:
                kname = knames[0]
                col_key = f"{label}: {short}"
                col_data[col_key] = kernel_configs[kname]
                col_keys.append(col_key)
                col_full_names[col_key] = {kname}
            else:
                for kname in knames:
                    tags = sorted(
                        {cfg.split("/")[0] for cfg in kernel_configs[kname].keys()}
                    )
                    tag = "/".join(tags)
                    col_key = f"{label}: {short} ({tag})"
                    col_data[col_key] = kernel_configs[kname]
                    col_keys.append(col_key)
                    col_full_names[col_key] = {kname}

    if not col_keys:
        print("No matching kernel data found for any source.")
        return

    all_row_keys: set[str] = set()
    for cfg_map in col_data.values():
        all_row_keys.update(cfg_map.keys())
    sorted_rows = sorted(all_row_keys)

    col_width = max(14, *(len(k) for k in col_keys)) + 2
    row_label_width = max(20, *(len(r) for r in sorted_rows)) + 2

    total_width = row_label_width + len(col_keys) * col_width + 4
    print("=" * total_width)
    print("Combined Kernel Summary")
    print("=" * total_width)
    print()

    header = f"{'Config':<{row_label_width}}"
    for ck in col_keys:
        header += f"  {ck:>{col_width - 2}}"
    print(header)
    sep = "-" * total_width
    print(sep)

    for row_key in sorted_rows:
        line = f"{row_key:<{row_label_width}}"
        for ck in col_keys:
            durs = col_data[ck].get(row_key)
            if durs:
                mean_val = statistics.mean(durs)
                cell = f"{mean_val:,.1f} µs"
            else:
                cell = "—"
            line += f"  {cell:>{col_width - 2}}"
        print(line)

    print(sep)
    print()

    print("Kernel mapping:")
    print("-" * 80)
    for ck in col_keys:
        names = sorted(col_full_names.get(ck, set()))
        if len(names) == 1:
            print(f"  {ck}: {names[0]}")
        else:
            print(f"  {ck}:")
            for n in names:
                print(f"    - {n}")
    print()


def print_ratio_table(
    stats_a: dict[str, list[float]],
    stats_b: dict[str, list[float]],
    label_a: str,
    label_b: str,
    thresholds: tuple[float, float] = (0.9, 1.1),
) -> None:
    """Print a ratio table comparing two sets of stats.

    *stats_a* and *stats_b* map ``row_key -> list[float]`` (durations).
    Prints ``label_a / label_b`` ratio for each row_key present in both.

    Thresholds ``(lo, hi)`` control the status tags:
      - ratio < lo  → ``✓ FAST``
      - lo ≤ ratio ≤ hi → ``✓ OK``
      - ratio > hi → ``⚠ SLOW``
    """
    lo, hi = thresholds
    all_keys = sorted(set(stats_a.keys()) & set(stats_b.keys()))

    if not all_keys:
        print(f"No paired {label_a}/{label_b} data found for comparison.")
        return

    print(f"\n{label_a} / {label_b} Ratio:")
    print("-" * 85)
    print(
        f"  {'Config':<25} | "
        f"{label_a + ' Mean (µs)':>18} | "
        f"{label_b + ' Mean (µs)':>18} | "
        f"{'Ratio':>10} | "
        f"{'Status':<10}"
    )
    print(f"  {'-' * 80}")

    for key in all_keys:
        a_mean = statistics.mean(stats_a[key])
        b_mean = statistics.mean(stats_b[key])
        ratio = a_mean / b_mean if b_mean > 0 else float("inf")

        if ratio <= lo:
            status = "✓ FAST"
        elif ratio <= hi:
            status = "✓ OK"
        else:
            status = "⚠ SLOW"

        print(
            f"  {key:<25} | "
            f"{a_mean:>18.2f} | "
            f"{b_mean:>18.2f} | "
            f"{ratio:>9.3f}x | "
            f"{status:<10}"
        )

    print()


# ---------------------------------------------------------------------------
# Default config parsing for TBE trace filenames
# ---------------------------------------------------------------------------


def _default_parse_config(filename: str) -> dict[str, str] | None:
    """Extract output_dtype and managed from common TBE trace filenames."""
    config: dict[str, str] = {}

    for managed_type in ("managed_caching", "managed", "device"):
        if f"_{managed_type}_" in filename or filename.endswith(
            f"_{managed_type}.json"
        ):
            config["managed"] = managed_type
            break

    for dtype in ("bf16", "fp16", "fp32"):
        if f"_out{dtype}_" in filename.lower() or f"out{dtype}" in filename.lower():
            config["output_dtype"] = dtype
            break

    dim_match = re.search(r"dim(\d+)", filename)
    if dim_match:
        config["dim"] = dim_match.group(1)

    if config.get("output_dtype") or config.get("managed"):
        return config
    return None


def _default_config_key(config: dict[str, str]) -> str:
    """Build a row key from the default parsed config."""
    parts = []
    if "output_dtype" in config:
        parts.append(config["output_dtype"])
    if "managed" in config:
        parts.append(config["managed"])
    if "dim" in config:
        parts.append(f"dim{config['dim']}")
    return "/".join(parts) if parts else "unknown"
