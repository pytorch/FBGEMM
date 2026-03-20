#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Shared data types for kernel trace analysis."""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from math import nan

StatsMap = dict[str, dict[str, "KernelStats"]]


@dataclass
class KernelStats:
    """Statistics for a kernel's execution times."""

    name: str
    durations_us: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.durations_us)

    @property
    def total_us(self) -> float:
        return sum(self.durations_us) if self.durations_us else 0.0

    @property
    def mean_us(self) -> float:
        return statistics.mean(self.durations_us) if self.durations_us else 0.0

    @property
    def median_us(self) -> float:
        return statistics.median(self.durations_us) if self.durations_us else 0.0

    @property
    def min_us(self) -> float:
        return min(self.durations_us) if self.durations_us else 0.0

    @property
    def max_us(self) -> float:
        return max(self.durations_us) if self.durations_us else 0.0

    @property
    def stdev_us(self) -> float:
        return (
            statistics.stdev(self.durations_us) if len(self.durations_us) > 1 else 0.0
        )

    # ------------------------------------------------------------------
    # Instance helpers
    # ------------------------------------------------------------------

    def get(self, attr: str, default: float = nan) -> float:
        """Safe attribute access returning *default* when count == 0."""
        if self.count > 0:
            return getattr(self, attr)
        return default

    def print_detail(self) -> None:
        """Print detailed stats for this KernelStats object."""
        if not self.durations_us:
            print(f"{self.name}: No data")
            return

        print(f"{self.name}:")
        print(f"  Count:  {self.count}")
        print(f"  Mean:   {self.mean_us:,.2f} us ({self.mean_us / 1000:,.4f} ms)")
        print(f"  Median: {self.median_us:,.2f} us ({self.median_us / 1000:,.4f} ms)")
        print(f"  Min:    {self.min_us:,.2f} us ({self.min_us / 1000:,.4f} ms)")
        print(f"  Max:    {self.max_us:,.2f} us ({self.max_us / 1000:,.4f} ms)")
        print(f"  Stdev:  {self.stdev_us:,.2f} us ({self.stdev_us / 1000:,.4f} ms)")
        print()

    # ------------------------------------------------------------------
    # Collection-level classmethods
    # ------------------------------------------------------------------

    @classmethod
    def aggregate(
        cls,
        kernels: dict[str, KernelStats],
        filter_pattern: re.Pattern | str | None = None,
        exclude_patterns: list[str] | None = None,
        name: str = "(aggregated)",
    ) -> KernelStats:
        """
        Aggregate multiple KernelStats into one, with optional filtering.

        Args:
            kernels: Dictionary mapping kernel names to KernelStats
            filter_pattern: If provided, only include kernels matching this pattern
                (compiled regex or string for substring match)
            exclude_patterns: If provided, exclude kernels matching any of these substrings
            name: Name for the aggregated KernelStats

        Returns:
            Aggregated KernelStats
        """
        agg = cls(name=name)
        for kname, stats in kernels.items():
            if filter_pattern is not None:
                if isinstance(filter_pattern, re.Pattern):
                    if not filter_pattern.search(kname):
                        continue
                elif isinstance(filter_pattern, str):
                    if filter_pattern not in kname:
                        continue
            if exclude_patterns:
                if any(pat in kname for pat in exclude_patterns):
                    continue
            agg.durations_us.extend(stats.durations_us)
        return agg

    @classmethod
    def aggregate_by_category(
        cls,
        kernels: dict[str, KernelStats],
        classify_fn: Callable[[str], str | None],
    ) -> dict[str, KernelStats]:
        """
        Group kernels by a user-provided classification function.

        Args:
            kernels: Dictionary mapping kernel names to KernelStats
            classify_fn: Function that takes a kernel name and returns a category
                string, or None to skip

        Returns:
            Dictionary mapping category names to aggregated KernelStats
        """
        by_cat: dict[str, KernelStats] = defaultdict(lambda: cls(name=""))
        for kname, stats in kernels.items():
            cat = classify_fn(kname)
            if cat is None:
                cat = kname[:60]
            by_cat[cat].name = cat
            by_cat[cat].durations_us.extend(stats.durations_us)
        return dict(by_cat)

    @classmethod
    def to_csv(
        cls,
        results: StatsMap,
        output_path: str,
        source_label: str = "data",
        extra_columns: list[str] | None = None,
        key_parser: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        """Export kernel stats to CSV.

        Args:
            results: {group_key: {kernel_name: KernelStats}}
            output_path: Path to write the CSV file
            source_label: Label for the source column
            extra_columns: Additional column names parsed from group_key
            key_parser: Optional callable (key) -> dict of extra column values
        """
        cls._write_csv(
            [(source_label, results)],
            output_path,
            extra_columns=extra_columns,
            key_parser=key_parser,
        )

    @classmethod
    def comparison_to_csv(
        cls,
        old_results: StatsMap,
        new_results: StatsMap,
        output_path: str,
        extra_columns: list[str] | None = None,
        key_parser: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        """Export old-vs-new comparison data to CSV.

        Args:
            old_results: {group_key: {kernel_name: KernelStats}} for old benchmark
            new_results: {group_key: {kernel_name: KernelStats}} for new benchmark
            output_path: Path to write the CSV file
            extra_columns: Additional column names parsed from group_key
            key_parser: Optional callable (key) -> dict of extra column values
        """
        cls._write_csv(
            [("old", old_results), ("new", new_results)],
            output_path,
            extra_columns=extra_columns,
            key_parser=key_parser,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _write_csv(
        cls,
        labeled_results: list[tuple[str, StatsMap]],
        output_path: str,
        extra_columns: list[str] | None = None,
        key_parser: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        """Write one or more labeled result sets to a single CSV file."""
        fieldnames = ["source"]
        if extra_columns:
            fieldnames.extend(extra_columns)
        else:
            fieldnames.append("config")
        fieldnames.extend(
            [
                "kernel_name",
                "count",
                "mean_us",
                "median_us",
                "min_us",
                "max_us",
                "stdev_us",
            ]
        )

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for source, results in labeled_results:
                for key in sorted(results.keys()):
                    for kname, stats in sorted(results[key].items()):
                        row: dict[str, object] = {
                            "source": source,
                            "kernel_name": kname,
                            "count": stats.count,
                            "mean_us": f"{stats.mean_us:.2f}",
                            "median_us": f"{stats.median_us:.2f}",
                            "min_us": f"{stats.min_us:.2f}",
                            "max_us": f"{stats.max_us:.2f}",
                            "stdev_us": f"{stats.stdev_us:.2f}",
                        }
                        if key_parser is not None:
                            row.update(key_parser(key))
                        elif extra_columns is None:
                            row["config"] = key
                        writer.writerow(row)

        print(f"CSV exported to: {output_path}")
