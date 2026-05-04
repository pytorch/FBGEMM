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

    def summary_tuple(self) -> tuple[int, float, float, float, float, float]:
        """Return ``(count, mean_us, median_us, stdev_us, min_us, max_us)``.

        Returns all-NaN tuple when there are no durations.
        """
        if not self.durations_us:
            _nan = float("nan")
            return 0, _nan, _nan, _nan, _nan, _nan
        return (
            self.count,
            self.mean_us,
            self.median_us,
            self.stdev_us,
            self.min_us,
            self.max_us,
        )

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
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_file(
        cls,
        trace_file: str,
        kernel_pattern: str,
        *,
        match_mode: str = "contains",
        device_type: str | None = None,
        name: str | None = None,
    ) -> KernelStats:
        """Load a trace file, extract matching durations, and return aggregated stats.

        This replaces the common 5-line pattern::

            trace = KinetoTrace.from_file(path)
            bucketed = trace.extract_durations(pattern)
            durations = [d for durs in bucketed.values() for d in durs]
            stats = KernelStats(name=pattern, durations_us=durations)

        Args:
            trace_file: Path to a ``.json`` or ``.json.gz`` Chrome trace.
            kernel_pattern: Pattern passed to
                :meth:`KinetoTrace.extract_durations`.
            match_mode: ``"contains"`` | ``"exact"`` | ``"startswith"`` |
                ``"regex"``.
            device_type: ``"cuda"``, ``"cpu"``, or ``None``.
            name: Name for the returned :class:`KernelStats`.  Defaults to
                *kernel_pattern*.
        """
        from fbgemm_gpu.bench.analysis.trace import KinetoTrace

        trace = KinetoTrace.from_file(trace_file)
        bucketed = trace.extract_durations(
            kernel_pattern, match_mode=match_mode, device_type=device_type
        )
        durations = [d for durs in bucketed.values() for d in durs]
        return cls(name=name or kernel_pattern, durations_us=durations)

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


# ======================================================================
# Canonical per-(config x kernel) CSV writer for the benchmark report.
# Schema is pinned by the stability contract in build_report_html.py.
# ======================================================================


def write_config_stats_csv(
    rows: list[tuple[dict[str, object], str, str, "KernelStats"]],
    output_path: str,
    config_columns: list[str],
) -> None:
    """Write one row per ``(config_tuple, kernel_name)`` to ``output_path``.

    ``rows`` is a list of ``(config_dict, kernel_base, kernel_name, stats)``
    tuples. Each ``config_dict`` MUST contain every column listed in
    ``config_columns`` (missing entries get an empty cell).

    Column order is the v1 canonical schema:
        ``config.<col1>, ..., config.<colK>, kernel_base, kernel_name,
         count, mean_us, stdev_us, median_us, min_us, max_us``

    ``count == 0`` rows are preserved; they are the first-class "kernel
    not dispatched for this config" signal. When ``count == 0``, the stat
    cells are written as empty strings rather than ``0.00`` to avoid
    misleading the human reader.
    """
    fieldnames: list[str] = [f"config.{c}" for c in config_columns]
    fieldnames.extend(
        [
            "kernel_base",
            "kernel_name",
            "count",
            "mean_us",
            "stdev_us",
            "median_us",
            "min_us",
            "max_us",
        ]
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config_dict, kernel_base, kernel_name, stats in rows:
            out: dict[str, object] = {}
            for col in config_columns:
                val = config_dict.get(col, "")
                out[f"config.{col}"] = "" if val is None else val
            out["kernel_base"] = kernel_base
            out["kernel_name"] = kernel_name
            if stats.count == 0:
                out["count"] = 0
                out["mean_us"] = ""
                out["stdev_us"] = ""
                out["median_us"] = ""
                out["min_us"] = ""
                out["max_us"] = ""
            else:
                out["count"] = stats.count
                out["mean_us"] = f"{stats.mean_us:.2f}"
                out["stdev_us"] = f"{stats.stdev_us:.2f}"
                out["median_us"] = f"{stats.median_us:.2f}"
                out["min_us"] = f"{stats.min_us:.2f}"
                out["max_us"] = f"{stats.max_us:.2f}"
            writer.writerow(out)

    print(f"Canonical stats CSV exported to: {output_path}")


CANONICAL_CSV_STAT_COLUMNS: tuple[str, ...] = (
    "kernel_base",
    "kernel_name",
    "count",
    "mean_us",
    "stdev_us",
    "median_us",
    "min_us",
    "max_us",
)


def read_config_stats_csv(
    csv_path: str,
) -> tuple[list[str], list[dict[str, object]]]:
    """Parse a canonical ``stats_summary.csv`` file.

    Returns ``(config_columns, rows)`` where ``config_columns`` is the
    list of column names (sans the ``config.`` prefix, in file order) and
    each row is a dict with keys ``config`` (nested dict of config values
    with numeric strings coerced to int/float), ``kernel_base``,
    ``kernel_name``, plus the stat columns (``count`` as int, others as
    float or ``None`` when the cell is empty / count is zero).
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        config_columns: list[str] = []
        for name in header:
            if name.startswith("config."):
                config_columns.append(name[len("config.") :])

        for required in CANONICAL_CSV_STAT_COLUMNS:
            if required not in header:
                raise ValueError(
                    f"CSV {csv_path} missing required column: {required!r} "
                    f"(header={header})"
                )

        rows: list[dict[str, object]] = []
        for raw in reader:
            config: dict[str, object] = {}
            for c in config_columns:
                config[c] = _coerce_scalar(raw.get(f"config.{c}", ""))
            row: dict[str, object] = {
                "config": config,
                "kernel_base": raw.get("kernel_base", "") or "",
                "kernel_name": raw.get("kernel_name", "") or "",
                "count": int(raw.get("count", "0") or 0),
            }
            for stat in ("mean_us", "stdev_us", "median_us", "min_us", "max_us"):
                cell = raw.get(stat, "")
                row[stat] = float(cell) if cell not in ("", None) else None
            rows.append(row)

    return config_columns, rows


def _coerce_scalar(value: str) -> object:
    """Coerce a CSV cell to int/float when possible, else return the
    stripped string (empty stays empty). Booleans-as-strings stay
    strings."""
    if value is None:
        return ""
    stripped = value.strip()
    if stripped == "":
        return ""
    try:
        if "." in stripped or "e" in stripped.lower():
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped
