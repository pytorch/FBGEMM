#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Unified Chrome trace parsing, kernel extraction, and statistics."""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from fbgemm_gpu.bench.analysis.types import KernelStats


class KinetoTrace:
    """Parsed Kineto / Chrome trace with kernel extraction helpers.

    Construct directly from a list of events, or use the ``from_file``
    classmethod to parse a ``.json`` / ``.json.gz`` trace file.
    """

    def __init__(
        self,
        events: list[dict[str, Any]],
        filepath: str | None = None,
    ) -> None:
        self.events = events
        self.filepath = filepath

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath: str) -> KinetoTrace:
        """Parse a Chrome trace JSON file (.json or .json.gz)."""
        filepath = str(filepath)
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rt") as f:
                data = json.load(f)
        else:
            with open(filepath, "r") as f:
                data = json.load(f)

        if isinstance(data, dict):
            if "traceEvents" in data:
                events = data["traceEvents"]
            else:
                for key in ("events", "data"):
                    if key in data:
                        events = data[key]
                        break
                else:
                    raise ValueError(f"Could not find trace events in {filepath}")
        elif isinstance(data, list):
            events = data
        else:
            raise ValueError(f"Unexpected trace format in {filepath}")

        return cls(events, filepath=filepath)

    @staticmethod
    def find_files(directory: str, recursive: bool = True) -> list[str]:
        """Find all .json and .json.gz trace files in a directory."""
        if recursive:
            files = sorted(
                glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)
                + glob.glob(os.path.join(directory, "**", "*.json.gz"), recursive=True)
            )
        else:
            files = sorted(
                glob.glob(os.path.join(directory, "*.json"))
                + glob.glob(os.path.join(directory, "*.json.gz"))
            )
        return files

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def kernel_names(self) -> set[str]:
        """Return unique kernel names (``ph=X`` events)."""
        return {
            event["name"]
            for event in self.events
            if event.get("ph") == "X" and "name" in event
        }

    def extract_durations(
        self,
        kernel_pattern: str,
        match_mode: str = "contains",
        device_type: str | None = None,
        skip_first_n: int = 0,
    ) -> dict[str, list[float]]:
        """Extract durations for kernels matching *kernel_pattern*.

        Args:
            kernel_pattern: String to match in kernel names.
            match_mode: ``"exact"`` | ``"contains"`` (default) |
                ``"startswith"`` | ``"regex"``.
            device_type: ``"cuda"`` (GPU kernel), ``"cpu"`` (CPU op), or
                ``None`` (no filtering).
            skip_first_n: Number of initial matching events to skip (warmup).

        Returns:
            Mapping of kernel name → list of durations in µs.
        """
        if device_type == "cuda":
            target_category = "kernel"
        elif device_type == "cpu":
            target_category = "cpu_op"
        else:
            target_category = None

        compiled_regex = re.compile(kernel_pattern) if match_mode == "regex" else None

        bucketed: dict[str, list[float]] = defaultdict(list)
        skip_counts: Counter[str] = Counter()

        for event in self.events:
            if event.get("ph") != "X":
                continue
            if target_category is not None and event.get("cat") != target_category:
                continue

            name = event.get("name", "")

            if match_mode == "exact":
                matches = name == kernel_pattern
            elif match_mode == "startswith":
                matches = name.startswith(kernel_pattern)
            elif match_mode == "regex":
                matches = compiled_regex.search(name) is not None
            else:
                matches = kernel_pattern in name

            if matches:
                dur = event.get("dur", 0)
                if dur > 0:
                    if skip_first_n > 0 and skip_counts[name] < skip_first_n:
                        skip_counts[name] += 1
                        continue
                    bucketed[name].append(float(dur))

        return bucketed

    def extract_stats(
        self,
        kernel_pattern: str,
        match_mode: str = "contains",
        device_type: str | None = None,
        skip_first_n: int = 0,
    ) -> dict[str, KernelStats]:
        """Extract durations and build :class:`KernelStats` in one step."""
        bucketed = self.extract_durations(
            kernel_pattern,
            match_mode=match_mode,
            device_type=device_type,
            skip_first_n=skip_first_n,
        )
        return self._build_stats(bucketed)

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    @classmethod
    def run_list_kernels(
        cls,
        dirs: list[str],
        pattern: str | None = None,
        classify_fn: Callable[[str], str | None] | None = None,
    ) -> None:
        """Print unique kernel names found across trace files in *dirs*.

        Args:
            dirs: Directories to search for trace files.
            pattern: Optional substring filter on kernel names.
            classify_fn: Optional function to tag each kernel with a category.
        """
        if not dirs:
            print(
                "Error: at least one trace directory is required for list-kernels mode",
                file=sys.stderr,
            )
            sys.exit(1)

        all_names: set[str] = set()
        for trace_dir in dirs:
            for trace_file in cls.find_files(trace_dir, recursive=True):
                try:
                    trace = cls.from_file(trace_file)
                    names = trace.kernel_names()
                    if pattern:
                        names = {n for n in names if pattern in n}
                    all_names.update(names)
                except Exception:
                    continue

        print(f"Kernels found{' (filtered)' if pattern else ''}:")
        print("-" * 80)
        for name in sorted(all_names):
            suffix = ""
            if classify_fn:
                cat = classify_fn(name)
                if cat:
                    suffix = f"  [{cat}]"
            print(f"  {name}{suffix}")
        print(f"\nTotal unique kernels: {len(all_names)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stats(
        durations_bucketed: dict[str, list[float]],
    ) -> dict[str, KernelStats]:
        """Convert a bucketed durations dict into :class:`KernelStats` objects."""
        result: dict[str, KernelStats] = {}
        for kname, durs in durations_bucketed.items():
            stats = KernelStats(name=kname)
            stats.durations_us = durs
            result[kname] = stats
        return result
