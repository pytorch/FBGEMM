#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Canonical per-``(config x kernel)`` CSV emitter for the benchmark
pipeline.

Inputs
------
* ``--trace-dir``: directory containing Kineto trace JSON files produced
  by one benchmark run.
* ``--config-map``: path to ``config_map.json`` that binds trace
  filenames to their structured config dict. Schema:

  .. code-block:: json

     {
       "config_columns": ["batch_size", "dim", "dtype"],
       "entries": [
         {"trace_file": "benchmark_0_..._trace_12345.json",
          "config": {"batch_size": 1024, "dim": 512, "dtype": "fp16"}},
         ...
       ]
     }

  Every entry's ``config`` dict MUST have every key listed in
  ``config_columns``. Extra keys are ignored; missing keys are rejected
  with a non-zero exit.

* ``--kernel-pattern`` (repeatable): ``name=PATTERN`` pairs describing
  which kernel(s) to extract. ``name`` is an informational label; the
  canonical CSV identifies kernels by their fully-qualified trace names.
  Patterns are interpreted via :meth:`KinetoTrace.extract_durations`
  with ``match_mode="contains"`` by default; add ``:regex`` suffix
  (``name=pat:regex``) to switch to regex matching.
* ``--output``: path to write ``stats_summary.csv``.

Behavior
--------
* For each config, the script collects matching kernel durations across
  all of that config's trace files and coalesces them via
  :meth:`KernelStats.aggregate`.
* When no matching kernel executions exist for a ``(config, kernel)``
  pair that appeared in any other config, a ``count=0`` row is still
  emitted so the report can show the dispatch gap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from fbgemm_gpu.bench.analysis.kernel_grouping import base_name_of
from fbgemm_gpu.bench.analysis.trace import KinetoTrace
from fbgemm_gpu.bench.analysis.types import KernelStats, write_config_stats_csv


def _config_tuple(cfg: dict[str, Any], columns: list[str]) -> tuple:
    return tuple(cfg[c] for c in columns)


def _parse_pattern_spec(spec: str) -> tuple[str, str, str]:
    """``name=pattern[:mode]`` -> ``(name, pattern, match_mode)``."""
    if "=" not in spec:
        raise ValueError(
            f"--kernel-pattern {spec!r} must be of form name=PATTERN[:mode]"
        )
    name, pattern = spec.split("=", 1)
    mode = "contains"
    if pattern.endswith(":regex"):
        pattern = pattern[: -len(":regex")]
        mode = "regex"
    elif pattern.endswith(":exact"):
        pattern = pattern[: -len(":exact")]
        mode = "exact"
    elif pattern.endswith(":startswith"):
        pattern = pattern[: -len(":startswith")]
        mode = "startswith"
    return name.strip(), pattern.strip(), mode


def _load_config_map(path: str) -> tuple[list[str], list[dict[str, Any]]]:
    with open(path) as f:
        data = json.load(f)
    columns = data.get("config_columns")
    entries = data.get("entries")
    if not isinstance(columns, list) or not isinstance(entries, list):
        raise ValueError(
            f"{path}: must have 'config_columns' (list) and 'entries' (list)"
        )
    for entry in entries:
        if "trace_file" not in entry or "config" not in entry:
            raise ValueError(f"{path}: each entry needs trace_file + config")
        missing = [c for c in columns if c not in entry["config"]]
        if missing:
            raise ValueError(
                f"{path}: entry {entry['trace_file']} missing config keys {missing}"
            )
    return columns, entries


def _collect_kernel_stats(
    trace_dir: str,
    config_map: list[dict[str, Any]],
    config_columns: list[str],
    patterns: list[tuple[str, str, str]],
) -> list[tuple[dict[str, Any], str, str, KernelStats]]:
    """Extract durations and bucket by ``(config_tuple, kernel_name)``.

    Returns a flat list of ``(config, kernel_base, kernel_name, stats)``
    tuples, one per ``(config, kernel_name)`` combination. Includes
    explicit ``count=0`` rows when a config did not dispatch a kernel
    that other configs did.
    """
    # bucket[cfg_tuple][kernel_name] -> list[float]
    bucket: dict[tuple, dict[str, list[float]]] = {}
    config_for: dict[tuple, dict[str, Any]] = {}
    all_kernels: set[str] = set()

    for entry in config_map:
        cfg = entry["config"]
        cfg_tuple = _config_tuple(cfg, config_columns)
        config_for[cfg_tuple] = {c: cfg[c] for c in config_columns}
        bucket.setdefault(cfg_tuple, {})

        trace_path = os.path.join(trace_dir, entry["trace_file"])
        if not os.path.exists(trace_path):
            print(f"warning: trace file not found: {trace_path}", file=sys.stderr)
            continue

        try:
            trace = KinetoTrace.from_file(trace_path)
        except Exception as e:
            print(
                f"warning: failed to parse {trace_path}: {e}",
                file=sys.stderr,
            )
            continue

        for _, pattern, match_mode in patterns:
            bucketed = trace.extract_durations(pattern, match_mode=match_mode)
            for kname, durs in bucketed.items():
                all_kernels.add(kname)
                bucket[cfg_tuple].setdefault(kname, []).extend(durs)

    rows: list[tuple[dict[str, Any], str, str, KernelStats]] = []
    for cfg_tuple, cfg_dict in config_for.items():
        kernel_durs = bucket[cfg_tuple]
        for kname in sorted(all_kernels):
            durs = kernel_durs.get(kname, [])
            stats = KernelStats(name=kname, durations_us=list(durs))
            rows.append((cfg_dict, base_name_of(kname), kname, stats))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit canonical per-(config x kernel) stats CSV.",
    )
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--config-map", required=True)
    parser.add_argument(
        "--kernel-pattern",
        action="append",
        required=True,
        metavar="NAME=PATTERN",
        help="Kernel pattern spec (repeatable). Pattern can be suffixed "
        "with :regex, :exact, or :startswith; default is contains.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        config_columns, entries = _load_config_map(args.config_map)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    patterns = [_parse_pattern_spec(s) for s in args.kernel_pattern]
    rows = _collect_kernel_stats(args.trace_dir, entries, config_columns, patterns)

    if not rows:
        print(
            "error: no (config, kernel) rows produced — check --trace-dir "
            "and --kernel-pattern",
            file=sys.stderr,
        )
        return 3

    write_config_stats_csv(rows, args.output, config_columns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
