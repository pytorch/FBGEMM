#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Compute cross-commit statistics and emit ``regression_stats.json``.

This is the ONLY source of truth for comparison numbers (pct_diff,
Welch's t, Cohen's d, BH-adjusted q). Downstream agents and the report
renderer read ``regression_stats.json`` and never recompute.

Inputs
------
``--commit LABEL:PATH`` (repeatable): one per commit being compared.
  ``LABEL`` is a short name (e.g. ``parent``, ``diff``) and ``PATH`` is
  that commit's canonical ``stats_summary.csv`` produced by
  ``aggregate_config_stats``.

``--diff D12345`` (optional): the Phabricator diff identifier, recorded
  verbatim in the output JSON.

``--output PATH`` (required): where to write ``regression_stats.json``.

Exit codes
----------
* 0 — success
* 2 — CLI error
* 3 — data error (mismatched config_columns, empty rows, malformed CSV)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from fbgemm_gpu.bench.analysis.sorting import sort_rows
from fbgemm_gpu.bench.analysis.statistics import bh_adjust, cohen_d, pct_diff, welch_t_p
from fbgemm_gpu.bench.analysis.types import read_config_stats_csv

FORMAT_VERSION: str = "v1"


def _parse_commit_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"--commit {spec!r} must be LABEL:PATH")
    label, path = spec.split(":", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(f"--commit {spec!r} has empty label or path")
    return label, path


def _row_key(row: dict[str, Any], config_columns: list[str]) -> tuple:
    cfg = row["config"]
    return (
        tuple(cfg.get(c) for c in config_columns),
        row["kernel_base"],
        row["kernel_name"],
    )


def _safe_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else float("nan")
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute cross-commit statistics for benchmark reports.",
    )
    parser.add_argument(
        "--commit",
        action="append",
        required=True,
        metavar="LABEL:PATH",
        help="Commit label and CSV path (repeatable).",
    )
    parser.add_argument("--diff", default="", help="Phabricator diff id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        commits = [_parse_commit_spec(s) for s in args.commit]
    except argparse.ArgumentTypeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if len(commits) < 1:
        print("error: at least one --commit is required", file=sys.stderr)
        return 2

    per_commit_rows: list[tuple[str, str, list[dict[str, Any]]]] = []
    config_columns: list[str] | None = None
    for label, path in commits:
        try:
            cols, rows = read_config_stats_csv(path)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        if config_columns is None:
            config_columns = cols
        elif cols != config_columns:
            print(
                f"error: config_columns mismatch between commits: "
                f"{config_columns} vs {cols} (from {path})",
                file=sys.stderr,
            )
            return 3
        per_commit_rows.append((label, path, rows))

    if config_columns is None:
        print("error: no commits supplied", file=sys.stderr)
        return 3

    # Union of (config, kernel) keys across commits.
    all_keys: dict[tuple, tuple[tuple, str, str]] = {}
    for _, _, rows in per_commit_rows:
        for row in rows:
            key = _row_key(row, config_columns)
            if key not in all_keys:
                all_keys[key] = (
                    key[0],
                    row["kernel_base"],
                    row["kernel_name"],
                )

    if not all_keys:
        print("error: no data rows across any commit", file=sys.stderr)
        return 3

    # Build per-row per-commit stats.
    rows_out: list[dict[str, Any]] = []
    for key in all_keys:
        cfg_tuple, kernel_base, kernel_name = all_keys[key]
        config_dict = dict(zip(config_columns, cfg_tuple))
        per_commit_entries: list[dict[str, Any]] = []
        for label, _, rows in per_commit_rows:
            found = next(
                (r for r in rows if _row_key(r, config_columns) == key),
                None,
            )
            if found is None:
                per_commit_entries.append(
                    {
                        "label": label,
                        "n": 0,
                        "mean_us": None,
                        "stdev_us": None,
                        "dispatched": False,
                    }
                )
                continue
            n = int(found.get("count", 0) or 0)
            mean_us = found.get("mean_us")
            stdev_us = found.get("stdev_us")
            per_commit_entries.append(
                {
                    "label": label,
                    "n": n,
                    "mean_us": mean_us if n > 0 else None,
                    "stdev_us": stdev_us if n > 0 else None,
                    "dispatched": n > 0,
                }
            )

        row: dict[str, Any] = {
            "config": config_dict,
            "kernel_base": kernel_base,
            "kernel_name": kernel_name,
            "per_commit": per_commit_entries,
            "pct_diff": None,
            "welch_t": None,
            "welch_p": None,
            "welch_df": None,
            "bh_q": None,
            "cohen_d": None,
        }
        rows_out.append(row)

    # Comparison columns only populated for exactly 2 commits AND both
    # sides dispatched.
    if len(per_commit_rows) == 2:
        for row in rows_out:
            a, b = row["per_commit"][0], row["per_commit"][1]
            if not (a["dispatched"] and b["dispatched"]):
                continue
            m1 = _safe_float(a["mean_us"])
            s1 = _safe_float(a["stdev_us"])
            m2 = _safe_float(b["mean_us"])
            s2 = _safe_float(b["stdev_us"])
            if not all(math.isfinite(v) for v in (m1, s1, m2, s2)):
                continue
            row["pct_diff"] = pct_diff(m1, m2)
            t, p, df = welch_t_p(a["n"], m1, s1, b["n"], m2, s2)
            row["welch_t"] = t if math.isfinite(t) else None
            row["welch_p"] = p if math.isfinite(p) else None
            row["welch_df"] = df if math.isfinite(df) else None
            d_val = cohen_d(m1, s1, m2, s2)
            row["cohen_d"] = d_val if math.isfinite(d_val) else None

        # Apply BH per kernel_base group.
        by_base: dict[str, list[dict[str, Any]]] = {}
        for row in rows_out:
            by_base.setdefault(row["kernel_base"], []).append(row)
        for _, group in by_base.items():
            pvals: list[float] = []
            for r in group:
                p = r.get("welch_p")
                pvals.append(
                    float(p)
                    if isinstance(p, (int, float)) and math.isfinite(float(p))
                    else float("nan")
                )
            qvals = bh_adjust(pvals)
            for r, q in zip(group, qvals):
                if isinstance(q, (int, float)) and math.isfinite(float(q)):
                    r["bh_q"] = float(q)
                else:
                    r["bh_q"] = None

    rows_out = sort_rows(rows_out, config_columns)

    payload = {
        "format_version": FORMAT_VERSION,
        "diff": args.diff,
        "commits": [
            {"label": label, "source_csv": path} for label, path, _ in per_commit_rows
        ],
        "config_columns": [f"config.{c}" for c in config_columns],
        "rows": _sanitize_for_json(rows_out),
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"regression_stats.json written to: {args.output}")
    return 0


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN / Infinity floats with ``None`` recursively so the
    resulting payload is strict-JSON-serializable (``allow_nan=False``).
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    return obj


if __name__ == "__main__":
    sys.exit(main())
